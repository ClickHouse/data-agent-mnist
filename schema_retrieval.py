"""Jev schema-retrieval eval: rank a warehouse's tables by relevance to a question.

Agent-free. For each question the warehouse tables are ranked by how likely each is
needed, using Jev (TypeSafe's System One model) in yes/no mode (one Noul per table,
rank by P(yes)), and the ranking is scored against the tables the gold SQL actually
reads (nDCG@10, recall@k, all-required@k). No SQL is executed and no gold answer is
used at ranking time. Writes only to the output directory.

Jev is reached through the native TypeSafe API. Set TYPESAFE_API_KEY; the SDK
(`typesafe-sdk`) is an optional extra, so pull it in with `uv run --with typesafe-sdk`
or `uv sync --extra jev`. TYPESAFE_BASE_URL overrides the host when you reach Jev
through a gateway instead of the public API.

    export TYPESAFE_API_KEY=...
    export DAM_DATA_ROOT=$PWD/examples/saas/out
    uv run --with typesafe-sdk schema_retrieval.py \
        --corpus . --system-prompt examples/saas/schema.md --table-namespaces marts,crm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


def table_regexes(namespaces: list[str]) -> tuple[re.Pattern, re.Pattern]:
    """Compile the schema-qualified table matchers for a warehouse's namespaces.

    The retrieval universe is the set of `<namespace>.<table>` names the schema prompt
    declares; gold tables are the ones a `FROM`/`JOIN` reaches. The namespaces are a
    parameter so this names no warehouse: the board passes its layer prefixes, the saas
    example passes `marts,crm` (cf. 09_dds_analysis.py --layer-marker)."""
    ns = "|".join(re.escape(n) for n in namespaces)
    table = re.compile(rf"\b((?:{ns})\.[a-z_][a-z0-9_]*)", re.I)
    fromjoin = re.compile(rf"(?:from|join)\s+((?:{ns})\.[a-z_][a-z0-9_]*)", re.I)
    return table, fromjoin


def load_universe(schema_text: str, table_re: re.Pattern) -> list[str]:
    """The distinct tables the system prompt declares (the retrieval universe)."""
    return sorted({m.lower() for m in table_re.findall(schema_text)})


# Query-construction prose that does not help decide which table is relevant. The
# preamble and the point-in-time/time-series rules sit before the first table (cut by
# start), the trailing guidance sits after the last table (cut by end), sql examples
# are fenced, and these prefixes catch the rest.
_GUIDANCE_PREFIXES = ("To get", "For monthly", "Standard queries", "- Total users",
                      "- Active users", "- New users", "An org can", "Always take the latest")


def schema_only(text: str) -> str:
    """Just the table and column definitions: drop the preamble, the query-construction
    rules, the sql examples, and the trailing guidance, keeping table headers and column
    lists. The doc's 'send only the fields the question needs'."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.lstrip().startswith("Table `")), 0)
    end = next((i for i, l in enumerate(lines) if l.strip().startswith("**MRR guidance")), len(lines))
    kept, in_fence = [], False
    for l in lines[start:end]:
        s = l.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or any(s.startswith(p) for p in _GUIDANCE_PREFIXES):
            continue
        kept.append(l)
    return "\n".join(kept).strip()


def gold_tables(gt_sql, fromjoin_re: re.Pattern) -> set[str]:
    """Tables the gold SQL reads. gt_sql is a list of statements (or a string)."""
    s = gt_sql if isinstance(gt_sql, str) else " ".join(gt_sql)
    return {m.lower() for m in fromjoin_re.findall(s)}


def load_questions(data_dir: Path, fromjoin_re: re.Pattern) -> list[dict]:
    """Usable questions with a gold table set (drops excluded and empty-gold)."""
    out = []
    for line in (data_dir / "annotated.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        if a.get("excluded") or not a.get("gt_sql"):
            continue
        gold = gold_tables(a["gt_sql"], fromjoin_re)
        if gold:
            out.append({"trace_id": a["trace_id"], "nl_question": a["nl_question"], "gold": gold})
    return out


def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = _dcg([1.0 if t in gold else 0.0 for t in ranked[:k]])
    idcg = _dcg([1.0] * min(len(gold), k))
    return dcg / idcg if idcg else 0.0


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def jev_client(client_cls):
    """A TypeSafeClient for Jev, reached through the native TypeSafe API.

    The key is read from TYPESAFE_API_KEY and never printed. TYPESAFE_BASE_URL overrides
    the host (None → the SDK default, api.typesafe.ai), which is how an internal gateway
    that serves System One is pointed at without naming it here.
    """
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("set TYPESAFE_API_KEY in the environment")
    os.environ.setdefault("TYPESAFE_API_KEY", key)  # the SDK reads this
    return client_cls(api_key=key, base_url=os.environ.get("TYPESAFE_BASE_URL"))


def rank_tables(client, noul_cls, question: str, schema: str, universe: list[str]):
    """One Jev call: a yes/no Noul per table, ranked by P(yes). Table names carry dots,
    so the question keys are positional and mapped back."""
    keymap = {f"t{i}": t for i, t in enumerate(universe)}
    questions = {k: noul_cls(instructions=f"Is the table `{t}` needed to answer the "
                             "question? The full warehouse schema is in the state.")
                 for k, t in keymap.items()}
    resp = client.system_one(state={"question": question, "schema": schema}, questions=questions)
    scored = [(keymap[k], float(getattr(resp.nouls[k], "noul", 0.0) or 0.0)) for k in keymap]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in scored], {t: round(p, 4) for t, p in scored}


def summarize(rows: list[dict], universe: int, corpus: str) -> str:
    n = len(rows)
    ks = range(1, 11)
    mean_ndcg = sum(r["ndcg@10"] for r in rows) / n
    lat = sorted(r["latency_s"] for r in rows)
    p90 = lat[min(len(lat) - 1, int(0.9 * len(lat)))]
    out = [f"# Jev schema retrieval on {corpus}", "",
           f"{n} questions, universe of {universe} tables, Jev yes/no ranking.", "",
           f"- mean nDCG@10: {mean_ndcg:.3f}",
           f"- p90 latency: {p90:.1f}s (median {lat[len(lat)//2]:.1f}s)",
           "- recall@k (share of gold tables in the top k):"]
    for k in ks:
        out.append(f"    - @{k}: {sum(r[f'recall@{k}'] for r in rows)/n:.3f}")
    # all-required@k: the share of questions with EVERY gold table in the top k. This is
    # the metric that matters for a prioritization note: it says how deep the note must go
    # to surface a question's whole table set.
    out.append("- all-required@k (share of questions with every gold table in the top k):")
    for k in ks:
        allreq = sum(set(r["gold"]).issubset(set(r["ranked"][:k])) for r in rows) / n
        out.append(f"    - @{k}: {allreq:.3f}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=".",
                    help="corpus dir under the data root (annotated.jsonl lives there)")
    ap.add_argument("--system-prompt", type=Path, default=HERE / "examples/saas/schema.md",
                    help="the schema prompt describing the warehouse; its tables are the "
                         "retrieval universe. Defaults to the saas example.")
    ap.add_argument("--table-namespaces", default="marts,crm",
                    help="comma-separated schema namespaces whose <ns>.<table> names form "
                         "the universe (e.g. marts,crm for the saas example)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-schema-chars", type=int, default=16000)
    ap.add_argument("--schema-only", action="store_true",
                    help="pass Jev only the table/column definitions, dropping the query rules, "
                         "sql examples, and guidance prose (state trim)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "schema_retrieval.jsonl")
    ap.add_argument("--report", type=Path, default=OUT_DIR / "schema_retrieval.md")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("set TYPESAFE_API_KEY in the environment")
    from typesafe_sdk import TypeSafeClient, Noul

    from paths import DATA  # DAM_DATA_ROOT or the default
    table_re, fromjoin_re = table_regexes([n.strip() for n in args.table_namespaces.split(",") if n.strip()])
    full = args.system_prompt.read_text()
    schema = schema_only(full) if args.schema_only else full[: args.max_schema_chars]
    universe = load_universe(full, table_re)
    print(f"state: {'schema-only' if args.schema_only else 'full'} ({len(schema)} chars)", flush=True)
    questions = load_questions(DATA / args.corpus, fromjoin_re)
    if args.limit:
        questions = questions[: args.limit]
    print(f"universe {len(universe)} tables; {len(questions)} questions; corpus {args.corpus}",
          flush=True)
    if not universe:
        raise SystemExit("empty table universe: check --table-namespaces against the schema prompt")

    def one(q: dict) -> dict:
        t0 = time.time()
        with jev_client(TypeSafeClient) as c:
            ranked, scores = rank_tables(c, Noul, q["nl_question"], schema, universe)
        gold = q["gold"]
        rec = {f"recall@{k}": recall_at(ranked, gold, k) for k in range(1, 11)}
        return {"trace_id": q["trace_id"], "gold": sorted(gold), "ranked": ranked,
                "scores": scores, "latency_s": round(time.time() - t0, 2),
                "ndcg@10": ndcg_at(ranked, gold, 10), **rec}

    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, q) for q in questions]
        for done, f in enumerate(as_completed(futs), 1):
            r = f.result()
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            rows.append(r)
            if done % 20 == 0:
                print(f"[{done}/{len(questions)}] ranked", flush=True)

    report = summarize(rows, len(universe), args.corpus)
    args.report.write_text(report)
    print("\n" + report)
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
