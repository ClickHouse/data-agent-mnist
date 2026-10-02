"""Does surfacing Jev's ranked tables to the agent help it answer?

Builds a "start with these tables" note from Jev's schema-retrieval ranking (top-k by
P(yes)) and either prepends it to the system prompt or delivers it in-loop only when
the agent diverges onto a low-ranked table, runs the candidate at the same turn budget,
grades with the judge panel, and compares to the baseline graded on the same ruler. The
note prioritizes, it does not prune: the full schema stays, so a question that needs a
lower-ranked table can still reach it.

Jev is reached through the native TypeSafe API (see schema_retrieval.jev_client): set
TYPESAFE_API_KEY, and TYPESAFE_BASE_URL to override the host.

    export TYPESAFE_API_KEY=...
    uv run --with typesafe-sdk schema_retrieval_agent.py --mode divergence --fresh-baseline \
        --db-path examples/saas/warehouse --system-prompt examples/saas/schema.md \
        --probe-table marts.usage_daily --snapshot-column day \
        --data-dir examples/saas/out --table-namespaces marts,crm --models gpt-4.1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from pathlib import Path

import schema_retrieval as sr  # rank_tables, load_universe, gold_tables, table_regexes

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
PASS = "pass"
RECOVERABLE = {"fail", "loss", "tie"}


def build_note(ranked: list[str], k: int) -> str:
    top = ranked[:k]
    lines = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(top))
    return ("The tables most likely relevant to this question, ranked, are:\n" + lines +
            "\nStart from these. The full schema above still applies and other tables may "
            "be needed, so do not treat this as the complete set.")


def make_divergence_gate(ranked: list[str], confident: set[str], fromjoin_re: re.Pattern, *,
                         topk: int = 5, style: str = "contrastive", cap: int = 1,
                         fire_after: int = 1, trigger_topk: int | None = None):
    """Fire in-loop when the agent looks off track on tables, a runtime proxy for reaching
    for the wrong one. It fires once the agent has issued `fire_after` queries and touched
    NO table in Jev's confident set across the whole trajectory so far; `fire_after`=1 fires
    on the first below-set query, higher values wait for persistent divergence and cut the
    false positives on capable models that briefly query a correct lower-ranked table. The
    hint is composed at fire time: `contrastive` names the table the query read and the
    ranked table to use instead; `list` gives the ranked top-k note. Fires at most `cap`."""
    fires = [0]
    n_queries = [0]
    n_off_track = [0]
    queried_all: set[str] = set()
    hints: list[str] = []

    def _compose(qt: set[str]) -> str:
        if style == "contrastive":
            queried = ", ".join(sorted(qt))
            rec = ", ".join(ranked[:3])
            return (f"Your last query read {queried}. For this question the most relevant "
                    f"tables, ranked, are: {rec}. Check whether {queried} holds the asked-for "
                    f"data; if not, query {ranked[0]} instead. The full schema still applies.")
        lines = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(ranked[:topk]))
        return ("The tables most likely relevant to this question, ranked, are:\n" + lines +
                "\nStart from these. The full schema above still applies.")

    def gate(sql: str, result: str | None = None) -> str | None:
        if fires[0] >= cap:
            return None
        qt = sr.gold_tables(sql, fromjoin_re)     # tables this query reads
        if not qt:
            return None
        n_queries[0] += 1
        queried_all.update(qt)
        # off_track: default is the confident-set rule (touched no P>=min_p table anywhere so far).
        # `trigger_topk` is the wider sensitivity knob: fire when THIS query missed Jev's top-N
        # ranked tables. Smaller N fires more. It keeps the contrastive note well-formed, since a
        # miss of the top-N means the query did not read ranked[0], so "query ranked[0] instead" holds.
        if trigger_topk is not None:
            off_track = not (qt & set(ranked[:trigger_topk]))
        else:
            off_track = not (queried_all & confident)
        # `fire_after` counts off-track queries. Under the confident-set rule every query so far
        # is off track whenever this one is (the rule is cumulative), so the two counts agree;
        # under the per-query top-N rule an on-track query must not count toward firing.
        n_off_track[0] += off_track
        if n_off_track[0] >= fire_after and off_track:
            fires[0] += 1
            h = _compose(qt)
            hints.append(h)
            return h
        return None

    gate.hints = hints                            # type: ignore[attr-defined]
    return gate


def load_annotated(data_dir: Path) -> dict[str, dict]:
    gt = {}
    for line in (data_dir / "annotated.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        if a.get("excluded") or not a.get("gt_results"):
            continue
        gt[a["trace_id"]] = a
    return gt


def load_baseline(data_dir: Path) -> dict[str, dict]:
    base = {}
    results = data_dir / "results.jsonl"
    if not results.exists():
        return base
    for line in results.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        base[r["trace_id"]] = {m: (c or {}) for m, c in (r.get("candidates") or {}).items()}
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "examples/saas/out",
                    help="dir holding annotated.jsonl (and results.jsonl for the recorded baseline)")
    ap.add_argument("--db-path", type=Path, default=HERE / "examples/saas/warehouse",
                    help="chDB warehouse for the agent's run_select_query tool")
    ap.add_argument("--system-prompt", type=Path, default=HERE / "examples/saas/schema.md",
                    help="schema prompt describing the warehouse")
    ap.add_argument("--probe-table", default="marts.usage_daily",
                    help="fact table used to check the DB is populated and derive the snapshot date")
    ap.add_argument("--snapshot-column", default="day",
                    help="date/datetime column on the probe table")
    ap.add_argument("--table-namespaces", default="marts,crm",
                    help="comma-separated schema namespaces for the retrieval universe and gold-table parse")
    ap.add_argument("--models", default="gpt-4.1")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mode", choices=("prepend", "divergence"), default="prepend",
                    help="prepend: put the top-k note in the system prompt up front (fires on "
                         "every run). divergence: deliver the note in-loop only when the agent "
                         "queries a table outside Jev's confident set, a runtime proxy for a "
                         "table-miss that needs no gold.")
    ap.add_argument("--min-p", type=float, default=0.5,
                    help="divergence mode: a table is in Jev's confident set at P(yes) >= this")
    ap.add_argument("--hint-style", choices=("list", "contrastive"), default="contrastive",
                    help="divergence mode: list = the ranked top-k note; contrastive = name the "
                         "table the query read and the ranked table to use instead")
    ap.add_argument("--trigger-topk", type=int, default=None,
                    help="wider divergence trigger: fire when a query missed Jev's top-N ranked tables "
                         "(per query). Smaller N fires more. Default None keeps the confident-set (P>=min_p) rule.")
    ap.add_argument("--fire-after", type=int, default=1,
                    help="divergence mode: fire only after the agent has issued this many queries "
                         "without touching a confident table (persistently off track). 1 fires on "
                         "the first below-set query; 2+ cuts false positives on capable models")
    ap.add_argument("--fresh-baseline", action="store_true",
                    help="run the baseline arm live instead of reading a recorded results.jsonl, "
                         "so a corpus with no recorded baseline can be measured. Forces population=all.")
    ap.add_argument("--population", choices=("all", "missed-gold"), default="missed-gold",
                    help="missed-gold: only cells whose recorded baseline failed and never "
                         "queried a gold table (the note can plausibly help). all: every cell "
                         "with a usable baseline (a blanket A/B, noisier).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent (question, model) cells in flight. The board eval "
                         "(06_eval) runs single-arm at 8; this paired eval does two runs "
                         "per cell, so ~16 matches board wall-clock. Watch shared rate limits.")
    ap.add_argument("--resume", action="store_true",
                    help="skip (question, model) cells already present in --out and append, "
                         "so a killed or throttled run continues instead of restarting")
    ap.add_argument("--gemini-gateway-judge", default=None,
                    help="optional: route the google judge seat through a gateway model id instead "
                         "of direct Vertex (for environments where direct Vertex is unreachable)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "schema_retrieval_agent.jsonl")
    ap.add_argument("--report", type=Path, default=OUT_DIR / "schema_retrieval_agent.md")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("set TYPESAFE_API_KEY in the environment")

    import bench
    # Optionally route the google judge seat through a gateway model instead of direct Vertex.
    if args.gemini_gateway_judge:
        _n = "gemini-gateway"
        _seats = {p: list(ms) for p, ms in bench.JUDGE_SEATS.items()
                  if p not in ("gemini", "google")}
        _seats["gateway"] = [_n]
        bench.JUDGE_SEATS = _seats
        bench.JUDGE_MODEL_IDS = {**bench.JUDGE_MODEL_IDS, _n: args.gemini_gateway_judge}
        bench.JUDGE_PROVIDER = {m: p for p, ms in _seats.items() for m in ms}
        print(f"judge panel: google seat via gateway as {args.gemini_gateway_judge!r} "
              f"(seats: {', '.join(sorted(_seats))})")

    from warehouse import Warehouse
    from typesafe_sdk import TypeSafeClient, Noul

    namespaces = [n.strip() for n in args.table_namespaces.split(",") if n.strip()]
    table_re, fromjoin_re = sr.table_regexes(namespaces)
    models = {m: bench.ALL_CANDIDATES[m] for m in args.models.split(",")}
    wh = Warehouse(db_path=args.db_path, system_prompt_path=args.system_prompt,
                   probe_table=args.probe_table, snapshot_column=args.snapshot_column)
    system_prompt = wh.system_prompt()
    universe = sr.load_universe(system_prompt, table_re)
    if not universe:
        raise SystemExit("empty table universe: check --table-namespaces against the schema prompt")
    gt = load_annotated(args.data_dir)
    baseline = load_baseline(args.data_dir)
    if not args.fresh_baseline and not baseline:
        raise SystemExit(f"no results.jsonl in {args.data_dir}; pass --fresh-baseline to run the "
                         "baseline arm live")
    print(f"universe {len(universe)} tables; budget {bench.MAX_TURNS} turns; data {args.data_dir}",
          flush=True)

    if args.fresh_baseline:
        args.population = "all"       # no recorded baseline to select a missed-gold subset from
    pairs = []
    for m in models:
        for tid, q in gt.items():
            if args.fresh_baseline:
                pairs.append((tid, m))       # baseline is run live in run_one
                continue
            cell = baseline.get(tid, {}).get(m)
            if not cell:
                continue
            rec = (cell.get("result_score") or {}).get("outcome")
            if rec == "error":
                continue
            if not (cell.get("sql_results") or (cell.get("final_answer") or "").strip()):
                continue
            if args.population == "missed-gold":
                gold = sr.gold_tables(q.get("gt_sql") or [], fromjoin_re)
                base_tables = sr.gold_tables(cell.get("sqls") or [], fromjoin_re)
                if rec == PASS or gold.issubset(base_tables):
                    continue      # only baseline failures that skipped a needed table
            pairs.append((tid, m))
    # Interleave round-robin by model so all models advance together, matching 06_eval,
    # instead of completing one model's cells before the next model starts.
    if len({m for _, m in pairs}) > 1:
        grp: dict[str, list] = {}
        for p in pairs:
            grp.setdefault(p[1], []).append(p)
        pairs = [p for col in zip_longest(*grp.values()) for p in col if p is not None]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"To run: {len(pairs)} (question, model) pairs [population={args.population}"
          f"{', fresh baseline' if args.fresh_baseline else ''}]", flush=True)

    def grade(q, m, results, answer, error) -> str:
        if str(error or "").startswith("max turns") and not (answer or "").strip():
            return "fail"
        return bench.judge_panel(q["nl_question"], q.get("gt_results", []),
                                 q.get("gt_answer", ""), results, answer, m)["outcome"]

    rank_cache: dict = {}                 # trace_id -> (ranked, scores) or ("error", msg); the
    rank_ready: dict = {}                 # ranking is model-independent, so it is computed once
    rank_lock = threading.Lock()          # per question: the first worker to claim a trace_id
                                          # computes it, later workers wait on its Event

    def run_one(item):
        tid, m = item
        q = gt[tid]
        cell = None if args.fresh_baseline else baseline[tid][m]
        # The Jev ranking depends only on the question, so it is computed once per trace_id
        # and shared by every model's cell. Workers on the same question interleave (round
        # robin by model), so the claim is made under the lock: the first worker computes,
        # the others wait on the Event and read the same ranking. TypeSafe can return a
        # transient 403/429/5xx under concurrency that the SDK does not retry, so the owner
        # retries with backoff; if it stays down, every cell for that question becomes an
        # error cell rather than crashing the pool, and --resume re-runs them.
        with rank_lock:
            ready = rank_ready.get(tid)
            owner = ready is None
            if owner:
                ready = rank_ready[tid] = threading.Event()
        if owner:
            outcome = None
            for attempt in range(4):
                try:
                    with sr.jev_client(TypeSafeClient) as c:
                        outcome = sr.rank_tables(c, Noul, q["nl_question"], system_prompt, universe)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        outcome = ("error", f"rank: {type(e).__name__}: {str(e)[:150]}")
                    else:
                        time.sleep(3 * (attempt + 1))
            with rank_lock:
                rank_cache[tid] = outcome
            ready.set()
        else:
            ready.wait()
            with rank_lock:
                outcome = rank_cache[tid]
        if outcome[0] == "error":
            return {"trace_id": tid, "model": m, "fired": 0,
                    "baseline_outcome": "error", "gated_outcome": "error",
                    "cell_error": outcome[1]}
        ranked, scores = outcome
        gate = None
        if args.mode == "divergence":
            confident = {t for t, p in scores.items() if p >= args.min_p} or {ranked[0]}
            gate = make_divergence_gate(ranked, confident, fromjoin_re, topk=args.topk,
                                        style=args.hint_style, fire_after=args.fire_after,
                                        trigger_topk=args.trigger_topk)
            prompt = system_prompt                    # note is delivered in-loop, not up front
        else:
            prompt = build_note(ranked, args.topk) + "\n\n" + system_prompt   # prepend the note
        try:
            replay = bench.run_candidate(q["nl_question"], m, models[m], wh.query, prompt, gate=gate)
        except Exception:  # noqa: BLE001
            replay = {"sqls": [], "sql_results": [], "final_answer": "", "error": "candidate_error"}
        fired = len(getattr(gate, "hints", [])) if gate is not None else 1
        try:
            gated = ("error" if replay.get("error") == "candidate_error"
                     else grade(q, m, replay.get("sql_results", []),
                                replay.get("final_answer", ""), replay.get("error")))
        except Exception:  # noqa: BLE001
            gated = "error"
        if args.fresh_baseline:
            try:
                base_run = bench.run_candidate(q["nl_question"], m, models[m], wh.query, system_prompt)
            except Exception:  # noqa: BLE001
                base_run = {"sqls": [], "sql_results": [], "final_answer": "", "error": "candidate_error"}
            base_sqls = base_run.get("sqls") or []
            base_results, base_answer, base_err = (base_run.get("sql_results", []),
                                                   base_run.get("final_answer", ""), base_run.get("error"))
            recorded = None
        else:
            base_run = cell
            base_sqls = cell.get("sqls") or []
            base_results, base_answer, base_err = (cell.get("sql_results") or [],
                                                   cell.get("final_answer", ""), cell.get("error"))
            recorded = (cell.get("result_score") or {}).get("outcome")
        try:
            base = ("error" if base_err == "candidate_error"
                    else grade(q, m, base_results, base_answer, base_err))
        except Exception:  # noqa: BLE001
            base = "error"
        gold = sr.gold_tables(q.get("gt_sql") or [], fromjoin_re)
        base_tables = sr.gold_tables(base_sqls, fromjoin_re)
        gated_tables = sr.gold_tables(replay.get("sqls") or [], fromjoin_re)
        missed = sorted(gold - base_tables)                     # gold tables the baseline skipped
        return {"trace_id": tid, "model": m, "topk": ranked[: args.topk], "fired": fired,
                "gold": sorted(gold), "missed_gold": missed,
                "missed_now_queried": sorted(set(missed) & gated_tables),  # note worked mechanically
                "note_covered_missed": sorted(set(missed) & set(ranked[: args.topk])),
                "baseline_outcome": base, "gated_outcome": gated,
                "recorded_baseline": recorded,
                "gated_turns": replay.get("turns"), "gated_latency": replay.get("latency"),
                "gated_usage": replay.get("usage"),
                "baseline_turns": base_run.get("turns"), "baseline_latency": base_run.get("latency"),
                "baseline_usage": base_run.get("usage")}

    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def _is_err(r):
        # A cell with an errored arm carries no paired measurement, so it is re-run. This is
        # the same rule summarize_gate_boards.py uses to drop cells from the board statistics.
        return "error" in (r.get("gated_outcome"), r.get("baseline_outcome"))
    resume = args.resume and args.out.exists()
    if resume:
        best = {}                       # (tid, m) -> row, a non-error row wins over an error row
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["trace_id"], r["model"])
            cur = best.get(key)
            if cur is None or (_is_err(cur) and not _is_err(r)):
                best[key] = r
        good = {k: r for k, r in best.items() if not _is_err(r)}
        rows = list(good.values())      # keep only graded cells; error cells get re-run
        already = set(good)
        pairs = [p for p in pairs if (p[0], p[1]) not in already]
        print(f"resume: {len(already)} graded cells, {len(pairs)} to (re)run", flush=True)
    with args.out.open("a" if resume else "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, p) for p in pairs]
        for done, f in enumerate(as_completed(futs), 1):
            r = f.result()
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            rows.append(r)
            if done % 20 == 0:
                print(f"[{done}/{len(pairs)}] run+graded", flush=True)

    report = summarize(rows, list(models), args.topk, bench.MAX_TURNS)
    args.report.write_text(report)
    print("\n" + report)
    print(f"wrote {args.out} and {args.report}")


def summarize(rows, models, topk, max_turns) -> str:
    out = [f"# Schema-retrieval prioritization vs baseline (top-{topk}, {max_turns} turns)", "",
           "Each cell is re-sampled, so a per-run flip carries the sampler's noise. When the "
           "note does not fire on a cell (divergence mode), that cell is a plain re-sample, so "
           "only the fired-cell counts are attributable to the note.", ""]
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        if not mr:
            continue
        bp = sum(r["baseline_outcome"] == PASS for r in mr)
        gp = sum(r["gated_outcome"] == PASS for r in mr)
        rec = sum(r["baseline_outcome"] in RECOVERABLE and r["gated_outcome"] == PASS for r in mr)
        reg = sum(r["baseline_outcome"] == PASS and r["gated_outcome"] in RECOVERABLE for r in mr)
        drift = sum(r["recorded_baseline"] is not None
                    and r["recorded_baseline"] != r["baseline_outcome"] for r in mr)
        fired = [r for r in mr if r.get("fired")]
        rec_f = sum(r["baseline_outcome"] in RECOVERABLE and r["gated_outcome"] == PASS for r in fired)
        reg_f = sum(r["baseline_outcome"] == PASS and r["gated_outcome"] in RECOVERABLE for r in fired)
        out += [f"## {m}", "",
                f"- cells: {len(mr)}; note fired on {len(fired)} ({len(fired)/len(mr):.0%})",
                f"- baseline pass {bp}/{len(mr)} ({bp/len(mr):.1%}) -> prioritized "
                f"{gp}/{len(mr)} ({gp/len(mr):.1%}); net {gp-bp:+d}",
                f"- recovered {rec}, regressed {reg} (on fired cells: {rec_f} / {reg_f})",
                f"- baseline re-grade drift vs recorded: {drift}/{len(mr)}", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
