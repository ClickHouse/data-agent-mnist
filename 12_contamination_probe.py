"""
Contamination probe: is the benchmark in any model's training data?

Two instruments, plus a positive control that validates them:

  1. Completion probe (guided instruction, Golchin & Surdeanu 2023): show a model
     the first ~60% of a benchmark question with dataset framing and ask it to
     complete the question EXACTLY. Overlap between the generation and the true
     tail (word-level similarity) measures memorisation, not skill — the tail of
     a specific user's question is not derivable from competence.
  2. Entity-recovery probe: the synthetic warehouse's entities are Faker draws
     with no world-knowledge prior, so asking for e.g. an organisation's
     `organization__id` can only be answered from memorisation. Exact-match
     recovery above chance is contamination evidence unique to synthetic data.
  3. Positive control: the identical completion probe over Spider training
     questions (public since 2018, demonstrably memorised) — so a null result on
     our items is evidence of absence, not of a weak probe.

Data governance: probes run ONLY against the provider endpoints already
configured in bench.py — the same endpoints every question transited during the
eval — so the probe adds no incremental disclosure. The Spider control is public
data. No other service is contacted.

    uv run --with datasets 12_contamination_probe.py [--models a,b] [--sample 50]
        [--entities 25] [--no-control] [--out PATH]

Resume-safe: (probe, item_id, model) triples already in the output are skipped.
"""
import argparse
import difflib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from paths import DATA
SYNTH     = DATA / "text2sqlbench-synthetic"
OUT       = SYNTH / "contamination_probe.jsonl"
SEED      = 42

COMPLETION_PROMPT = (
    'The following is the beginning of a question from the "<<DATASET>>" text-to-SQL '
    "benchmark. Complete the question with its exact continuation as it appears in "
    "the benchmark. Reply with ONLY the continuation text, nothing else.\n\n"
    "Question (beginning): <<PREFIX>>")

# The entity probe needs one table holding a name and a stable id, which is a
# property every warehouse has and not one only ours has. Table and columns are
# configuration so the probe travels; the defaults are the board's.
ENTITY_TABLE  = "dbt_dds.dim_organization_current"
ENTITY_NAME   = "organization__name"
ENTITY_ID     = "organization__id"

# NOTE: the entity prompt said "contains an organization named" and "that
# organization's" before the table became configurable. Earlier runs used that
# wording, so a re-run is not strictly comparable to them.
ENTITY_PROMPT = (
    'In the "<<DATASET>>" text-to-SQL benchmark, the synthetic warehouse table '
    '<<TABLE>> contains a record named "<<NAME>>". '
    "What is that record's <<IDCOL>> in the benchmark's warehouse? "
    "Reply with ONLY the id.")


def _complete(model_key: str, prompt: str, max_tokens: int = 300) -> str:
    """Single-shot completion through each candidate's provider client.

    Routing matches the eval with one deliberate exception: gpt-5.6 is probed via
    chat/completions rather than the Responses API. The eval's Responses-only routing
    exists because chat/completions rejects FUNCTION TOOLS alongside its default
    reasoning effort (bench.OPENAI_RESPONSES_ONLY); the probe sends no tools, so
    chat/completions runs it with reasoning intact, and the memorisation measurement
    does not depend on the API surface."""
    model_id = bench.ALL_CANDIDATES[model_key]
    if model_key in bench.MANTLE_RESPONSES_CANDIDATES:
        # Gemma 4 (bedrock-mantle /openai/v1): chat/completions accepts no tools here
        # and the probe sends none, but reasoning must be off for a bare completion —
        # same treatment as qwen3p8-max below (the probe measures completion, not
        # deliberation). max_tokens is rejected; the newer param is required.
        resp = bench.retry(lambda: bench.mantle_openai_client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max(max_tokens, 2048), reasoning_effort="none",
            temperature=0,
        ), what=f"probe.completions[{model_key}]")
        choice = resp.choices[0]
        text = ((choice.message.content if choice.message else "") or "").strip()
        if not text and choice.finish_reason == "length":
            raise RuntimeError(f"probe truncated before any visible output (finish=length, {model_id})")
        return text
    if model_key in bench.CANDIDATES or model_key in bench.MANTLE_CANDIDATES:
        if model_key in bench.CANDIDATES:  # Bedrock converse
            resp = bench.retry(lambda: bench.bedrock.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens},  # temp deprecated on Opus 4.8+
            ), what=f"probe.converse[{model_key}]")
            return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()
        client = bench.mantle_client
    elif model_key in bench.OPENAI_CANDIDATES:
        client = bench.openai_client
    elif model_key in bench.GEMINI_CANDIDATES:
        client = bench.gemini_global_client if model_key in bench.GEMINI_GLOBAL else bench.gemini_client
    elif model_key in bench.FIREWORKS_CANDIDATES:
        client = bench.fireworks_client
    elif model_key in bench.GATEWAY_CANDIDATES:  # ClickHouse inference gateway (OpenAI-compat)
        client = bench.gateway_client
    elif model_key in bench.ANTHROPIC_CANDIDATES:  # native Messages API (Fable / Mythos-class)
        # Mirror the eval's native routing: adaptive-thinking-only models take effort
        # via output_config and reject an explicit thinking config; give the reasoning
        # budget room so the completion isn't truncated inside a thinking block.
        extra = ({"output_config": {"effort": "high"}} if model_id in bench.ADAPTIVE_THINKING_ONLY
                 else {"thinking": {"type": "disabled"}})
        resp = bench.retry(lambda: bench.anthropic_native.messages.create(
            model=model_id, max_tokens=max(max_tokens, 4096),
            messages=[{"role": "user", "content": prompt}], extra_body=extra),
            what=f"probe.messages[{model_key}]")
        return "".join(b.text for b in resp.content
                       if getattr(b, "type", None) == "text").strip()
    else:
        raise KeyError(model_key)
    # Budget and token-kwarg are separate concerns. Kimi (K2 Thinking / K2.6) emits
    # reasoning before visible text and needs the larger budget or the completion
    # truncates inside the thinking block (observed: empty generations at 300 tokens);
    # but only o*/gpt-5* take OpenAI's max_completion_tokens — Kimi is served over
    # Fireworks/Mantle, which use max_tokens (matching run_candidate_openai_compat).
    _openai_reasoning = model_id.startswith(("o", "gpt-5"))
    _big = _openai_reasoning or "thinking" in model_id or "kimi" in model_id
    _token_kwarg = "max_completion_tokens" if _openai_reasoning else "max_tokens"
    kwargs = {_token_kwarg: max(max_tokens, 2048) if _big else max_tokens}
    if "qwen3p8-max" in model_id:
        # Its separate-channel reasoning is unbounded on completion-style prompts
        # (starved 90%+ of cells even at 8192); the probe measures bare
        # completion, so reasoning is disabled — same design intent as the
        # Anthropic branch's thinking:disabled above.
        kwargs["reasoning_effort"] = "none"
    if not _big:
        kwargs["temperature"] = 0
    resp = bench.retry(lambda: client.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}], **kwargs),
        what=f"probe.completions[{model_key}]")
    # A null message (e.g. a Vertex safety-filtered response) is an empty generation,
    # not an infra error: the model produced nothing, which for both probe types
    # scores as zero memorisation rather than a retry-forever cell. But empty WITH
    # finish_reason=length is budget starvation (reasoning ate the cap), not an
    # empty generation — scoring it 0 would fabricate a clean probe.
    choice = resp.choices[0]
    msg = choice.message
    text = ((msg.content if msg else "") or "").strip()
    if not text and choice.finish_reason == "length":
        raise RuntimeError(f"probe truncated before any visible output (finish=length, {model_id})")
    return text


def similarity(a: str, b: str) -> float:
    """Word-level sequence similarity in [0, 1]."""
    return difflib.SequenceMatcher(None, " ".join(a.split()).lower().split(),
                                   " ".join(b.split()).lower().split()).ratio()


def load_our_items(sample: int, annot: Path | None = None):
    """Question prefixes to probe for memorisation.

    Returns nothing without reading anything when sample is 0. It used to read
    the annotation file unconditionally, so `--sample 0` still needed the board's
    corpus and the probe could not run at all outside this repository, even with
    the completion half switched off.
    """
    if sample <= 0:
        return []
    path = annot or (SYNTH / "annotated.jsonl")
    if not path.exists():
        raise SystemExit(
            f"no annotated questions at {path}.\n"
            f"  The default is the board's. Pass --annot for your own, or "
            f"--sample 0 --no-control to run the entity probe alone.")
    rng = random.Random(SEED)
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("excluded") and r.get("gt_results") and len(r["nl_question"].split()) >= 24:
                rows.append(r)
    rng.shuffle(rows)
    items = []
    for r in rows[:sample]:
        w = r["nl_question"].split()
        cut = int(len(w) * 0.6)
        items.append({"probe": "ours_completion", "id": r["trace_id"][:8],
                      "prefix": " ".join(w[:cut]), "tail": " ".join(w[cut:]),
                      "dataset": "data-agent-mnist"})
    return items


def load_entities(n: int, table: str = ENTITY_TABLE, name_col: str = ENTITY_NAME,
                  id_col: str = ENTITY_ID, db_path: Path | None = None,
                  session_timezone: str | None = None):
    """Sample (name, id) pairs to probe for.

    db_path matters as much as the table name: making the table configurable
    while still opening the board store meant a foreign table name was queried
    against our warehouse, so the probe could not run against either example
    instance and would fail with UNKNOWN_TABLE rather than saying why.
    """
    if n <= 0:
        return []
    import chdb.session as chs
    s = chs.Session(str(db_path or (SYNTH / "chdb")))
    if session_timezone:
        s.query(f"SET session_timezone = '{session_timezone}'")
    out = s.query(
        f"SELECT {name_col}, {id_col} FROM {table} "
        f"ORDER BY cityHash64({id_col}) LIMIT {n}", "JSONEachRow")
    items = []
    for line in str(out).splitlines():
        if line.strip():
            r = json.loads(line)
            items.append({"probe": "entity_recovery", "id": str(r[id_col])[:8],
                          "name": r[name_col], "answer": r[id_col],
                          "table": table, "id_col": id_col})
    return items


def load_spider(sample: int, split: str = "train"):
    """Positive-control items across an exposure gradient: train and dev public
    (with gold SQL) since 2018 — dev the more heavily quoted in papers and eval
    harnesses — and the official 2,147-question TEST split, held private for the
    leaderboard and public only since ~2024, i.e. a much shorter crawl window."""
    # Nothing, and no import, for zero items. `datasets` is an optional
    # dependency and load_dataset DOWNLOADS, so importing before checking the
    # count meant --sample 0 still pulled three corpora to produce nothing, and
    # failed outright without the dependency. The other two loaders short-circuit
    # the same way.
    if sample <= 0:
        return []
    from datasets import load_dataset
    rng = random.Random(SEED)
    if split == "test":
        ds = load_dataset("Mogine/SpiderTest", split="test")
    else:
        ds = load_dataset("xlangai/spider", split=split)
    idx = rng.sample(range(len(ds)), min(sample * 2, len(ds)))
    probe = {"train": "spider_control", "validation": "spider_dev_control",
             "test": "spider_test_control"}[split]
    items = []
    for i in idx:
        q = ds[i]["question"]
        w = q.split()
        if len(w) < 12:
            continue
        cut = int(len(w) * 0.6)
        items.append({"probe": probe, "id": f"spider-{split}-{i}",
                      "prefix": " ".join(w[:cut]), "tail": " ".join(w[cut:]),
                      "dataset": "Spider"})
        if len(items) == sample:
            break
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(bench.ALL_CANDIDATES))
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--entities", type=int, default=25)
    ap.add_argument("--entity-table", default=ENTITY_TABLE,
                    help="table holding a name and a stable id for the entity probe")
    ap.add_argument("--entity-name-column", default=ENTITY_NAME)
    ap.add_argument("--entity-id-column", default=ENTITY_ID)
    ap.add_argument("--annot", type=Path, default=None,
                    help="annotated questions for the completion probe "
                         "(default: the board's; unused when --sample 0)")
    ap.add_argument("--db-path", type=Path, default=None,
                    help="warehouse to sample entities from (default: the board's)")
    ap.add_argument("--session-timezone", default=None,
                    help="pin the chDB session timezone, for a DateTime grain")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    items = load_our_items(args.sample, args.annot) + load_entities(
        args.entities, args.entity_table, args.entity_name_column,
        args.entity_id_column, args.db_path, args.session_timezone)
    if not args.no_control:
        items += (load_spider(args.sample, "train") + load_spider(args.sample, "validation")
                  + load_spider(args.sample, "test"))

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("score") is not None:   # errored calls are retried, not skipped
                    done.add((r["probe"], r["id"], r["model"]))

    todo = [(it, m) for it in items for m in models if (it["probe"], it["id"], m) not in done]
    print(f"{len(items)} items x {len(models)} models; {len(todo)} calls to run "
          f"({len(done)} cached)")

    def run(pair):
        it, m = pair
        try:
            if it["probe"] == "entity_recovery":
                prompt = (ENTITY_PROMPT.replace("<<DATASET>>", "data-agent-mnist")
                                       .replace("<<TABLE>>", it["table"])
                                       .replace("<<IDCOL>>", it["id_col"])
                                       .replace("<<NAME>>", it["name"]))
                gen = _complete(m, prompt, max_tokens=100)
                score = 1.0 if str(it["answer"]).lower() in gen.lower() else 0.0
            else:
                gen = _complete(m, COMPLETION_PROMPT.replace("<<DATASET>>", it["dataset"])
                                                     .replace("<<PREFIX>>", it["prefix"]))
                score = similarity(gen, it["tail"])
            return {"probe": it["probe"], "id": it["id"], "model": m,
                    "score": round(score, 4), "generation": gen[:500]}
        except Exception as e:  # noqa: BLE001 — record and move on
            return {"probe": it["probe"], "id": it["id"], "model": m,
                    "score": None, "error": str(e)[:200]}

    n = 0
    with args.out.open("a") as f:
        for _it, res, exc in bench.map_concurrent(run, todo, workers=8):
            if res:
                f.write(json.dumps(res) + "\n")
                f.flush()
                n += 1
                if n % 100 == 0:
                    print(f"  {n}/{len(todo)}")
            elif exc:
                print(f"  ! {exc}", file=sys.stderr)

    # summary
    by = defaultdict(list)
    for line in args.out.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("score") is not None and r["model"] in models:
                by[(r["model"], r["probe"])].append(r["score"])
    probes = ["ours_completion", "entity_recovery", "spider_control",
              "spider_dev_control", "spider_test_control"]
    print(f"\n{'model':24}" + "".join(f"{p:>18}" for p in probes))
    for m in models:
        cells = []
        for p in probes:
            v = by.get((m, p))
            cells.append(f"{sum(v)/len(v):>17.3f} " if v else f"{'--':>18}")
        print(f"{m:24}" + "".join(cells))


if __name__ == "__main__":
    main()
