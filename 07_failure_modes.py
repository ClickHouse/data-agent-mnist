"""
Classify every failed (question, candidate) pair into a DAB failure mode.

For each `fail` in the eval results (`06_eval.py` output), an LLM judge labels the
failure as one of five modes from the DAB paper (arxiv 2603.20576):

    FM1  No attempt          — issued no SQL / returned without querying
    FM2  Wrong plan          — logic can't produce the right answer even if executed perfectly
    FM3  Wrong data selection — right plan, wrong table/column/entity
    FM4  Wrong implementation — right plan + data, wrong computation (formula/join key/agg order)
    FM5  Runtime error        — SQL failed to execute (syntax, timeout, API error)

FM1 (no SQL) and FM5 (execution error) are decided cheaply from the record; only the
genuine wrong-answer cases (FM2/3/4) go to the model. Writes `fm_labels.jsonl`
({trace_id, model, fm, reason}) consumed by `06_benchmark_report.ipynb`. Resume-safe:
re-run to label only pairs not already present.

    uv run 07_failure_modes.py [--limit N] [--model ID] [--workers N]

Ported from the retired notebooks/05_synthetic_benchmark.ipynb; adapted for the v2
majority-vote ground truth (`gt_sql`) and parallelized via bench.map_concurrent.
"""
import argparse
import json
import re
from pathlib import Path

import bench
from paths import DATA
SYNTH_DIR   = DATA / "text2sqlbench-synthetic"
RESULTS     = SYNTH_DIR / "results.jsonl"
ANNOTATED   = SYNTH_DIR / "annotated.jsonl"
FM_LABELS   = SYNTH_DIR / "fm_labels.jsonl"

FM_PROMPT = """\
You are evaluating why a text-to-SQL agent failed on a benchmark question \
against a ClickHouse data warehouse.

Question: {question}

Ground-truth (majority-vote) SQLs:
{gt_sqls}

Failed candidate — SQLs issued in order:
{cand_sqls}

SQL results (first 300 chars each):
{sql_results}

Candidate final answer: {final_answer}
Candidate error: {error}

Classify this failure as exactly one of:
FM2 — Wrong plan: query logic is fundamentally incorrect — cannot produce the correct answer \
even with perfect execution (wrong aggregation, wrong grouping, missing join, wrong time window, \
unfounded assumptions about the schema)
FM3 — Wrong data selection: plan is conceptually correct but uses the wrong table, column, or entity
FM4 — Wrong implementation: correct plan and correct data sources, but incorrect computation \
(wrong formula, wrong JOIN key, off-by-one, incorrect aggregation order, type mismatch)

Respond with JSON only, no other text: {{"fm": "FM2", "reason": "one sentence"}}"""


# Budget-exhaustion errors (ran out of turns OR output tokens) — a separate "no answer"
# failure, not a wrong-answer FM. Excluded from the FM breakdown (reported in Answered).
BUDGET_ERRORS = ("max turns", "max output tokens")
def _budget_fail(cand: dict) -> bool:
    e = str(cand.get("error") or "")
    return any(b in e for b in BUDGET_ERRORS)


def classify(question: str, gt_sqls: list, cand: dict, model_id: str) -> dict:
    """Shortcut FM1/FM5 from the record; ask the model for FM2/3/4."""
    sqls  = cand.get("sqls", [])
    error = cand.get("error")
    if not sqls:
        return {"fm": "FM1", "reason": "candidate issued no SQL queries"}
    if error and not _budget_fail(cand):
        return {"fm": "FM5", "reason": str(error)[:200]}
    # warehouse.query returns failed SQL as "Error: ..." strings (top-level error stays None);
    # a run whose every query errored couldn't execute at all -> FM5, not a wrong plan.
    res = cand.get("sql_results", [])
    if res and all(str(r).lstrip().startswith("Error") for r in res):
        return {"fm": "FM5", "reason": "all SQL queries returned execution errors"}

    results_preview = "\n".join(
        f"  [{i+1}] {(r or '')[:300]}" for i, r in enumerate(cand.get("sql_results", []))
    ) or "  (not available)"
    prompt = FM_PROMPT.format(
        question=question,
        gt_sqls="\n".join(f"  {i+1}. {s}" for i, s in enumerate(gt_sqls)) or "  (none)",
        cand_sqls="\n".join(f"  {i+1}. {s}" for i, s in enumerate(sqls)),
        sql_results=results_preview,
        final_answer=(cand.get("final_answer") or "")[:300] or "(none)",
        error=error or "none",
    )
    resp = bench.retry(lambda: bench.bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 256},
    ), what="fm-classify")
    raw = "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'"fm"\s*:\s*"(FM\d)"', raw)
        return {"fm": m.group(1) if m else "FM?", "reason": raw[:200]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--annot",   type=Path, default=ANNOTATED)
    ap.add_argument("--out",     type=Path, default=FM_LABELS)
    ap.add_argument("--model",   type=str, default=bench.JUDGE_MODEL, help="classifier model id")
    ap.add_argument("--limit",   type=int, default=None, help="label at most N new pairs")
    ap.add_argument("--workers", type=int, default=8, help="concurrent classifications (1 = sequential)")
    args = ap.parse_args()

    results = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    gt_sql  = {json.loads(l)["trace_id"]: (json.loads(l).get("gt_sql") or [])
               for l in args.annot.read_text().splitlines() if l.strip()}

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["trace_id"], r["model"]))

    # Turn-limited runs (exhausted MAX_TURNS) are a separate "ran out of budget" failure,
    # not a wrong-answer FM — skip them so they aren't mislabeled FM2-4 by their partial SQL.
    todo = [(r["trace_id"], r["nl_question"], m, c)
            for r in results
            for m, c in r["candidates"].items()
            if (c.get("result_score") or {}).get("outcome") == "fail"
            and not _budget_fail(c)
            and (r["trace_id"], m) not in done]
    if args.limit is not None:   # `is not None` so --limit 0 labels 0, not all
        todo = todo[:args.limit]
    print(f"FM classification — {len(todo)} failures to label ({len(done)} already done), model {args.model}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "a") as f:
        for (tid, q, m, cand), fm, exc in bench.map_concurrent(
            lambda item: classify(item[1], gt_sql.get(item[0], []), item[3], args.model),
            todo, args.workers,
        ):
            if exc or not fm:
                print(f"  ! {m} {tid[:8]}: {exc}")
                continue
            f.write(json.dumps({"trace_id": tid, "model": m,
                                "fm": fm.get("fm", "FM?"), "reason": fm.get("reason", "")}) + "\n")
            f.flush()
            written += 1
            if written % 25 == 0:
                print(f"  {written}/{len(todo)}")
    print(f"\nDone. {written} new labels → {args.out}")


if __name__ == "__main__":
    main()
