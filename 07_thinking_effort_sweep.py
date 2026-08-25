"""
Thinking x effort sweep for the direct-Anthropic candidate.

Runs the text2sqlbench agentic loop over an 8-cell grid — thinking {off, on} x
effort {low, medium, high, xhigh} — via the native Messages API runner
(bench.run_candidate_messages_api), scores each cell with the existing
cross-model judge panel against the majority-vote ground truth, and reports
pass-rate / answered / turns / latency / output-tokens per cell.

Parallel (ThreadPoolExecutor): each (cell, question) does its agentic run then
its judge panel in one worker. chDB is a single non-thread-safe session, so DB
queries are serialized behind a lock — model calls (the slow part) stay parallel.
Resume-safe; local output only (unreleased-model results stay local).

    uv run 07_thinking_effort_sweep.py [--limit N] [--workers W] [--cells c1,c2]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
import obs
from warehouse import DATA_DIR, Warehouse

ANNOT_PATH = DATA_DIR / "annotated.jsonl"
OUT_PATH   = Path(__file__).resolve().parent / "sweep_thinking_effort.jsonl"

# MLflow run/label for the candidate. Deliberately NOT the vendor's internal
# name or model_id: results for an unreleased model stay local, so only an
# anonymized label reaches the
# shared tracker. Override with --mlflow-model-label (or MLFLOW_MODEL_LABEL)
# once the model is public.
DEFAULT_MLFLOW_LABEL = os.environ.get("MLFLOW_MODEL_LABEL", "anthropic-eap-candidate")
MLFLOW_EXPERIMENT    = "text2sqlbench-thinking-effort"

THINKING = ["off", "on"]
EFFORT   = ["low", "medium", "high", "xhigh"]
CELLS    = [f"think-{t}-{e}" for t in THINKING for e in EFFORT]


def cell_params(cell: str) -> tuple[str, str]:
    _, thinking, effort = cell.split("-")
    return thinking, effort


def _incomplete(r: dict) -> bool:
    """A run that exhausted its budget (max turns or max output tokens) without
    producing a final answer. work() skips judging it (scores it a fail) and
    cell_stats drops it from the answered denominator, so a truncated run is
    never credited as a clean answer."""
    return (r.get("error") in (f"max turns ({bench.MAX_TURNS})", bench.ERR_MAX_OUTPUT_TOKENS)
            and not (r.get("final_answer") or "").strip())


def load_ground_truth(path: Path = ANNOT_PATH) -> dict[str, dict]:
    gt = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("excluded") or not r.get("gt_results"):
            continue
        gt[r["trace_id"]] = r
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, choices=sorted(bench.ANTHROPIC_CANDIDATES),
                    help="direct-Anthropic candidate name (resolved via bench.ANTHROPIC_CANDIDATES)")
    ap.add_argument("--limit", type=int, default=None, help="questions per cell (smoke)")
    ap.add_argument("--workers", type=int, default=12, help="parallel workers")
    ap.add_argument("--cells", type=str, default=None, help="comma-separated subset of cells")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--mlflow-model-label", type=str, default=DEFAULT_MLFLOW_LABEL,
                    help="anonymized model label for MLflow (never a vendor-internal name)")
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = ap.parse_args()

    cand_name = args.candidate
    model_id  = bench.ANTHROPIC_CANDIDATES[cand_name]
    cells = [c.strip() for c in args.cells.split(",")] if args.cells else list(CELLS)
    cells = [c for c in cells if c]                     # drop blanks (trailing comma)
    invalid = [c for c in cells if c not in CELLS]
    if invalid:                                          # validate like --candidate
        ap.error(f"invalid --cells {invalid}; choose from {CELLS}")
    wh    = Warehouse()
    db_lock = threading.Lock()
    out_lock = threading.Lock()

    def ch_query(q: str) -> str:
        with db_lock:                                # chDB session is not thread-safe
            return wh.query(q)

    system_prompt = wh.system_prompt()
    gt = load_ground_truth()
    tids = list(gt)
    if args.limit is not None:
        tids = tids[: args.limit]
    print(f"DB ready: {wh.n_rows:,} rows. cells={len(cells)} questions={len(tids)} "
          f"-> {len(cells) * len(tids)} runs, workers={args.workers}")

    # Resume: skip (cell, trace_id) already written.
    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["cell"], r["trace_id"]))
                rows.append(r)

    tasks = [(c, tid) for c in cells for tid in tids if (c, tid) not in done]
    print(f"to run: {len(tasks)} ({len(done)} already done)")

    def work(cell: str, tid: str) -> dict:
        thinking, effort = cell_params(cell)
        q = gt[tid]
        question, gt_results, gt_answer = q["nl_question"], q["gt_results"], q.get("gt_answer", "")
        try:
            r = bench.run_candidate_messages_api(
                question, model_id, ch_query, system_prompt,
                thinking=thinking, effort=effort)
            if _incomplete(r):
                score = {"outcome": "fail", "tally": {}, "panel": [],
                         "reasoning": "budget exhausted, no final answer"}
            else:
                score = bench.judge_panel(question, gt_results, gt_answer,
                                          r["sql_results"], r.get("final_answer", ""), cand_name)
        except Exception as e:                       # noqa: BLE001
            r = {"sqls": [], "sql_results": [], "final_answer": "", "turns": 0,
                 "latency": None, "error": str(e), "output_tokens": 0}
            score = {"outcome": "error", "reasoning": str(e)}
        return {
            "cell": cell, "thinking": thinking, "effort": effort,
            "trace_id": tid, "question": question,
            "outcome": score["outcome"], "tally": score.get("tally"),
            "turns": r["turns"], "latency": r.get("latency"),
            "output_tokens": r.get("output_tokens"),
            "sqls": r.get("sqls", []), "sql_results": r.get("sql_results", []),
            "final_answer": r.get("final_answer", ""), "error": r.get("error"),
        }

    def append(row: dict):
        with out_lock:
            with args.out.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, c, tid): (c, tid) for c, tid in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            append(row)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(tasks):
                print(f"  {n_done}/{len(tasks)}  ({time.time()-t0:.0f}s)  "
                      f"last={row['cell']} {row['outcome']} turns={row['turns']}")

    summarize(rows, cells)
    if not args.no_mlflow:
        log_to_mlflow(rows, cells, args.mlflow_model_label, len(tids))
    wh.close()
    print(f"\nDone in {time.time()-t0:.0f}s -> {args.out}")


def cell_stats(rs: list[dict]) -> dict:
    """Aggregate metrics for one cell's rows. pass=1, tie=0.5; errors excluded
    from the denominator; answered_rate excludes turn-limited fails."""
    scored   = [r for r in rs if r["outcome"] != "error"]
    passes   = sum({"pass": 1.0, "tie": 0.5}.get(r["outcome"], 0.0) for r in scored)
    # Incomplete == the no-answer case work() skips judging for (budget exhausted
    # AND empty final answer). A fail that produced an answer on the last turn was
    # judged and counts as answered, even at turns == MAX_TURNS.
    answered = [r for r in scored if not _incomplete(r)]
    ans_pass = sum({"pass": 1.0, "tie": 0.5}.get(r["outcome"], 0.0) for r in answered)
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "n_scored": len(scored),
        "pass_rate": passes / len(scored) if scored else 0.0,
        "answered_n": len(answered),
        "answered_rate": ans_pass / len(answered) if answered else 0.0,
        "avg_turns": avg([r["turns"] for r in scored if r["turns"]]),
        "avg_latency_s": avg([r["latency"] for r in scored if r["latency"]]),
        "avg_output_tokens": avg([r["output_tokens"] for r in scored
                                  if r["output_tokens"] is not None]),
        "n_error": sum(1 for r in rs if r["outcome"] == "error"),
    }


def summarize(rows: list[dict], cells: list[str]):
    print(f"\n{'cell':22s} {'n':>3} {'pass%':>6} {'answ%':>6} {'turns':>6} {'lat_s':>6} {'out_tok':>8}")
    for c in cells:
        s = cell_stats([r for r in rows if r["cell"] == c])
        print(f"{c:22s} {s['n_scored']:>3} {s['pass_rate']*100:>6.1f} {s['answered_rate']*100:>6.1f} "
              f"{s['avg_turns']:>6.1f} {s['avg_latency_s']:>6.1f} {s['avg_output_tokens']:>8.0f}")


def log_to_mlflow(rows: list[dict], cells: list[str], model_label: str, n_questions: int):
    """One MLflow run per cell (params thinking/effort + the cell_stats metrics).
    Uses an anonymized model label, so no vendor-internal name reaches the tracker."""
    mlflow = obs.setup_mlflow(MLFLOW_EXPERIMENT)
    if not mlflow:
        print("MLflow not logged (MLFLOW_TRACKING_URI unset / unreachable).")
        return
    for c in cells:
        s = cell_stats([r for r in rows if r["cell"] == c])
        if not s["n_scored"]:
            continue
        thinking, effort = cell_params(c)
        try:
            with mlflow.start_run(run_name=f"{model_label}-{c}"):
                mlflow.log_params({
                    "model": model_label, "thinking": thinking, "effort": effort,
                    "backend": "messages_api", "benchmark": "synthetic-thinking-effort",
                    "eval_method": "judge_panel_majority", "n_questions": n_questions,
                })
                mlflow.log_metrics({
                    "pass_rate": s["pass_rate"], "answered_rate": s["answered_rate"],
                    "n_scored": s["n_scored"], "answered_n": s["answered_n"],
                    "avg_turns": s["avg_turns"], "avg_latency_s": s["avg_latency_s"],
                    "avg_output_tokens": s["avg_output_tokens"],
                })
        except Exception as e:                       # noqa: BLE001
            print(f"  [mlflow error for {c}: {e}]")
    print(f"MLflow: logged {len(cells)} cells to '{MLFLOW_EXPERIMENT}' as '{model_label}-*'")


if __name__ == "__main__":
    main()
