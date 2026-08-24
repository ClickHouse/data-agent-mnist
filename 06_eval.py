"""
Evaluate candidates against the majority-vote ground truth (AI-997).

Every model in bench.ALL_CANDIDATES (Opus included) runs the agentic tool loop
against the synthetic DWH; each candidate is scored by a provider-diverse judge
panel (bench.judge_panel) that excludes the candidate's own model, so there is no
self-evaluation. Reads annotated.jsonl (skipping questions excluded by the
all-annotators-disagree filter), writes results.jsonl with each judge's vote.

Resume-safe (skips candidate-questions already in results.jsonl). MLflow + Langfuse
optional via obs.py.

    uv run 06_eval.py [--limit N] [--models m1,m2,...]

Writes data/benchmarks/text2sqlbench-synthetic/results.jsonl.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
import obs
from warehouse import DATA_DIR, Warehouse
from verify_board import verify_subset

ANNOT_PATH   = DATA_DIR / "annotated.jsonl"
RESULTS_PATH = DATA_DIR / "results.jsonl"


def load_ground_truth(annot_path: Path = ANNOT_PATH) -> dict[str, dict]:
    """Usable ground truth keyed by trace_id (drops AI-997-excluded questions)."""
    gt = {}
    for line in annot_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("excluded"):
            continue
        if not r.get("gt_results"):
            continue
        gt[r["trace_id"]] = r
    return gt


def _merge_meta(base: dict, kw: dict) -> dict:
    """Fold usage kwargs into one update(), merging rather than clobbering metadata."""
    out = dict(kw)
    out["metadata"] = {**base, **out.get("metadata", {})}
    return out


def _usage_details(usage: dict | None) -> dict:
    """-> kwargs for Langfuse `update()` carrying token usage natively.

    Langfuse computes cost from `usage_details` on a generation, keyed by the
    observation's model, and rolls it up across traces. Passing the block as plain
    metadata instead would store it but leave it uncosted and unaggregatable,
    which is the opposite of what it is for.

    `input` excludes cache reads and writes on every provider path
    (bench.TokenUsage normalises that), so the buckets are disjoint and Langfuse
    can price each at its own rate.

    Reasoning tokens are NOT emitted as a usage_details key. Langfuse sums every
    key when `total` is absent, and reasoning already sits inside the provider's
    completion figure, so a `reasoning` sibling double-counts exactly the models
    it is meant to illuminate (measured: a run totalling 673 in the artifact came
    back as 704 from the API, over by its 31 reasoning tokens). It rides in
    metadata instead, where it is visible but neither summed nor priced.

    Returns no usage kwargs at all when usage is unknown, so a run without
    capture shows as absent rather than as a zero-cost run.
    """
    if not usage or usage.get("calls_missing_usage", 0) >= usage.get("api_calls", 1):
        return {}
    details = {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
    }
    for src, dst in (("cache_read_tokens", "cache_read_input_tokens"),
                     ("cache_write_tokens", "cache_creation_input_tokens")):
        if usage.get(src):
            details[dst] = usage[src]
    return {"usage_details": details,
            "metadata": {"reasoning_tokens": usage.get("reasoning_tokens", 0),
                         "api_calls": usage.get("api_calls")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N new questions")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated subset of candidates (default: all)")
    ap.add_argument("--annot", type=Path, default=ANNOT_PATH, help="input annotated.jsonl path")
    ap.add_argument("--out", type=Path, default=RESULTS_PATH, help="output results.jsonl path")
    ap.add_argument("--probe-table", default=None,
                    help="fact table used to check the DB is populated and to "
                         "derive the snapshot date (default: the board's)")
    ap.add_argument("--snapshot-column", default=None,
                    help="date column in --probe-table (default: the board's)")
    ap.add_argument("--system-prompt", type=Path, default=None,
                    help="schema context for that warehouse (default: prompts/system.md)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent (question, candidate) pairs in flight (1 = sequential)")
    ap.add_argument("--rescore-errors", action="store_true",
                    help="drop existing error-outcome rows for the requested --models so "
                         "the resume pass re-scores them (e.g. after a runner fix)")
    ap.add_argument("--db-path", type=Path, default=None,
                    help="chDB dir to score against (default: the primary 'chdb'). The board is "
                         "a multi-epoch patchwork; use the split-eval orchestrator to route "
                         "each question to its manifest-assigned DB.")
    ap.add_argument("--no-verify-db", action="store_true",
                    help="skip the board reproduction guard (not recommended)")
    args = ap.parse_args()

    results_path = args.out
    _wh_kw   = {k: v for k, v in (('db_path', args.db_path),
                                  ('system_prompt_path', args.system_prompt),
                                  ('probe_table', args.probe_table),
                                  ('snapshot_column', args.snapshot_column))
                if v is not None}
    wh       = Warehouse(**_wh_kw)
    langfuse = obs.get_langfuse()
    mlflow   = obs.setup_mlflow()
    print(f"DB ready: {wh.n_rows:,} rows, snapshot {wh.snapshot_date}")

    models = (
        {m: bench.ALL_CANDIDATES[m] for m in args.models.split(",")}
        if args.models else dict(bench.ALL_CANDIDATES)
    )
    gt = load_ground_truth(args.annot)
    print(f"Ground truth: {len(gt)} usable questions; candidates: {len(models)}")

    # Board guard: refuse to score questions against a DB that no longer reproduces
    # them (the failure that produced Fable's bogus 14.5%). Verifies the mounted DB
    # is the manifest-assigned epoch for these questions and still reproduces them.
    if not args.no_verify_db:
        db_path = (args.db_path or (DATA_DIR / "chdb")).resolve()
        verify_subset(set(gt), db_path.name, db_path=db_path)

    # Load existing results. Resume is per (question, candidate): adding new
    # candidates scores only the missing pairs and merges them into existing rows.
    results: dict[str, dict] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                results[r["trace_id"]] = r

    def save():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results.values()) + "\n")

    # --rescore-errors treats an existing error-outcome entry as re-scorable (below)
    # WITHOUT deleting it up front: score_pair overwrites the entry in place only when
    # the pair is actually processed, so a partial `--limit` run leaves unprocessed
    # error rows intact instead of wiping them on save().
    todo = []  # (trace_id, [models not yet scored for it])
    rescore_queued = 0
    for tid in gt:
        cands = results.get(tid, {}).get("candidates", {})
        have = set()
        for m in models:
            c = cands.get(m)
            if c is None:
                continue
            if args.rescore_errors and (c.get("result_score") or {}).get("outcome") == "error":
                rescore_queued += 1
                continue
            have.add(m)
        miss = [m for m in models if m not in have]
        if miss:
            todo.append((tid, miss))
    if args.rescore_errors:
        print(f"--rescore-errors: {rescore_queued} error rows queued for re-scoring "
              f"(overwritten in place as processed; unprocessed rows kept)")
    random.Random(bench.EVAL_SEED).shuffle(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    # Flatten to (question, candidate) pairs — the unit of parallelism. Each pair
    # runs the agentic loop + judge panel independently; the only shared state a
    # worker touches is the lock-serialized chDB session (wh.query) and the
    # thread-safe SDK clients, so pairs run concurrently and merge into `results`
    # single-threaded in the consumer loop below.
    pairs = [(tid, m) for tid, miss in todo for m in miss]
    print(f"To score: {len(pairs)} (question, candidate) pairs across {len(todo)} questions "
          f"({args.workers} workers)")

    # Running pass-rates over ALL already-scored data (not just this run).
    running = {m: {"passes": 0.0, "n": 0} for m in models}
    for row in results.values():
        for m in models:
            o = row.get("candidates", {}).get(m, {}).get("result_score", {}).get("outcome")
            if o == "pass":            running[m]["passes"] += 1.0; running[m]["n"] += 1
            elif o == "tie":           running[m]["passes"] += 0.5; running[m]["n"] += 1
            elif o in ("fail", "loss"): running[m]["n"] += 1

    # One MLflow run per candidate (best-effort).
    run_ids = {}
    if mlflow:
        for m, mid in models.items():
            try:
                with mlflow.start_run(run_name=m) as run:
                    mlflow.log_params({
                        "model_id": mid, "benchmark": "synthetic", "n_questions": len(gt),
                        "eval_method": "judge_panel_majority", "gt_method": "majority_vote_3",
                        "backend": "mantle" if m in bench.MANTLE_CANDIDATES else "converse",
                    })
                    run_ids[m] = run.info.run_id
            except Exception as e:
                print(f"WARNING: MLflow init failed for {m}: {e}")

    def score_pair(pair: tuple[str, str]) -> dict:
        """Run one candidate on one question and score it. Runs in a worker thread;
        touches only wh.query (locked) and thread-safe clients. Each pair gets its
        own Langfuse observation — worker threads start with a fresh context, so
        relying on a shared 'current' span would corrupt the trace tree."""
        tid, m = pair
        q          = gt[tid]
        question   = q["nl_question"]
        with langfuse.start_as_current_observation(
            # "generation", not "span": Langfuse costs and aggregates token usage
            # only on generation observations, so a span leaves usage inert in
            # metadata with no cost view, no model rollup and no filtering. The
            # candidate run is a (multi-turn) model invocation, so generation is
            # also the honest type for it.
            name=f"eval:{m}", as_type="generation", input={"question": question},
            model=models[m],
            metadata={"trace_id": tid, "gt_from": q.get("gt_from")},
        ) as span:
            try:
                replay = bench.run_candidate(question, m, models[m], wh.query, wh.system_prompt())
                if (replay.get("error") == f"max turns ({bench.MAX_TURNS})"
                        and not replay.get("final_answer", "").strip()):
                    score = {"outcome": "fail",
                             "reasoning": f"max turns ({bench.MAX_TURNS}) reached with no final answer"}
                else:
                    score = bench.judge_panel(
                        question, q.get("gt_results", []), q.get("gt_answer", ""),
                        replay["sql_results"], replay.get("final_answer", ""), m)
            except Exception as e:
                replay = {"sqls": [], "sql_results": [], "final_answer": "", "turns": 0,
                          "latency": None, "error": str(e)}
                score  = {"outcome": "error", "reasoning": str(e)}
            span.update(output={
                "final_answer": replay["final_answer"], "outcome": score["outcome"],
                "panel": score.get("panel"), "votes": score.get("votes"),
            }, **_merge_meta({"turns": replay["turns"], "error": replay["error"]},
                              _usage_details(replay.get("usage"))))
        return {
            "sqls": replay["sqls"], "sql_results": replay["sql_results"],
            "final_answer": replay["final_answer"], "turns": replay["turns"],
            "latency": replay.get("latency"), "result_score": score,
            "served_model": replay.get("served_model"),
            "error": replay["error"],
            # Exact per-candidate token usage (AI-1540). None for cells written by
            # the error path, or by a runner predating capture — consumers must
            # treat a missing/None block as "unknown", never as zero.
            "usage": replay.get("usage"),
        }

    SAVE_EVERY = 20                        # results.jsonl is multi-MB; don't rewrite per pair
    completed = 0
    try:
        for (tid, m), cand, exc in bench.map_concurrent(score_pair, pairs, args.workers):
            completed += 1
            if exc is not None:            # score_pair swallows its own errors; this is a pool-level failure
                cand = {"sqls": [], "sql_results": [], "final_answer": "", "turns": 0,
                        "latency": None, "error": str(exc),
                        "result_score": {"outcome": "error", "reasoning": str(exc)}}
            q   = gt[tid]
            row = results.setdefault(tid, {"trace_id": tid, "nl_question": q["nl_question"],
                                           "gt_from": q.get("gt_from"), "candidates": {}})
            row["candidates"][m] = cand

            outcome = cand["result_score"]["outcome"]
            running[m]["passes"] += {"pass": 1.0, "tie": 0.5}.get(outcome, 0.0)
            if outcome != "error":
                running[m]["n"] += 1
            pass_rate = running[m]["passes"] / running[m]["n"] if running[m]["n"] else 0.0
            print(f"[{completed}/{len(pairs)}] {m:<22} {outcome:<5} turns={cand['turns']} "
                  f"pr={pass_rate:.2f}  {q['nl_question'][:44]}")
            if m in run_ids:
                try:
                    with mlflow.start_run(run_id=run_ids[m]):
                        mlflow.log_metrics({
                            "pass_rate": pass_rate,
                            "cumulative_passes": running[m]["passes"],
                            "turns": cand["turns"],
                        }, step=completed)
                except Exception as e:
                    print(f"  [mlflow error: {e}]")

            if completed % SAVE_EVERY == 0:
                save()
    finally:
        save()

    langfuse.flush()
    print(f"\nDone. Saved to {results_path}")
    wh.close()


if __name__ == "__main__":
    main()
