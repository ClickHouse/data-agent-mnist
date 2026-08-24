"""
Post-eval statistics over results.jsonl (no model calls, fully offline).

1. Leaderboard uncertainty, following Miller (2024), "Adding Error Bars to Evals"
   (arXiv:2411.00640): the pass rate is a plain mean over questions, so the CLT
   standard error of the mean applies and bootstrapping is unnecessary. Reports
   SE = sd(s_i)/sqrt(n) per model (infra errors excluded from the denominator;
   pass=1, tie=0.5) and CI95 = mean +/- 1.96*SE. Model comparisons use the paired
   difference test on per-question score differences vs the top model, which
   cancels question difficulty.

2. Column-linker firing rate: the result-equivalence check links columns by a
   zero-shot model call ONLY when the two result sets' column sets differ
   (identity fast-path otherwise). results.jsonl + annotated.jsonl carry every
   compared result string, so the firing rate is derivable offline.

    uv run 08_results_stats.py [--results PATH]
"""
import argparse
import importlib
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import RETIRED_CANDIDATES, _parse_result  # noqa: E402  (offline parser, no client use)
from paths import DATA
RESULTS   = DATA / "text2sqlbench-synthetic/results.jsonl"
ANNOTATED = DATA / "text2sqlbench-synthetic/annotated.jsonl"
SCORE     = {"pass": 1.0, "tie": 0.5, "fail": 0.0}


def load(results_path: Path):
    """-> (models, outcome matrix [n_questions x n_models] with NaN = infra error,
    tl mask [same shape]). tl marks turn-limited cells (error == "max turns (N)"),
    including the judged-with-partial-answer ones."""
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    models = sorted({m for r in rows for m in r["candidates"]} - RETIRED_CANDIDATES.keys())
    mat = np.full((len(rows), len(models)), np.nan)
    tl = np.zeros((len(rows), len(models)), dtype=bool)
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            c = r["candidates"].get(m) or {}
            rs = c.get("result_score") or {}
            if rs.get("outcome") in SCORE:
                mat[i, j] = SCORE[rs["outcome"]]
            tl[i, j] = "max turns" in str(c.get("error") or "")
    return rows, models, mat, tl


def answered_stats(mat: np.ndarray, tl: np.ndarray):
    """Canonical "Answered pass rate": pass rate over scored, non-turn-limited
    cells only (AI-1820 metric hygiene — this is the single implementation the
    notes/deck/figures quote). Returns (answered% [m], tl count [m]).
    NOTE: an optimistic bound on the unbounded-budget score, not an estimate of
    it — models turn-limit disproportionately on questions they also fail with
    extended budgets (measured ceilings land 10-18 pts below Answered)."""
    a = mat.copy()
    a[tl] = np.nan
    with np.errstate(invalid="ignore"):
        answered = np.nanmean(a, axis=0)
    # tl count includes turn-limited cells whatever their scoring status, matching
    # the board tables; the rate above uses scored cells only.
    return answered, np.sum(tl, axis=0)


def pass_rates(mat: np.ndarray) -> np.ndarray:
    return np.nanmean(mat, axis=0)


def clt_stats(mat: np.ndarray):
    """-> (mean [m], se [m], z_vs_top [m], p_two_sided [m], top). Paired test vs top."""
    if mat.size == 0 or mat.shape[1] == 0:
        raise SystemExit("no candidates in results.jsonl — nothing to analyze")
    mean = pass_rates(mat)
    cnt  = np.sum(~np.isnan(mat), axis=0)
    se   = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(cnt)
    top  = int(np.argmax(mean))
    z = np.full(mat.shape[1], np.nan)
    p = np.full(mat.shape[1], np.nan)
    for j in range(mat.shape[1]):
        if j == top:
            continue
        both = ~np.isnan(mat[:, top]) & ~np.isnan(mat[:, j])
        d = mat[both, top] - mat[both, j]
        if d.size < 2:
            continue                       # need >=2 paired points for a variance
        sed = d.std(ddof=1) / np.sqrt(d.size)
        if sed == 0:
            # zero variance in the paired differences: no spread to test against.
            # Identical models (mean diff 0) -> p=1; a constant non-zero gap is a
            # degenerate perfect separation -> p=0. Leaves z as NaN (undefined).
            p[j] = 1.0 if d.mean() == 0 else 0.0
            continue
        z[j] = d.mean() / sed
        p[j] = 2 * (1 - 0.5 * (1 + erf(abs(z[j]) / sqrt(2))))
    return mean, se, z, p, top


def linker_firing(rows, annotated_path: Path = ANNOTATED) -> dict:
    """Fraction of compared (gt, candidate) result pairs whose column sets differ.
    Ground-truth result sets live in annotated.jsonl, keyed by trace_id."""
    gt_by_trace = {a["trace_id"]: a.get("gt_results") or []
                   for a in (json.loads(l) for l in annotated_path.read_text().splitlines() if l.strip())}
    # Mismatched pair, loudly. Every skipped pair reports as "not fired", so
    # results scored against a different run's ground truth print 0.0 firing and
    # look like a measurement rather than an empty intersection.
    if rows and not (set(gt_by_trace) & {r["trace_id"] for r in rows}):
        raise SystemExit(
            f"no trace_id in {annotated_path} appears in the results.\n"
            f"  These are from different runs. Pass --annot alongside --results.")
    fired = same = skipped = 0
    pairs: set = set()
    def _parse(s):
        parsed = _parse_result(s)
        if not parsed:
            return None
        return (tuple(sorted({k for r in parsed for k in r})), len(parsed))
    for r in rows:
        gt = [c for c in gt_by_trace.get(r["trace_id"], []) if c and c != "(empty result)" and not c.startswith("Error:")]
        gt_parsed = [p for p in (_parse(g) for g in gt) if p]
        for cand in r["candidates"].values():
            if (cand.get("result_score") or {}).get("outcome") not in SCORE:
                continue
            cres = [c for c in (cand.get("sql_results") or []) if c and c != "(empty result)" and not c.startswith("Error:")]
            for cp in (_parse(c) for c in cres):
                if cp is None:
                    continue
                cc, cn = cp
                # Mirror annotators_agree's any()-short-circuit: among the row-count-equal
                # GT alternatives, if ANY shares this candidate result's column set the
                # identity fast-path satisfies the match and no linker call is made;
                # only when none do does the equivalence check fall to the linker. Counting
                # per row-count-equal PAIR (the earlier version) over-counted linker use,
                # since one identity-matching GT alternative suppresses it in the eval.
                rc_eq = [gc for gc, gn in gt_parsed if gn == cn]
                if not rc_eq:
                    continue               # row-count mismatch: linker never reached
                if any(set(cc) == set(gc) for gc in rc_eq):
                    same += 1              # identity fast-path, no model call
                else:
                    fired += 1
                    pairs.update((gc, cc) for gc in rc_eq)
        if not gt_parsed:
            skipped += 1
    total = fired + same
    return {"candidate_results_compared": total, "identity_fast_path": same, "needs_linker": fired,
            "needs_linker_pct": 100 * fired / total if total else 0.0,
            "distinct_colset_pairs": len(pairs), "questions_no_gt_cols": skipped}


def emit_json(models, mean, se, ans, tl_cnt, n_questions, out: Path, rows=None):
    """Write the canonical board summary. Notes, deck, paper and figures read
    THIS file instead of hand-carrying numbers: the published Answered column
    was wrong for 20 of 26 rows because a scratch script used `turns >= MAX_TURNS`
    (which drops runs that concluded ON the final turn) instead of the
    ran-out-of-budget error (AI-1785)."""
    import numpy as _np
    from bench import MAX_TURNS
    names = importlib.import_module("09_dds_analysis").NAMES
    # Per-model effort. A "turn" is one MODEL CALL, not a SQL call and not a
    # dialogue turn: a single call can emit several tool_use blocks, so a model
    # that batches queries gets more warehouse access out of the same budget.
    # Reporting both makes that visible instead of implying turns == queries.
    import statistics as _st
    effort = {}
    for k in models:
        cs = [(r["candidates"].get(k) or {}) for r in (rows or [])]
        cs = [c for c in cs if (c.get("turns") or 0) > 0]
        if not cs:
            continue
        t = [c["turns"] for c in cs]
        q = [len(c.get("sqls") or []) for c in cs]
        effort[k] = {
            "turns": _st.median(t), "queries": _st.median(q),
            "queries_per_turn": round(_st.median([qq / tt for tt, qq in zip(t, q)]), 2),
        }
    out_rows = [{"key": k, "name": names.get(k, k),
                 "pass": round(100 * mean[j], 1), "se": round(100 * se[j], 1),
                 "answered": None if _np.isnan(ans[j]) else round(100 * ans[j], 1),
                 "tl": int(tl_cnt[j]), **(effort.get(k) or {})}
                for j, k in enumerate(models)]
    out_rows.sort(key=lambda r: -r["pass"])
    payload = {"n_questions": n_questions, "n_models": len(models),
               "max_turns": MAX_TURNS,
               "answered_definition": ("pass rate over scored runs whose error is not "
                                       "'max turns (N)'; a run that concluded ON the final "
                                       "turn counts as answered, not as turn-limited"),
               "turn_definition": ("one model call in the agentic loop, not a SQL "
                                   "call and not a dialogue turn; one call may issue "
                                   "several queries"),
               "models": out_rows}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    # Pairs with --results and has to move with it. The linker stats read ground
    # truth from annotated.jsonl, keyed by trace_id; left on the board default it
    # crashes when the board data is absent, and does something worse when it is
    # present, silently scoring one run's results against another run's ground
    # truth. No trace_id matches, so every pair is skipped and the section reports
    # zero firing as though it had measured something (AI-1862).
    ap.add_argument("--annot", type=Path, default=ANNOTATED,
                    help="annotated.jsonl holding the ground truth for --results")
    ap.add_argument("--emit-json", type=Path, default=None,
                    help="write the canonical board summary (single source of truth for "
                         "notes/deck/paper/figures)")
    args = ap.parse_args()

    rows, models, mat, tl = load(args.results)
    print(f"{len(rows)} questions x {len(models)} models "
          f"({int(np.isnan(mat).sum())} infra-error cells excluded)\n")

    mean, se, z, p, top = clt_stats(mat)
    order = np.argsort(-mean)
    print(f"CLT standard errors (Miller 2024, arXiv:2411.00640); CI95 = mean +/- 1.96*SE;")
    print(f"paired difference test vs {models[top]}:")
    answered, tl_cnt = answered_stats(mat, tl)
    print(f"{'model':24} {'pass':>6} {'SE':>5} {'95% CI':>16} {'z':>6} {'p(2s)':>7} {'answered':>9} {'tl':>4}")
    for j in order:
        lo, hi = mean[j] - 1.96 * se[j], mean[j] + 1.96 * se[j]
        zs = "--" if j == top or not np.isfinite(z[j]) else f"{z[j]:.2f}"
        ps = "--" if j == top or not np.isfinite(p[j]) else f"{p[j]:.3f}"
        aj = f"{100*answered[j]:8.1f}" if not np.isnan(answered[j]) else "     ---"
        print(f"{models[j]:24} {100*mean[j]:6.1f} {100*se[j]:5.1f} "
              f"[{100*lo:5.1f}, {100*hi:5.1f}] {zs:>6} {ps:>7} {aj} {tl_cnt[j]:4d}")

    if args.emit_json:
        emit_json(models, mean, se, answered, tl_cnt, len(rows), args.emit_json, rows)

    print("\nColumn-linker firing (judge-side equivalence checks):")
    for k, v in linker_firing(rows, args.annot).items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
