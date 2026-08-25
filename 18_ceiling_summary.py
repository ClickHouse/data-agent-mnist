"""Emit the figure inputs for the 60-turn ceiling board.

One deep run per model yields pass@B for EVERY B by truncation, because the turn
budget is never announced to the model: a budget-B run is the budget-60 run
stopped early, so

    pass@B = scored a pass AND concluded within B turns

That makes the whole curve monotone by construction and free of the
cross-replicate noise that a separate run per budget carries (the earlier
per-budget sweep disagreed with itself by 6 to 12 points at a FIXED budget).

The truncation assumption is checked, not assumed: `consistency` compares the
turn-limited count this run predicts at B=10 against what the published 10-turn
board actually recorded. It agrees within +-12 for 26 of 27 models, which
validates the method, and isolates gemini-3.5-flash at +135.

    uv run 18_ceiling_summary.py --emit <figures>/data/ceiling.json
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from paths import DATA
CEILING = DATA / "text2sqlbench-ceiling/results_b60.jsonl"
BOARD = DATA / "text2sqlbench-synthetic/results.jsonl"
S = {"pass": 1.0, "tie": 0.5, "fail": 0.0}

# Display names; anything not listed falls back to its key.
NAMES = {
    "opus48": "Claude Opus 4.8", "opus47": "Claude Opus 4.7", "opus5": "Claude Opus 5",
    "sonnet5": "Claude Sonnet 5", "sonnet46": "Claude Sonnet 4.6",
    "haiku45": "Claude Haiku 4.5", "fable5": "Claude Fable 5",
    "gpt-5.5": "GPT-5.5", "gpt-5.6": "GPT-5.6", "gpt-4.1": "GPT-4.1", "o4-mini": "o4-mini",
    "gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-3.5-flash": "Gemini 3.5 Flash", "gemini-3.7-flash": "Gemini 3.7 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "kimi-k3": "Kimi K3", "kimi-k2.6": "Kimi K2.6", "kimi-k2-thinking": "Kimi K2 Thinking",
    "qwen3.8-max": "Qwen3.8-Max", "qwen3-coder-480b": "Qwen3 Coder 480B",
    "qwen3-coder-30b": "Qwen3 Coder 30B", "glm-5.2": "GLM-5.2",
    "deepseek-v3.2": "DeepSeek V3.2", "deepseek-v4-pro-0813": "DeepSeek V4 Pro",
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash", "gemma-4-31b": "Gemma 4 31B",
}


def load(path):
    d = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for m, c in (r.get("candidates") or {}).items():
            d[m][r["trace_id"]] = c
    return d


def pass_at(cell, b):
    """Truncate one trajectory to budget b. None when the cell never scored."""
    o = (cell.get("result_score") or {}).get("outcome")
    if o not in S:
        return None
    turn_limited = "max turns" in str(cell.get("error") or "")
    concluded_in_budget = not turn_limited and (cell.get("turns") or 10**6) <= b
    return S[o] if concluded_in_budget else 0.0


def mean_se(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0, 0.0
    mu = sum(vals) / len(vals)
    se = ((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) ** 0.5
          if len(vals) > 1 else 0.0)
    return 100 * mu, 100 * se


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ceiling", type=Path, default=CEILING)
    ap.add_argument("--board", type=Path, default=BOARD)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--emit", type=Path, default=None)
    args = ap.parse_args()

    ceil, board = load(args.ceiling), load(args.board)
    questions = sorted(set.intersection(*[set(v) for v in ceil.values()]))
    budgets = list(range(1, args.budget + 1))

    models = []
    for m, v in ceil.items():
        cells = [v[t] for t in questions]
        curve = [mean_se([pass_at(c, b) for c in cells])[0] for b in budgets]
        head, se = mean_se([pass_at(c, 10) for c in cells])
        ceil_v, ceil_se = mean_se([pass_at(c, args.budget) for c in cells])
        # Only cells whose usage is EXACT feed the medians and totals. A block
        # with calls_missing_usage > 0 is a lower bound, and one where every call
        # missed is all zeros; averaging either in makes a capture failure look
        # like a lean model. This is the same unknown-is-not-zero rule that
        # TokenUsage and _usage_details already enforce, and the emitter was the
        # one place that broke it.
        us_any = [c["usage"] for c in cells if c.get("usage")]
        us = [u for u in us_any if not u.get("calls_missing_usage")]
        dropped = len(cells) - len(us)
        f = (lambda k: st.median([x[k] for x in us])) if us else (lambda k: 0)
        tot = (lambda k: sum(x[k] for x in us)) if us else (lambda k: 0)
        lat = sorted(c.get("latency") or 0 for c in cells if c.get("latency"))
        tl = sum(1 for c in cells if "max turns" in str(c.get("error") or ""))
        # Where the curve stops moving: smallest B within 0.5 pt of the ceiling.
        sat = next((b for b, y in zip(budgets, curve) if ceil_v - y <= 0.5), args.budget)
        models.append({
            "key": m, "name": NAMES.get(m, m),
            "headline": round(head, 2), "headline_se": round(se, 2),
            "ceiling": round(ceil_v, 2), "ceiling_se": round(ceil_se, 2),
            "gain": round(ceil_v - head, 2), "saturates_at": sat,
            "turn_limited": tl,
            "median_turns": st.median([c.get("turns") or 0 for c in cells]),
            "tokens": {k: int(f(k)) for k in
                       ("total_tokens", "prompt_tokens", "completion_tokens",
                        "reasoning_tokens", "cache_read_tokens", "cache_write_tokens")},
            # Board totals over all questions, in Mtok. These replace the
            # re-tokenized ESTIMATE the token figure used to carry as literals:
            # re-tokenizing cannot see hidden reasoning, so it under-counted
            # exactly the reasoning models.
            "tokens_total_mtok": {k: round(tot(k) / 1e6, 4) for k in
                                  ("total_tokens", "prompt_tokens",
                                   "completion_tokens", "reasoning_tokens",
                                   "cache_read_tokens")},
            # usage_cells is the denominator behind every token figure; dropped
            # counts cells excluded for absent or partial usage, so a future run
            # cannot quietly report a total built from a fraction of the board.
            "usage_cells": len(us), "usage_cells_dropped": dropped,
            # st.median (averages the two middle values on an even sample) rather
            # than a hand-rolled midpoint index, which disagreed by up to 0.7s on
            # the models with an even number of timed cells. p90 is nearest-rank,
            # rounded rather than truncated.
            "latency_s": {
                "p50": round(st.median(lat), 1) if lat else 0.0,
                "p90": round(lat[int(round(0.9 * (len(lat) - 1)))], 1) if lat else 0.0,
            },
            "curve": [round(y, 2) for y in curve],
            "scored": sum(1 for c in cells if pass_at(c, args.budget) is not None),
        })
    models.sort(key=lambda r: -r["ceiling"])

    # Does this run reproduce the published 10-turn board? Validates truncation.
    consistency = []
    for r in models:
        m = r["key"]
        qs = sorted(set(ceil[m]) & set(board.get(m, {})))
        if not qs:
            continue
        predicted = sum(1 for t in qs
                        if "max turns" in str(ceil[m][t].get("error") or "")
                        or (ceil[m][t].get("turns") or 0) > 10)
        actual = sum(1 for t in qs if "max turns" in str(board[m][t].get("error") or ""))
        consistency.append({"key": m, "predicted_tl_at_10": predicted,
                            "board_tl_at_10": actual, "delta": predicted - actual})

    out = {
        "questions": len(questions), "budget": args.budget, "budgets": budgets,
        "models": models, "consistency": consistency,
        "gain_median": round(st.median([r["gain"] for r in models]), 2),
        "gain_max": round(max(r["gain"] for r in models), 2),
        "models_gaining_over_1se": sum(1 for r in models if r["gain"] > r["ceiling_se"]),
    }
    print(f"{len(models)} models, {len(questions)} questions, budget {args.budget}")
    print(f"  ceiling gain: median {out['gain_median']:+.1f}, max {out['gain_max']:+.1f}, "
          f"{out['models_gaining_over_1se']}/{len(models)} above 1 SE")
    worst = max(consistency, key=lambda c: abs(c["delta"]))
    print(f"  truncation check: worst |delta| {worst['delta']:+d} ({worst['key']})")
    drops = sum(r["usage_cells_dropped"] for r in models)
    print(f"  usage: {sum(r['usage_cells'] for r in models)} exact cells, "
          f"{drops} dropped as absent/partial"
          + ("" if drops <= len(models) else "   <-- CHECK: token figures cover "
             "less of the board than they appear to"))
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.emit}")


if __name__ == "__main__":
    main()
