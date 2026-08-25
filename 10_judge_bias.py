"""
Judge leniency vs in-group bias, from the recorded panel votes (offline).

Favourable-vote rate per judge seat x candidate provider (vote value: pass=1,
tie=1/2, fail=0), the seat's overall leniency, and the leniency-adjusted in-group
residual: centre each seat on its own mean, then compare its vote on its own
family against the other seats' votes on that same family. Substitute judges
count toward their seat's provider (sonnet46 -> Anthropic, gemini-2.5-flash ->
Google), matching the report notebook's methodology.

    uv run 10_judge_bias.py [--plot out.pdf]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from bench import RETIRED_CANDIDATES  # noqa: E402
from paths import DATA
RESULTS   = DATA / "text2sqlbench-synthetic/results.jsonl"

JUDGE_PROV = {"opus48": "Anthropic", "sonnet46": "Anthropic",
              "gpt-5.5": "OpenAI", "gpt-5.4": "OpenAI",
              "gemini-2.5-pro": "Google", "gemini-2.5-flash": "Google"}
CAND_PREFIX = [("opus", "Anthropic"), ("sonnet", "Anthropic"), ("haiku", "Anthropic"),
               ("gpt", "OpenAI"), ("o4", "OpenAI"), ("gemini", "Google"), ("gemma", "Google"),
               ("nova", "Amazon"), ("deepseek", "DeepSeek"), ("kimi", "Moonshot"),
               ("qwen", "Qwen"), ("glm", "Zhipu")]
VOTE = {"pass": 1.0, "tie": 0.5, "fail": 0.0, "equivalent": 1.0, "not_equivalent": 0.0}
JUDGES = ["Anthropic", "OpenAI", "Google"]
COLS   = JUDGES + ["DeepSeek", "Zhipu", "Qwen", "Moonshot"]  # Amazon dropped: candidates retired


def cand_prov(m: str) -> str:
    return next((p for pre, p in CAND_PREFIX if m.startswith(pre)), "?")


def compute():
    cell = defaultdict(lambda: [0.0, 0])
    lean = defaultdict(lambda: [0.0, 0])
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for m, c in r["candidates"].items():
            if m in RETIRED_CANDIDATES:
                continue
            for v in ((c.get("result_score") or {}).get("votes") or []):
                jp = JUDGE_PROV.get(v.get("judge"))
                o = v.get("outcome") or v.get("verdict")
                if jp and o in VOTE:
                    cell[(jp, cand_prov(m))][0] += VOTE[o]; cell[(jp, cand_prov(m))][1] += 1
                    lean[jp][0] += VOTE[o]; lean[jp][1] += 1
    rate = {k: 100 * s / n for k, (s, n) in cell.items() if n}
    leniency = {j: 100 * s / n for j, (s, n) in lean.items() if n}
    bias = {}
    for p in JUDGES:
        # Sparse/partial results.jsonl may lack a seat's own-family cell or the other
        # seats' votes on that family — skip the seat rather than raise (Bugbot).
        if (p, p) not in rate or p not in leniency:
            continue
        ig  = rate[(p, p)] - leniency[p]
        oth = [rate[(oj, p)] - leniency[oj] for oj in JUDGES
               if oj != p and (oj, p) in rate and oj in leniency]
        if not oth:
            continue
        bias[p] = ig - sum(oth) / len(oth)
    return rate, leniency, bias


def plot(rate, leniency, bias, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.1),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    cols = COLS + ["avg"]
    mat = np.array([[rate.get((j, c), np.nan) for c in COLS] + [leniency.get(j, np.nan)]
                    for j in JUDGES])
    im = ax1.imshow(mat, cmap="YlOrBr", vmin=15, vmax=80, aspect="auto")
    ax1.set_xticks(range(len(cols))); ax1.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax1.set_yticks(range(len(JUDGES))); ax1.set_yticklabels([f"{j} judge" for j in JUDGES], fontsize=9)
    for i in range(len(JUDGES)):
        for k in range(len(cols)):
            if not np.isfinite(mat[i, k]):     # sparse data: leave missing cells blank
                continue
            ax1.text(k, i, f"{mat[i, k]:.0f}", ha="center", va="center", fontsize=8,
                     color="black" if mat[i, k] < 60 else "white",
                     fontweight="bold" if cols[k] == JUDGES[i] or cols[k] == "avg" else "normal")
    ax1.axvline(len(COLS) - 0.5, color="black", lw=1)
    ax1.set_title("Favourable-vote rate (%): judge seat x candidate provider", fontsize=10, fontweight="bold")

    y = np.arange(len(JUDGES))
    vals = [bias.get(j, np.nan) for j in JUDGES]
    ax2.axvspan(-2, 2, color="0.92")
    ax2.axvline(0, color="black", lw=1)
    ax2.barh(y, [0 if not np.isfinite(v) else v for v in vals], 0.5,
             color=["#999" if not np.isfinite(v) else "#555" if abs(v) <= 2 else "#b25a00"
                    for v in vals])
    ax2.set_yticks(y); ax2.set_yticklabels(JUDGES, fontsize=9); ax2.invert_yaxis()
    ax2.set_xlim(-6, 6)
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        ax2.text(v + (0.25 if v >= 0 else -0.25), i, f"{v:+.1f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9, fontweight="bold")
    ax2.set_title("Leniency-adjusted in-group bias (pts)\nshaded: $\\pm$2-pt band", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()
    rate, leniency, bias = compute()
    def _cell(x):
        return f"{x:8.0f}" if x is not None else f"{'--':>8}"   # missing != 0% favour (Bugbot)
    print(f"{'judge':10}" + "".join(f"{c[:6]:>8}" for c in COLS) + f"{'| avg':>8}")
    for j in JUDGES:
        print(f"{j:10}" + "".join(_cell(rate.get((j, c))) for c in COLS)
              + _cell(leniency.get(j)))
    print("\nleniency-adjusted in-group bias:",
          ", ".join(f"{j} {bias[j]:+.1f}" for j in JUDGES if j in bias) or "(insufficient votes)")
    if args.plot:
        plot(rate, leniency, bias, args.plot)


if __name__ == "__main__":
    main()
