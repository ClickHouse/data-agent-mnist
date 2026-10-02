"""Render the failure-mode report charts from the Jev labeler output.

Reads the per-cell labels written by label_failure_modes.py and renders three PNGs
into the output directory:

  1. fm_crosswalk.png — Jev sub-mode distribution, colored by the coarse failure mode
     (FM1 to FM5, per 07_failure_modes.py) each sub-mode maps to.
  2. fm_by_model.png  — per-model failure-mode profile in the coarse taxonomy.
  3. fm_cdf.png       — the correct_despite_fail (judge-of-judges) distribution with
     the 0.65 threshold marked.

matplotlib is pulled in at run time rather than declared (imported inside main so the
module imports without it):

  uv run --with matplotlib plot_failure_modes.py [--labels PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Jev sub-mode -> blog failure mode (07_failure_modes.py / DAB paper). `budget` and
# `not a model failure` are the two buckets the blog keeps outside FM1 to FM5. A few
# sub-modes straddle two codes (a missing join is FM2, a wrong join key FM4); each is
# placed by its dominant reading.
FM_OF = {
    "stopped_early": "FM1 no attempt",
    "misread_question": "FM2 wrong plan", "wrong_filter": "FM2 wrong plan",
    "wrong_time_window": "FM2 wrong plan",
    "wrong_identifier": "FM3 wrong data", "unresolved_lookup": "FM3 wrong data",
    "wrong_table": "FM3 wrong data", "wrong_column": "FM3 wrong data",
    "wrong_join": "FM4 wrong impl", "wrong_grain": "FM4 wrong impl",
    "wrong_aggregation": "FM4 wrong impl",
    "clickhouse_error": "FM5 runtime error",
    "ran_out_of_budget": "budget", "tool_loop": "budget",
    "ambiguous_question": "not a model failure",
    "correct_but_judged_fail": "not a model failure",
}
FM_ORDER = ["FM1 no attempt", "FM2 wrong plan", "FM3 wrong data", "FM4 wrong impl",
            "FM5 runtime error", "budget", "not a model failure"]
# Okabe-Ito, colorblind-safe.
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#999999"]
FM_COLOR = {fm: OKABE[i] for i, fm in enumerate(FM_ORDER)}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=HERE / "out" / "failure_labels.jsonl",
                    help="per-cell labels JSONL from label_failure_modes.py")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out",
                    help="directory to write the PNGs into")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load(args.labels)
    # A sub-mode the crosswalk does not know rolls up to "not a model failure", so a new
    # labeler sub-mode is charted rather than crashing the run.
    fm_of_sub = {r["sub_mode"]: FM_OF.get(r["sub_mode"], "not a model failure") for r in rows}
    for r in rows:
        r["fm"] = fm_of_sub[r["sub_mode"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    fm_tot = collections.Counter(r["fm"] for r in rows)
    print("FM totals:", {k: fm_tot[k] for k in FM_ORDER if fm_tot[k]})

    # 1. sub-mode distribution, colored by the blog failure mode it rolls up to
    sm = collections.Counter(r["sub_mode"] for r in rows).most_common()
    labels = [s for s, _ in sm][::-1]
    vals = [v for _, v in sm][::-1]
    colors = [FM_COLOR[fm_of_sub[s]] for s in labels]
    fig, ax = plt.subplots(figsize=(9.5, 7), dpi=160)
    ax.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("failed cells")
    ax.set_title(f"Jev sub-modes (n={n}), colored by blog failure mode")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FM_COLOR[fm]) for fm in FM_ORDER]
    ax.legend(handles, FM_ORDER, title="blog failure mode", loc="lower right",
              fontsize=8, title_fontsize=9)
    ax.margins(x=0.1)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fm_crosswalk.png", bbox_inches="tight")

    # 2. per-model failure-mode profile, stacked
    models = [m for m, _ in collections.Counter(r["model"] for r in rows).most_common()]
    per = {m: collections.Counter(r["fm"] for r in rows if r["model"] == m) for m in models}
    fig2, ax2 = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    left = [0] * len(models)
    for fm in FM_ORDER:
        seg = [per[m][fm] for m in models]
        ax2.barh(models, seg, left=left, color=FM_COLOR[fm], label=fm)
        left = [a + b for a, b in zip(left, seg)]
    ax2.set_xlabel("failed cells")
    ax2.set_title("Failure-mode profile by model (blog taxonomy, via Jev)")
    ax2.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig2.tight_layout()
    fig2.savefig(args.out_dir / "fm_by_model.png", bbox_inches="tight")

    # 3. judge-of-judges flag distribution
    cdf = [r["correct_despite_fail"] for r in rows]
    fig3, ax3 = plt.subplots(figsize=(9.5, 4.5), dpi=160)
    ax3.hist(cdf, bins=20, range=(0, 1), color="#0072B2", edgecolor="white")
    ax3.axvline(0.65, ls="--", lw=2, color="#D55E00")
    ax3.text(0.66, ax3.get_ylim()[1] * 0.9, "0.65: genuine grader\nerrors above",
             color="#D55E00", fontsize=9, va="top")
    ax3.set_xlabel("Jev correct_despite_fail (probability the candidate is right despite the fail)")
    ax3.set_ylabel("failed cells")
    ax3.set_title(f"Judge-of-judges flag across {n} failures")
    fig3.tight_layout()
    fig3.savefig(args.out_dir / "fm_cdf.png", bbox_inches="tight")

    print(f"wrote fm_crosswalk.png, fm_by_model.png, fm_cdf.png to {args.out_dir}")


if __name__ == "__main__":
    main()
