"""
Dimensional-layer discriminator analysis over the eval results (offline, no model
calls).

Splits the graded questions by whether the majority-vote ground-truth SQL touches
the dimensional layer, and reports per model:
  * pass rate on each split (pass=1, tie=1/2; infra errors excluded) and the gap;
  * discovery rate — the fraction of those questions where the candidate's own SQL
    touches the layer at all (did it FIND the dimensional layer?);
  * FM2 share of its concluded failures there (wrong plan vs everything else).

The layer is identified by a substring in the SQL, `--layer-marker`, defaulting to
the board's `dbt_dds`. The measurement is the portable part of this script and the
identifier is not: any two-layer warehouse has the same question, and the runnable
example is deliberately built with the same split (4 flat-mart, 4 CRM-hop), so
`--layer-marker crm.` reproduces this analysis there (AI-1858).

This is the analysis behind the release deck's "Why the board moved" slide and the
paper's dds-discriminator section/figure — promoted from a /tmp probe to a committed
script so the numbers are reproducible.

    uv run 09_dds_analysis.py [--plot out.png] [--tex]
    uv run 09_dds_analysis.py --dataset <dir> --layer-marker "crm."
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from bench import RETIRED_CANDIDATES  # noqa: E402
from paths import DATA
SYNTH     = DATA / "text2sqlbench-synthetic"
LAYER_MARKER = "dbt_dds"          # board default; see --layer-marker
SCORE     = {"pass": 1.0, "tie": 0.5, "fail": 0.0}

NAMES = {"opus48": "Claude Opus 4.8", "opus47": "Claude Opus 4.7", "gpt-5.6": "GPT-5.6",
         "opus5": "Claude Opus 5",
         "fable5": "Claude Fable 5", "kimi-k3": "Kimi K3", "kimi-k2.6": "Kimi K2.6",
         "gpt-5.5": "GPT-5.5", "qwen3.8-max": "Qwen3.8-Max",
         "sonnet5": "Claude Sonnet 5", "sonnet46": "Claude Sonnet 4.6",
         "glm-5.2": "GLM-5.2", "deepseek-v4-pro": "DeepSeek V4 Pro (Apr preview)",
         "deepseek-v4-pro-0813": "DeepSeek V4 Pro", "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
         "gemini-2.5-pro": "Gemini 2.5 Pro", "haiku45": "Claude Haiku 4.5",
         "deepseek-v4-flash": "DeepSeek V4 Flash (Apr preview)", "kimi-k2-thinking": "Kimi K2 Thinking",
         "o4-mini": "o4-mini", "gemini-3.5-flash": "Gemini 3.5 Flash",
         "qwen3-coder-480b": "Qwen3-Coder 480B", "gemini-3.1-pro-preview": "Gemini 3.1 Pro (prev.)",
         "deepseek-v3.2": "DeepSeek V3.2", "gemma-4-31b": "Gemma 4 31B", "gpt-4.1": "GPT-4.1",
         "gemini-2.5-flash": "Gemini 2.5 Flash", "gemini-3.7-flash": "Gemini 3.7 Flash",
         "qwen3-coder-30b": "Qwen3-Coder 30B",
         "nova-pro": "Amazon Nova Pro", "nova-lite": "Amazon Nova Lite",
         "nova-micro": "Amazon Nova Micro"}


def compute(dataset: Path = SYNTH, marker: str = LAYER_MARKER):
    gt = {}
    for line in (dataset / "annotated.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            gt[r["trace_id"]] = r
    usable = {t for t, r in gt.items() if not r.get("excluded") and r.get("gt_results")}
    dds    = {t for t in usable if any(marker in (s or "") for s in gt[t].get("gt_sql", []))}

    fm = {}
    for line in (dataset / "fm_labels.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            fm[(r["trace_id"], r["model"])] = r["fm"]

    agg = defaultdict(lambda: dict(dp=0.0, dn=0, disc=0, dtot=0, np=0.0, nn=0, f2=0, ff=0,
                                   ap=0.0, an=0, pass_no_touch=0, dds_tables=[]))
    for line in (dataset / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r["trace_id"]
        if t not in usable:
            continue
        for m, c in r["candidates"].items():
            if m in RETIRED_CANDIDATES:
                continue
            o = (c.get("result_score") or {}).get("outcome")
            a = agg[m]
            if o in SCORE:
                a["ap"] += SCORE[o]; a["an"] += 1
            if t in dds:
                # Discovery, tables/q and pass rate all share the scored denominator
                # (o in SCORE): an infra error is not the model failing to discover, so
                # counting it in dtot would dilute discovery against pass rate (Bugbot).
                if o in SCORE:
                    a["dtot"] += 1
                    a["dp"] += SCORE[o]; a["dn"] += 1
                    touched = any(marker in (s or "") for s in c.get("sqls", []))
                    if touched:
                        a["disc"] += 1
                        a["dds_tables"].append(len({x for s in c.get("sqls", [])
                                                    for x in re.findall(re.escape(marker) + r"\.?(\w+)", s or "")}))
                    # Passing a layer-labelled question WITHOUT touching it: the GT
                    # route used the dims, but an equivalent marts route exists and the
                    # model found it. Discovery therefore conflates route choice with
                    # capability — report this separately.
                    if o in ("pass", "tie") and not touched:
                        a["pass_no_touch"] += 1
                    if o == "fail":
                        a["ff"] += 1
                        if fm.get((t, m)) == "FM2":
                            a["f2"] += 1
            elif o in SCORE:
                a["np"] += SCORE[o]; a["nn"] += 1

    rows = []
    for m, a in agg.items():
        rows.append({
            "model": m, "name": NAMES.get(m, m),
            "overall": 100 * a["ap"] / a["an"] if a["an"] else 0.0,
            "dds":     100 * a["dp"] / a["dn"] if a["dn"] else 0.0,
            "non":     100 * a["np"] / a["nn"] if a["nn"] else 0.0,
            "disc":    100 * a["disc"] / a["dtot"] if a["dtot"] else 0.0,
            "fm2_share": 100 * a["f2"] / a["ff"] if a["ff"] else 0.0,
            "pass_no_touch": a["pass_no_touch"],
            "avg_dds_tables": (sum(a["dds_tables"]) / len(a["dds_tables"])) if a["dds_tables"] else 0.0,
        })
    rows.sort(key=lambda r: -r["overall"])
    return rows, len(dds), len(usable - dds)


def plot(rows, out: Path, marker: str = LAYER_MARKER):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [r["name"] for r in rows]
    ddsv, non, disc = [r["dds"] for r in rows], [r["non"] for r in rows], [r["disc"] for r in rows]
    y, h = np.arange(len(labels)), 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.38 * len(labels) + 1.4),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    ax1.barh(y + h / 2, non, h, label=f"non-{marker}", color="#c9c9c9")
    ax1.barh(y - h / 2, ddsv, h, label=marker, color="#FAD000")
    ax1.set_yticks(y); ax1.set_yticklabels(labels); ax1.invert_yaxis()
    ax1.set_xlabel("Pass rate (%)")
    ax1.set_title(f"Pass rate: {marker} vs non-{marker}", fontweight="bold")
    ax1.legend(loc="lower right"); ax1.set_xlim(0, 85)
    for i, (d, n) in enumerate(zip(ddsv, non)):
        ax1.annotate(f"{d - n:+.0f}", (max(d, n) + 1.5, i), va="center", fontsize=8,
                     color="#b00" if d < n else "#080")
    ax2.barh(y, disc, 0.6, color=["#2a9d3a" if v >= 80 else "#e07b00" for v in disc])
    ax2.set_yticks(y); ax2.set_yticklabels([]); ax2.invert_yaxis()
    ax2.set_xlabel("%")
    ax2.set_title(f"Found the dims\n(SQL touched {marker})", fontweight="bold")
    ax2.set_xlim(0, 100); ax2.axvline(80, ls="--", c="#888", lw=1)
    for i, v in enumerate(disc):
        ax2.annotate(f"{v:.0f}%", (v - 13 if v > 20 else v + 2, i), va="center", fontsize=8,
                     color="w" if v > 20 else "#333")
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", type=Path, default=None, help="write the two-panel figure PNG here")
    ap.add_argument("--tex", action="store_true", help="emit a LaTeX tabular body for the paper")
    ap.add_argument("--dataset", type=Path, default=SYNTH,
                    help="directory holding annotated/results/fm_labels.jsonl")
    ap.add_argument("--layer-marker", default=LAYER_MARKER,
                    help="substring identifying the dimensional layer in SQL "
                         f"(default {LAYER_MARKER!r}; the example uses 'crm.')")
    args = ap.parse_args()

    rows, n_dds, n_non = compute(args.dataset, args.layer_marker)
    print(f"{args.layer_marker} GT questions: {n_dds}   other: {n_non}\n")
    print(f"{'model':24} {'dds%':>6} {'non%':>6} {'gap':>6} {'disc%':>6} {'P-noTouch':>10} {'avg#ddsT':>9} {'FM2%':>6}")
    for r in rows:
        print(f"{r['name']:24} {r['dds']:6.1f} {r['non']:6.1f} {r['dds'] - r['non']:+6.1f} "
              f"{r['disc']:6.1f} {r['pass_no_touch']:10d} {r['avg_dds_tables']:9.2f} {r['fm2_share']:6.0f}")

    if args.tex:
        print("\n% LaTeX body (Model & dds & non-dds & gap & discovery & pass-w/o-dds & tables/q):")
        for r in rows:
            print(f"{r['name']:<24} & {r['dds']:.1f} & {r['non']:.1f} & "
                  f"{r['dds'] - r['non']:+.1f} & {r['disc']:.0f}\\% & "
                  f"{r['pass_no_touch']} & {r['avg_dds_tables']:.1f} \\\\")
    if args.plot:
        plot(rows, args.plot, args.layer_marker)


if __name__ == "__main__":
    main()
