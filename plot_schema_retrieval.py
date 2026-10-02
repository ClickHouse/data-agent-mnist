"""Render the schema-retrieval charts for the report.

    uv run --with matplotlib plot_schema_retrieval.py \
        --retrieval out/schema_retrieval.jsonl \
        --stage2 out/sra_missed.jsonl

--retrieval is the output of schema_retrieval.py. --stage2 is the output of
schema_retrieval_agent.py in prepend mode on the missed-gold population, for the
two models the chart shows:

    uv run --with typesafe-sdk schema_retrieval_agent.py --mode prepend \
        --population missed-gold --models gpt-4.1,qwen3-coder-30b \
        --out out/sra_missed.jsonl

matplotlib is pulled in at run time rather than declared (imported in __main__ so the
module imports without it). A chart whose input file is missing is skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def _rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def retrieval_curve(src: Path):
    rows = _rows(src)
    n = len(rows)
    ks = list(range(1, 11))
    recall = [sum(len(set(r["ranked"][:k]) & set(r["gold"])) / len(r["gold"]) for r in rows) / n
              for k in ks]
    allreq = [sum(set(r["gold"]).issubset(set(r["ranked"][:k])) for r in rows) / n for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, recall, "-o", label="recall@k (share of gold tables found)", color="#1f77b4")
    ax.plot(ks, allreq, "-s", label="all-required@k (every gold table found)", color="#d62728")
    ax.set_xlabel("k (tables retrieved)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(f"Jev schema retrieval on Data Agent MNIST ({n} questions, 18 tables)")
    fig.tight_layout()
    p = OUT / "sr_retrieval_curve.png"
    fig.savefig(p, dpi=140)
    print("wrote", p)


def stage2_recovery(src: Path):
    rows = _rows(src)
    models = ["gpt-4.1", "qwen3-coder-30b"]
    base, prio = [], []
    labels = []
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        n = len(mr)
        base.append(sum(r["baseline_outcome"] == "pass" for r in mr) / n * 100)
        prio.append(sum(r["gated_outcome"] == "pass" for r in mr) / n * 100)
        rec = sum(r["baseline_outcome"] != "pass" and r["gated_outcome"] == "pass" for r in mr)
        reg = sum(r["baseline_outcome"] == "pass" and r["gated_outcome"] != "pass" for r in mr)
        labels.append(f"{m}\n(n={n}, +{rec} / -{reg})")
    x = range(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - w / 2 for i in x], base, w, label="baseline", color="#aaaaaa")
    ax.bar([i + w / 2 for i in x], prio, w, label="with top-5 table note", color="#2ca02c")
    ax.set_ylabel("pass rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(prio) * 1.3)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Table-note prioritization on baseline table-miss failures")
    for i, (b, pr) in enumerate(zip(base, prio)):
        ax.text(i - w / 2, b + 0.5, f"{b:.0f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, pr + 0.5, f"{pr:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    p = OUT / "sr_stage2_recovery.png"
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (used by the chart functions)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval", type=Path, default=OUT / "schema_retrieval.jsonl",
                    help="schema_retrieval.py output for the recall / all-required curve")
    ap.add_argument("--stage2", type=Path, default=OUT / "sra_missed.jsonl",
                    help="schema_retrieval_agent.py prepend-mode output for the recovery chart")
    args = ap.parse_args()
    for chart, src in ((retrieval_curve, args.retrieval), (stage2_recovery, args.stage2)):
        if src.exists():
            chart(src)
        else:
            print(f"skip {chart.__name__}: {src} not found")
