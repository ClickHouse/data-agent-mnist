"""
Failure-mode breakdown heatmap from fm_labels.jsonl (offline).

Row-normalises each model's concluded failures across the DAB modes (FM1 no
attempt, FM2 wrong plan, FM3 wrong data, FM4 wrong implementation, FM5 runtime
error; turn-limited runs are excluded upstream as "ran out of budget") and
prints the per-model percentages plus the FM2 range across models.

    uv run 11_fm_heatmap.py [--plot out.pdf]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from paths import DATA
FM_LABELS = DATA / "text2sqlbench-synthetic/fm_labels.jsonl"
FMS = ["FM1", "FM2", "FM3", "FM4", "FM5"]

# Display names + board order (overall pass rate, desc) — keep in sync with 09.
ORDER = ["opus48", "fable5", "kimi-k3", "gpt-5.6", "gpt-5.5", "opus47", "opus5", "qwen3.8-max", "sonnet5",
         "sonnet46", "glm-5.2", "deepseek-v4-pro-0813", "gemini-2.5-pro",
         "deepseek-v4-flash-0731", "haiku45", "kimi-k2.6", "o4-mini",
         "qwen3-coder-480b", "gemini-3.5-flash", "kimi-k2-thinking", "gemma-4-31b", "deepseek-v3.2",
         "gemini-3.1-pro-preview", "gemini-2.5-flash", "gpt-4.1",
         "qwen3-coder-30b", "gemini-3.7-flash"]
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


def compute():
    cnt = defaultdict(Counter)
    for line in FM_LABELS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cnt[r["model"]][r["fm"]] += 1
    rows = []
    for m in ORDER:
        c = cnt[m]
        tot = sum(c.values())
        rows.append((NAMES.get(m, m), [100 * c[f] / tot if tot else 0 for f in FMS], tot))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()
    rows = compute()
    print(f"{'model':24}" + "".join(f"{f:>6}" for f in FMS) + f"{'n':>6}")
    for name, pct, tot in rows:
        print(f"{name:24}" + "".join(f"{p:6.0f}" for p in pct) + f"{tot:6d}")
    fm2 = [pct[1] for _, pct, _ in rows]
    print(f"\nFM2 (wrong plan) range across models: {min(fm2):.0f}%..{max(fm2):.0f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        mat = np.array([pct for _, pct, _ in rows])
        labels = [name for name, _, _ in rows]
        fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(labels) + 1.0))
        ax.imshow(mat, cmap="YlOrBr", vmin=0, vmax=90, aspect="auto")
        ax.set_xticks(range(len(FMS)))
        ax.set_xticklabels(["FM1\nno attempt", "FM2\nwrong plan", "FM3\nwrong data",
                            "FM4\nwrong impl.", "FM5\nruntime"], fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for k in range(len(FMS)):
                ax.text(k, i, f"{mat[i, k]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="black" if mat[i, k] < 55 else "white")
        ax.set_title("Failure modes, % of concluded failures (row-normalised)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(args.plot, bbox_inches="tight")
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
