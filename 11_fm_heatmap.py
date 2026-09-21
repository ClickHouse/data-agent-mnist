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
from paths import SYNTH_DIR
FM_LABELS = SYNTH_DIR / "fm_labels.jsonl"
FMS = ["FM1", "FM2", "FM3", "FM4", "FM5"]

# Board order (pass rate at the 60-turn budget, desc). A model that has failure
# labels but is missing here used to vanish from the table without a word, so
# compute() now refuses instead. Display names are kept in sync with 09.
ORDER = ["fable51", "deepseek-v4-pro-0813", "kimi-k3", "gemini-3.1-pro-preview",
         "fable5", "opus48", "gpt-5.5", "qwen3.8-max", "kimi-k2.6", "glm-5.2",
         "gpt-5.6", "opus5", "gemini-3.8-flash", "sonnet46",
         "deepseek-v4-flash-0731", "gemini-3.7-flash", "opus47", "sonnet5",
         "deepseek-v3.2", "haiku45", "gemma-4-31b", "kimi-k2-thinking", "o4-mini",
         "gemini-3.5-flash", "gemini-2.5-pro", "qwen3-coder-480b",
         "qwen3-coder-30b", "gemini-2.5-flash", "gpt-4.1"]
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
         "gemini-3.8-flash": "Gemini 3.8 Flash",
         "fable51": "Claude Fable 5.1",
         "qwen3-coder-480b": "Qwen3-Coder 480B", "gemini-3.1-pro-preview": "Gemini 3.1 Pro (prev.)",
         "deepseek-v3.2": "DeepSeek V3.2", "gemma-4-31b": "Gemma 4 31B", "gpt-4.1": "GPT-4.1",
         "gemini-2.5-flash": "Gemini 2.5 Flash", "gemini-3.7-flash": "Gemini 3.7 Flash",
         "qwen3-coder-30b": "Qwen3-Coder 30B",
         "nova-pro": "Amazon Nova Pro", "nova-lite": "Amazon Nova Lite",
         "nova-micro": "Amazon Nova Micro"}


def _retired() -> set[str]:
    """Retired model keys, when the registry is readable.

    This script is offline by design and the published mirror ships only
    config/models.example.yaml, so importing the registry at module scope would
    break it there. Without a registry every labelled key has to be in ORDER,
    which is the stricter reading and the right one when we cannot tell a
    retired candidate from a forgotten one.
    """
    try:
        from registry import RETIRED_CANDIDATES
    except (ImportError, FileNotFoundError):
        return set()
    return set(RETIRED_CANDIDATES)


def compute():
    cnt = defaultdict(Counter)
    for line in FM_LABELS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cnt[r["model"]][r["fm"]] += 1
    retired = _retired()
    missing = sorted(m for m in cnt if m not in ORDER and m not in retired)
    if missing:
        raise SystemExit(
            f"fm_labels.jsonl has models with no place in ORDER: {', '.join(missing)}.\n"
            f"  Add them to ORDER (and to NAMES) in this file, in board order.")
    rows = []
    for m in ORDER:
        c = cnt[m]
        tot = sum(c.values())
        if not tot:            # on the board, not labelled in this corpus
            continue
        rows.append((NAMES.get(m, m), [100 * c[f] / tot for f in FMS], tot))
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
