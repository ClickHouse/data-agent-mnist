# data-agent-mnist

A harness for measuring how well LLM agents answer analytical questions against a
data warehouse, by running the agentic loop, building majority-vote ground truth,
and scoring with a provider-diverse judge panel.

**This repository is the harness, not the benchmark.** It ships no questions, no
warehouse and no results. Those are ours and stay private. What is here is the
machinery that produced them, so the method can be inspected, criticised and run
against your own warehouse.

Staged as `README.md` from `README.harness.md` in the source tree, because the
monorepo's own README describes the board run rather than the harness.

## Start here

`example/README.md` is a complete run against a small synthetic warehouse that
ships with the harness: three API keys to a scored board, in four commands. It is
also this repository's acceptance test. If the example cannot run from this tree
alone, the boundary between harness and benchmark is drawn in the wrong place.

## What it does

Four stages, each a script you can run on its own.

| | |
|---|---|
| `05_annotate.py` | runs several models independently over your questions and keeps the result set at least two of three agree on. Ground truth is the agreement, not any one model's answer. |
| `06_eval.py` | runs each candidate through the agentic loop and scores it against that ground truth, with a judge panel seating one model per provider so no provider can hold a majority of the votes. |
| `08_results_stats.py` | the board: pass rate, standard errors, paired difference tests. |
| `09` to `18` | analyses. Failure modes, judge bias, the flat-mart versus dimensional-layer split, a contamination probe, turn-budget ceilings. |

`bench.py` holds the agentic loop and the judge. `warehouse.py` wraps the
warehouse and the schema prompt. `registry.py` reads the model catalog from
configuration.

## Pointing it at your own warehouse

By configuration, not by editing code.

```
DAM_MODELS_CONFIG   the model registry; copy config/models.example.yaml
DAM_DATA_ROOT       where your datasets live
DAM_QUESTIONS       your question set (or pass --questions)
```

and per run: `--db-path`, `--system-prompt`, `--probe-table`,
`--snapshot-column`. Those four describe a warehouse: where it is, how to explain
its schema to a model, and which fact table carries the row count and the
snapshot date.

The schema prompt is the part worth spending time on. It is the model's only
description of your warehouse, and much of what the harness measures is how well
models navigate what it tells them.

Defaults throughout are the ones our own board uses, and this repository ships
none of the data behind them. Each fails with the flag to pass rather than a
missing-file error.

## Known limitations

Worth reading before trusting a number.

- **Result comparison assumes money.** Numerics are rounded to two decimal places
  and compared within 5%, which is right for spend and wrong for small
  magnitudes: `0.0001` and `0.0002` compare equal. A domain with concentrations,
  probabilities or rates needs the tolerance made configurable first.
- **The warehouse is a local chDB store.** Pointing at a live cluster means
  exporting a snapshot or replacing the `Warehouse` class. Deliberate, since a
  benchmark wants a warehouse that does not move underneath it.
- **Judges are models.** Two runs of the same questions can disagree, by more
  than you would like on a small question set. The example shows this happening
  on purpose.

## Requirements

Python 3.13 and [uv](https://docs.astral.sh/uv/). `uv sync` installs everything.
Provider access is whatever your registry names: the shipped example needs three
API keys and no cloud account.
