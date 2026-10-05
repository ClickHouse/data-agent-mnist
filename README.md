# data-agent-mnist

**Turn your own data warehouse into a benchmark for analytical agents.**

There is no single best model for everyone. Which one is right depends on your
schema, your join depth, and the questions your people actually ask, and a public
leaderboard cannot tell you any of that. This is the machinery for building the
benchmark that can: it takes your warehouse and your questions and produces a
scored board, with ground truth you did not have to hand-write.

## Why not an existing benchmark

Text-to-SQL benchmarks translate one question into one query against a schema
handed to the model up front, and score a string match against a single gold
query. A decade of them rests on those assumptions. Agentic analytics breaks all
of them: the agent discovers the schema itself, takes as many turns as it needs,
issues several queries, and is judged on whether it answered the question rather
than on how it phrased the SQL.

The difficulty also lives somewhere those benchmarks do not look. It is in the
size and multi-hop join structure of a real warehouse, not in a compact
hand-built schema. A benchmark that abstracts the schema away measures the wrong
thing, which is why this one is built to run against yours.

So it is designed for three properties:

| | |
|---|---|
| **personalised** | your questions, your data, your warehouse. The output is which model is right *for you*, not a global ranking. |
| **schema-aware** | the real schema, at its real width and join depth, because that is where the difficulty is. |
| **replayable** | every evaluation runs against an identical warehouse state, from one command, which a live production system cannot promise. |

## How it works

Four stages. You supply a warehouse and a set of questions; the harness supplies
everything after that.

**1. A replayable warehouse.** Deterministic seeding from committed data, so every
run scores against identical state. If your questions came from production
traffic they will name real entities, so the seeder plants each referenced entity
under exactly the identifier its question uses, then buries it in a synthetic
population. Present but not conspicuous: a question whose referent is missing is
unanswerable, and one that is the only row in the table is trivial.

**2. Ground truth without an answer key.** Real questions do not arrive with
validated answers, and hand-authoring SQL over a wide warehouse neither scales
nor produces anything better than one person's opinion. Instead several models
from different providers each solve every question independently, running the
full agentic loop. Where at least two agree by independent paths, that result
becomes ground truth; where all disagree, the question is dropped as not reliably
answerable. Provider diversity is the point: two labs' models agreeing
independently is much harder to explain away as shared training bias.

**3. Scoring by a panel, not a judge.** One seat per provider, so no provider
holds a majority, and a seat never goes to the candidate itself, so no model
scores its own answers. A judge from the candidate's provider does vote, and
`10_judge_bias.py` and `13_judge_bt.py` measure how much that matters. A
deterministic equivalence check runs first and its
verdict reaches the panel as an authoritative data signal, because two correct
answers can disagree on form: one names a column `total_dollar_usage`, the other
`monthly_spend`. When one answer's columns contain the other's, the shared
columns map by identity and no model is consulted; a column-linker model is asked
only when the two column sets genuinely differ. In our own run, 84.9% of scored
result sets matched only through that linked mapping.

![How a candidate answer is scored](docs/judge-panel.png)

**4. A board.** Pass rate with standard errors and paired difference tests, plus
analyses for failure modes, judge bias, turn-budget ceilings, contamination, and
whether models find the dimensional layer or stop at the flat one.

## The repository is the harness, not the benchmark

It ships no questions, no warehouse and no results beyond the two worked
examples. What is here is the machinery, so the method can be inspected,
criticised and pointed at a warehouse of your own.

## Getting started

Python 3.13 and [uv](https://docs.astral.sh/uv/). Three API keys, each a direct
signup with no cloud account attached.

```bash
uv sync

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export FIREWORKS_API_KEY=...
export DAM_MODELS_CONFIG=$PWD/config/models.example.yaml
export DAM_DATA_ROOT=$PWD/examples/saas/out        # any writable path

# 1. build the example warehouse (loads committed CSVs into chDB, ~1s, no network)
uv run examples/saas/seed_example.py

# 2. ground truth: three models answer independently, majority agreement wins
uv run 05_annotate.py \
  --questions examples/saas/questions.jsonl \
  --db-path examples/saas/warehouse --system-prompt examples/saas/schema.md \
  --probe-table marts.usage_daily --snapshot-column day \
  --out examples/saas/out/annotated.jsonl --no-classify

# 3. score every candidate against it
uv run 06_eval.py \
  --annot examples/saas/out/annotated.jsonl --out examples/saas/out/results.jsonl \
  --db-path examples/saas/warehouse --system-prompt examples/saas/schema.md \
  --probe-table marts.usage_daily --snapshot-column day --no-verify-db

# 4. the board
uv run 08_results_stats.py \
  --results examples/saas/out/results.jsonl --annot examples/saas/out/annotated.jsonl
```

Steps 2 and 3 call models, so they cost money: 8 questions by 3 annotators, then
8 by 7 candidates plus a 3-judge panel on each. A few cents and a few minutes.

Step 1 and the test suite (`uv run pytest tests -q`) reach no provider, so the
keys above can stay as placeholders, but they still need `DAM_MODELS_CONFIG` and
`DAM_DATA_ROOT`: the modules resolve a registry and a data root at import.
There is deliberately no default registry, so importing without one tells you to
supply it rather than quietly running the example catalog as though it were
yours. `.github/workflows/tests.yml` is this paragraph as a runnable file.

Three providers is not decoration. Ground truth is the result set at least two of
three annotators agree on, and the judge panel seats one model per provider so no
provider can hold a majority of the votes. With two, "at least two agree" becomes
"all must agree" and a split judge vote scores a tie instead of resolving.

### Two instances

`examples/` holds two worked warehouses, and the commands above run the first.

- **`saas/`** is generic: a flat daily mart plus a small CRM star schema, `Date`
  grain, short column names. Start here.
- **`clickhouse-dwh/`** is a cloud-database vendor's shape, with the real table
  and column names of the warehouse the harness was built against and wholly
  synthetic rows. Longer names with a `__` convention, a `DateTime` grain and so
  timezone sensitivity, and one more hop to reach a fact.

They exist as a pair on purpose. One instance shows the harness is reproducible;
two of the same shape show nothing more. These differ on the axes that break
portability, so running both is the evidence that the schema is configuration
rather than an assumption. Each directory has its own README and its own
commands, and both are this repository's acceptance test: if they cannot run from
this tree alone, the boundary between harness and benchmark is drawn in the wrong
place.

## What a run checks before it spends money

A board run calls models for hours. Two guards run at the start of `06_eval.py`
so that a run which cannot be trusted stops before the first paid call, instead
of producing a number that looks like a result.

**The warehouse still reproduces the board.** `verify_board.py --emit` writes a
manifest that pins every question to the warehouse it was scored against,
records the hashes of the ground truth and the results, and records the chDB
engine release and the session timezone the board was built with. Before
scoring, the eval re-runs the cached candidates' own SQL against the mounted
warehouse and refuses to continue if the stored results no longer reproduce,
because that means the warehouse moved underneath the ground truth. It also
refuses when the engine version or the session timezone differs from the
manifest: that is the wrong environment, not drift to measure. Pin `chdb-core`
in the environment you score with (the `chdb` package does not pin the engine),
and pass `--session-timezone` whenever the warehouse has a `DateTime` grain,
since values render in the session zone and the snapshot date moves with the
reader's locale. `--no-verify-db` skips the guard, which the example commands do
because an example warehouse has no manifest yet.

**Every credential works.** `preflight.py` resolves every provider the selected
candidates, the judge seats and the column linker will touch: an identity call
for AWS, a token mint for Vertex application-default credentials, and key
presence for the API-key providers. A missing or unusable credential aborts the
run. The judge seats and the linker matter most here: they sit on different
providers from the candidate, and a dead one does not stop the expensive half
of the run, it degrades the scoring one question at a time. `--no-preflight`
skips the check.

## The scripts

One per stage, each runnable on its own.

The numbering starts at `05` because it is the whole pipeline's, and the first
four stages are not here. They pull traces from a live warehouse, curate
questions from them, and anonymise the result, so they are specific to one
warehouse and they handle personal data. Seeding is stage `03`; the harness
seeds from `examples/saas/seed_example.py` instead, which needs neither. Nothing
downstream of `05` depends on them: give the pipeline a `questions.jsonl` and a
warehouse and it runs.

| | |
|---|---|
| `05_annotate.py` | stage 2 above: builds ground truth by agreement. |
| `06_eval.py` | stage 3: runs each candidate through the agentic loop and scores it against that ground truth. |
| `06b_split_eval.py` | the same, for a board whose questions were scored against more than one warehouse epoch: routes each question to the warehouse its ground truth was computed on. |
| `verify_board.py` | the integrity guard described above; `--emit` writes the manifest, the default mode checks the whole board against it. |
| `preflight.py` | the credential check described above; imported by `06_eval.py`, not run on its own. |
| `08_results_stats.py` | the board: pass rate, standard errors, paired difference tests. |
| `07_failure_modes.py`, `11_fm_heatmap.py` | label every failure as one of five modes (no attempt, wrong plan, wrong data selection, wrong implementation, runtime error) and plot the per-model shares. |
| `07_thinking_effort_sweep.py` | the same loop over a grid of thinking and reasoning-effort settings for one model, so a budget-shaped knob is measured rather than assumed. |
| `09_dds_analysis.py` | splits the board by whether the ground-truth SQL reaches the dimensional layer, and reports how often each model found it at all. |
| `10_judge_bias.py` | judge leniency and in-group residual per seat, from the recorded votes. |
| `13_judge_bt.py` | a many-facet Rasch model over the same votes: candidate ability, judge severity and an own-family coefficient, with bootstrap intervals. `--selftest` recovers planted gaps with no board data and no credentials. |
| `12_contamination_probe.py` | asks whether the questions are in a model's training data: a completion probe, an entity-recovery probe, and a public benchmark as positive control. |
| `18_ceiling_summary.py` | pass rate at every turn budget from one deep run per model: the budget is never announced to the model, so a run at a smaller budget is the deep run stopped early. |

`bench/` holds the agentic loops (`runners.py`, `librechat.py`), the judge
(`judge.py`, `completion.py`), the result-set scoring (`scoring.py`) and the
provider clients (`clients.py`). `warehouse.py` wraps the
warehouse and the schema prompt. `registry.py` reads the model catalog from
configuration. The board and the analyses in `08`, `09`,
`10`, `11`, `13` and `18` are offline: they read `results.jsonl` and call no
model. The failure-mode labeler, the effort sweep and the contamination probe
do call models.

## Where candidates run

A candidate is one entry in the model registry with a `provider`. The runners
cover OpenAI, Anthropic's Messages API, Fireworks, AWS Bedrock, Vertex AI, and
any OpenAI-compatible endpoint under `provider: gateway`, which is how a vLLM,
Ollama or self-hosted model joins the board. Reasoning models are flagged in the
registry, and the runners send the token argument and the reasoning settings
each API expects.

Two hooks sit in the loop itself.

**A gate on every query.** Every runner accepts an optional `gate(sql, result)`
callback. It sees each executed query and its result, and may return a note that
is prepended to the tool message the model reads next. The result the judge
grades is stored unchanged, so a gated run and its baseline compare at an equal
turn budget. This is the hook for experiments that steer the agent mid-run, for
instance a schema-retrieval hint when a query reads a table the question does
not need.

**The product instead of the loop.** `provider: librechat` drives a LibreChat
agent over its HTTP API as the candidate, under the board's own system prompt,
and reads the trajectory back from the agent's Langfuse trace, so the same
questions and the same panel score the product surface rather than the harness's
own loop. The difference between the two numbers is the result. The instance
URL and login come from `LIBRECHAT_BASE_URL`, `LIBRECHAT_EMAIL` and
`LIBRECHAT_PASSWORD`, with `LIBRECHAT_TENANT_ID` for a multi-tenant deployment.
The agent's SQL tool must query the same warehouse the direct runners use, or
the ground truth stops applying; wiring that tool is part of your deployment,
not of this repository.

## Optional: ranking and labeling with Jev

Two tools use [Jev](https://docs.typesafe.ai), TypeSafe's System One classifier, as a
cheap typed-judgment component rather than a text generator. They are optional: install
the extra and set a TypeSafe key.

```bash
uv sync --extra jev            # or run any script with: uv run --with typesafe-sdk ...
export TYPESAFE_API_KEY=...
```

- `schema_retrieval.py` ranks the warehouse's tables by how likely each is needed for a
  question (one yes/no judgment per table), scored against the gold SQL's tables
  (nDCG@10, recall@k). `schema_retrieval_agent.py` feeds that ranking back to the agent,
  up front or in-loop when it diverges onto a low-ranked table, and measures whether it
  helps. Pass `--table-namespaces` for your schema's prefixes (the examples use
  `marts,crm`).
- `label_failure_modes.py` reads a graded run and labels each failed cell with a failure
  sub-mode, which stated rule would have prevented it, and whether the grader itself
  erred. The rule set is configuration (`config/rules.example.yaml`; copy it for your
  warehouse).

Both read the schema from your warehouse or examples and name no fixed table, and
`TYPESAFE_BASE_URL` points them at a gateway instead of the public API.

## Pointing it at your own warehouse

By configuration, not by editing code.

```
DAM_MODELS_CONFIG   the model registry; copy config/models.example.yaml
DAM_DATA_ROOT       where your datasets live
DAM_CORPUS          the dataset directory under that root (default text2sqlbench-synthetic)
DAM_QUESTIONS       your question set (or pass --questions)
```

and per run: `--db-path`, `--system-prompt`, `--probe-table`,
`--snapshot-column`, and `--session-timezone` for a `DateTime` grain. The first
four describe a warehouse: where it is, how to explain its schema to a model,
and which fact table carries the row count and the snapshot date.

`DAM_CORPUS` exists so that a second corpus is one variable rather than a forked
pipeline: a development set that people iterate against can live beside a board
that stays frozen, and every stage follows the variable.

The schema prompt is the part worth spending time on. It is the model's only
description of your warehouse, and much of what the harness measures is how well
models navigate what it tells them.

**Numeric comparison is a policy in the registry.** The default is built for
money: round to two decimal places, allow 5% relative drift, no absolute floor.
That catches 59K against 70K and ignores sub-cent noise, and it is wrong for a
warehouse of concentrations, probabilities or counts, where a factor-of-two
error fits inside 5% and two small values both round to zero. A `scoring`
section in the model config sets the policy, and both the annotate and the eval
stage apply it:

```yaml
scoring:
  round_decimals: null   # keep full precision
  rel_tol: 0.0
  abs_tol: 1.0e-9
```

Two numbers agree when they are within the absolute or the relative tolerance.
`config/models.example.yaml` carries the currency default with the alternative
shown beside it.

Defaults name the dataset the harness was developed against, which is not
included here. Each one fails with the flag to supply rather than a missing-file
error.

## Known limitations

Worth reading before trusting a number.

- **The warehouse is a local chDB store.** Pointing at a live cluster means
  exporting a snapshot or replacing the `Warehouse` class. Deliberate, since a
  benchmark wants a warehouse that does not move underneath it.
- **Judges are models.** Two runs of the same questions can disagree, by more
  than you would like on a small question set. The example shows this happening
  on purpose.
- **The guard protects a board, not a first run.** Until `verify_board.py --emit`
  has written a manifest for your warehouse, `06_eval.py` needs `--no-verify-db`,
  and nothing checks that the warehouse matches the ground truth you built.
  Emit the manifest as soon as the first eval has written results.
- **The product-surface runner needs a trace store.** It reads the agent's turns,
  SQL and token usage from Langfuse, so a LibreChat deployment without tracing
  cannot be scored this way.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
