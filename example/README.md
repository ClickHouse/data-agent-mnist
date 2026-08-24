# Runnable example

A complete run of the harness against a small synthetic warehouse, from three API
keys to a scored board. Nothing here touches the private benchmark, so this is
also the check that the harness works for someone who has none of our data: if
this cannot run from the published tree alone, the split between harness and
benchmark is wrong.

## What you need

Three API keys, each a direct signup, no cloud account:

```
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export FIREWORKS_API_KEY=...
export DAM_MODELS_CONFIG=$PWD/config/models.example.yaml
export DAM_DATA_ROOT=$PWD/example/out          # any writable path
```

`DAM_DATA_ROOT` is where benchmark datasets live. The example does not use any, but
the modules it loads resolve the root at import and fail loudly when it is missing,
so it needs to point somewhere that exists. Any directory will do.

Three providers is not decoration. Ground truth is the result set at least two of
three annotators agree on, and the judge panel seats one model per provider so no
provider can hold a majority of the votes. With two providers both degrade: "at
least two agree" becomes "all must agree", and a split judge vote scores `tie`
instead of resolving.

## Run it

```bash
# 1. load the committed CSVs into a chDB store (~1s, no network)
uv run example/seed_example.py

# 2. build majority-vote ground truth with three annotators
uv run 05_annotate.py \
  --questions example/questions.jsonl \
  --db-path example/warehouse --system-prompt example/schema.md \
  --probe-table marts.usage_daily --snapshot-column day \
  --out example/out/annotated.jsonl --no-classify

# 3. score every candidate against it
uv run 06_eval.py \
  --annot example/out/annotated.jsonl --out example/out/results.jsonl \
  --db-path example/warehouse --system-prompt example/schema.md \
  --probe-table marts.usage_daily --snapshot-column day --no-verify-db

# 4. the board
uv run 08_results_stats.py \
  --results example/out/results.jsonl --annot example/out/annotated.jsonl
```

Steps 2 and 3 call models, so they cost money: 8 questions times 3 annotators,
then 8 times 7 candidates plus a 3-judge panel on each. A few cents and a few
minutes at the time of writing.

## What you should see

```
8 questions x 7 models (0 infra-error cells excluded)

model                      pass    SE           95% CI      z   p(2s)  answered   tl
claude-sonnet             100.0   0.0 [100.0, 100.0]     --      --    100.0    0
gpt-4.1-mini              100.0   0.0 [100.0, 100.0]     --   1.000    100.0    0
kimi                      100.0   0.0 [100.0, 100.0]     --   1.000    100.0    0
o4-mini                   100.0   0.0 [100.0, 100.0]     --   1.000    100.0    0
claude-haiku               87.5  12.5 [ 63.0, 112.0]   1.00   0.317     87.5    0
gpt-4.1                    87.5  12.5 [ 63.0, 112.0]   1.00   0.317     87.5    0
glm                        81.2  13.2 [ 55.5, 107.0]   1.43   0.154     81.2    0

Column-linker firing (judge-side equivalence checks):
  candidate_results_compared: 54
  identity_fast_path: 33
  needs_linker: 21
  needs_linker_pct: 38.9
  distinct_colset_pairs: 15
  questions_no_gt_cols: 0
```

**This is not a benchmark result and should not be read as one.** Four of seven
models tie at 100% here, so these eight questions do not discriminate between
them, and nothing on the board separates at any conventional threshold (the
largest gap is p = 0.15).

Two runs of this example, same questions and same models, produced different
orderings: `claude-haiku` and `gpt-4.1` scored 100% in one and 87.5% in the other,
and `glm` moved between 81.2 and 87.5. That is the point rather than a caveat.
Eight questions cannot resolve differences of a few points, which is why the real
board uses 201. Expect your own run to differ again: the models are sampled and
the judges are models too.

## The warehouse

Deliberately shaped, not arbitrary. Two layers, because the benchmark's central
question is whether a model finds the right one:

- **`marts.usage_daily`** is a denormalized mart, one row per tenant per day, with
  tenant attributes carried on every row. Most questions need this table alone.
- **`crm.*`** is a small star schema. Nothing in it carries a tenant name, so
  answering from here means resolving a name to an account via `dim_account`, then
  joining facts on `account_key`.

A benchmark over one flat table would run and demonstrate nothing, because every
model can filter a single table and the question of layer choice never arises. Of
the eight questions, four are answerable from the mart alone and four need the
CRM hop.

8 tenants, 120 days, 960 usage rows. Small on purpose: three annotators over eight
questions is minutes, not hours.

The rows are committed as CSV under `example/data/`, and `seed_example.py` only
loads them into a chDB store. Two reasons to ship the data rather than generate it
at seed time. It is reviewable: 70 KB of CSV shows exactly what you are about to
run against, where a chDB directory is 60 binary files that cannot be diffed. And
it is stable: the names come from Faker, whose output changes between releases,
and two of the eight questions name a tenant directly (`ex-003` Hughes Ltd,
`ex-005` Hull-Hart). Generating at seed time would let a Faker upgrade rewrite the
warehouse under those questions, breaking two and leaving six passing.

`generate_example_data.py` is how the CSVs were made and is not part of running the
example. Faker rather than hand-invented names: inventing was tried first and is a
trap, since screening a list of plausible company names turned up real firms behind
most of them, several in the same sector, and even coined-looking tokens were
registered businesses. Account owners are identifiers (`analyst-1`) rather than
person names, because no invented person name is nobody's real name.

`example/warehouse/` and `example/out/` are generated and gitignored: chDB rewrites
its store directory on open, so a committed copy drifts on every run.

## Adapting it

Point the harness at your own data by changing configuration, not code:

- `DAM_MODELS_CONFIG` selects the model registry. Copy
  `config/models.example.yaml` and put your own models in it.
- `DAM_DATA_ROOT` selects where benchmark datasets live.
- `--db-path`, `--system-prompt`, `--probe-table` and `--snapshot-column` describe
  a warehouse: where it is, how to explain its schema to a model, and which fact
  table to use for the populated check and the snapshot date.

The schema prompt is the part worth spending time on. It is the model's only
description of your warehouse, and the harness measures how well models navigate
what it tells them.
