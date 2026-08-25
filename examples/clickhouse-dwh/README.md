# Example: a cloud-database vendor's warehouse

The same harness and the same method as `../saas/`, against a warehouse with a
different shape and different naming conventions. **Real table and column names,
wholly synthetic rows.** No organization, revenue figure or support case here
corresponds to anything.

## Why a second instance

One instance shows the harness is reproducible. Two of the same shape show
nothing more. This one differs on the axes that actually break portability:

| | `../saas/` | here |
|---|---|---|
| naming | `tenant_id`, `daily_spend` | `organization__id`, `organization__dollar_usage` |
| date grain | `Date` | `DateTime`, and therefore timezone-sensitive |
| hop to a fact | name to account to fact | name to account to **bridge** to fact |
| column collisions | none | `organization__*` on both the mart and the dim |
| tables | 4 | 4, from 18 in the original |

If the harness handles both without code changes, the schema really is
configuration. That is the claim this directory exists to test.

## Run it

```bash
export DAM_MODELS_CONFIG=$PWD/config/models.example.yaml
export DAM_DATA_ROOT=$PWD/examples/clickhouse-dwh/out

uv run examples/clickhouse-dwh/seed_dwh.py

uv run 05_annotate.py \
  --questions examples/clickhouse-dwh/questions.jsonl \
  --db-path examples/clickhouse-dwh/warehouse \
  --system-prompt examples/clickhouse-dwh/schema.md \
  --session-timezone UTC \
  --out examples/clickhouse-dwh/out/annotated.jsonl --no-classify

uv run 06_eval.py \
  --annot examples/clickhouse-dwh/out/annotated.jsonl \
  --out examples/clickhouse-dwh/out/results.jsonl \
  --db-path examples/clickhouse-dwh/warehouse \
  --system-prompt examples/clickhouse-dwh/schema.md \
  --session-timezone UTC --no-verify-db

uv run 08_results_stats.py \
  --results examples/clickhouse-dwh/out/results.jsonl \
  --annot examples/clickhouse-dwh/out/annotated.jsonl
```

No `--probe-table` or `--snapshot-column` here: this instance happens to use the
names those default to, which is worth noticing. The `../saas/` instance passes
both, and that is the more representative case.

## `--session-timezone UTC` is not optional

`timestamp_hour` is `DateTime`, and chDB parses **and renders** DateTime in the
session timezone. Measured on this fixture:

| session zone | `max(timestamp_hour)` | snapshot date |
|---|---|---|
| `UTC` | `2026-06-30 00:00:00` | 2026-06-30 |
| `America/Los_Angeles` | `2026-06-29 17:00:00` | **2026-06-29** |

The snapshot date is what every relative window in the questions resolves
against, so without the flag the same fixture answers differently depending on
where it is read, with no error to say so. `../saas/` uses `Date` and is immune.
This instance keeps `DateTime` because the original does, and because a benchmark
that quietly depends on the reader's locale is worth being able to demonstrate.

## The two layers

- **`dbt_marts_general.usage_history`** is the mart: one row per organization per
  day, with organization attributes carried on every row. Usage, revenue, tier,
  cloud and region questions need this table alone.
- **`dbt_dds.*`** is a CRM mirror, and nothing in it carries
  `organization__name`. Reaching a support case from a name means resolving
  `dim_account_current.account__name` to `account__key`, then joining
  `fct_case.case__account_key`. Reaching usage from an account attribute needs
  `dim_organization_current` as a bridge back to `organization__id`.

Four of the eight questions sit on each side. That split is what the board
measures: not whether a model can write SQL, but whether it finds the right
layer.

10 organizations, 90 days, 900 usage rows. Small on purpose.

## The data

Committed as CSV under `data/`, loaded by `seed_dwh.py`. Generated once by
`generate_dwh_data.py` and then frozen, for the same two reasons as the other
instance: 60 KB of CSV can be read and diffed where a chDB store cannot, and
Faker's output changes between releases while `questions.jsonl` names
organizations directly.

Names come from Faker. Inventing them was tried first for the other instance and
is a trap: screening a list of plausible company names turned up real firms
behind most of them. Case owners are identifiers (`support-1`) rather than person
names.

`warehouse/` and `out/` are generated and gitignored.
