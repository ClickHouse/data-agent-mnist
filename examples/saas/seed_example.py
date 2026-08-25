"""Seed the example warehouse: a small, wholly synthetic analytics DWH.

This exists so the harness can be run by someone who has none of our data. It is
not a miniature of our warehouse and does not need to be; what it has to preserve
is the SHAPE that makes the benchmark interesting:

  * a flat mart, one row per tenant per day, where most questions are answerable
    by filtering and aggregating a single table;
  * a small dimensional layer that needs a multi-hop join to resolve a name to an
    account and then to its facts.

That split is the whole tension the board measures. A benchmark over one flat
table would run fine and demonstrate nothing, because every model can filter one
table and the question of whether a model finds the right layer never arises.

THE DATA IS COMMITTED AS CSV under examples/saas/data/, and this script only loads it.
It was generated once with Faker (see generate_example_data.py) and then frozen,
for two reasons. It is reviewable: 70 KB of CSV shows exactly what ships, where a
chDB directory is 60 opaque files. And it is stable: Faker's output can change
between releases, and examples/saas/questions.jsonl names tenants directly, so a Faker
upgrade would silently break two questions and leave the rest passing.

Faker rather than hand-invented names, for the original generation. Inventing was
tried first and is a trap: screening a list of plausible-sounding companies turned
up real firms behind most of them, several in the same sector, and even
coined-looking tokens were registered businesses. Faker guarantees nothing either,
but nothing here is concealed, so there is no assurance to falsify. That is the
difference from 02b_anonymize, where a name colliding with the one it replaced
would make a scrub look complete while a real name survived.

    uv run examples/saas/seed_example.py [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdb import session as chdb_session  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "warehouse"
CSV_DIR = Path(__file__).resolve().parent / "data"

DDL = """
CREATE DATABASE IF NOT EXISTS marts;
CREATE DATABASE IF NOT EXISTS crm;

-- flat mart: one row per tenant per day, joins already done
--
-- ORDER BY (day, tenant_id), not (tenant_id, day). A sparse primary index only
-- helps a query that filters on a PREFIX of the sort key, and five of the eight
-- questions here filter a time window without naming a tenant, so leading with
-- tenant_id would make those full scans. Leading with day keeps them index-served
-- and still prunes the per-tenant questions once the range narrows.
--
-- The fixture is 960 rows, so nothing here is measurable. It is written this way
-- because a published ClickHouse example gets copied, and a sort key that
-- defeats its own query pattern teaches the wrong habit.
CREATE TABLE IF NOT EXISTS marts.usage_daily (
    day             Date,
    tenant_id       String,
    tenant_name     String,
    -- LowCardinality for the columns with a handful of distinct values (3 plans,
    -- 5 regions). Dictionary-encoded rather than storing the string per row.
    plan            LowCardinality(String),
    region          LowCardinality(String),
    queries         UInt32,
    storage_gb      Float64,
    compute_minutes Float64,
    -- Money is Decimal, not Float64. Float cannot represent most decimal
    -- fractions exactly, so sums drift and equality is unreliable; Decimal64(2)
    -- is exact to the cent. Measures that are genuinely continuous
    -- (storage, compute minutes) stay Float64.
    daily_spend     Decimal64(2)
) ENGINE = MergeTree ORDER BY (day, tenant_id);

-- dimensional layer: answering from here needs a name -> account -> fact hop
CREATE TABLE IF NOT EXISTS crm.dim_account (
    account_key  String,
    tenant_id    String,
    account_name String,
    owner        LowCardinality(String),
    signed_on    Date
) ENGINE = MergeTree ORDER BY account_key;

-- The fact tables keep account_key first: every question reaching them resolves
-- a name to an account before filtering, so the account IS the prefix.
CREATE TABLE IF NOT EXISTS crm.fct_opportunity (
    opportunity_id String,
    account_key    String,
    stage          LowCardinality(String),
    amount         Decimal64(2),
    opened_on      Date,
    -- Nullable, deliberately, against the general advice to prefer a DEFAULT.
    -- Here NULL is the fact being recorded: the opportunity has not closed. A
    -- sentinel date would need every reader to know the sentinel, and the
    -- questions ask "still open", which is exactly IS NULL.
    closed_on      Nullable(Date)
) ENGINE = MergeTree ORDER BY (account_key, opened_on);

CREATE TABLE IF NOT EXISTS crm.fct_support_case (
    case_id     String,
    account_key String,
    priority    LowCardinality(String),
    opened_on   Date,
    closed_on   Nullable(Date)
) ENGINE = MergeTree ORDER BY (account_key, opened_on);
"""

def _statements(ddl: str):
    """Split DDL into statements, ignoring punctuation inside comments.

    A naive ddl.split(";") breaks on a semicolon in a `--` comment, cutting a
    CREATE TABLE in half and reporting "Unmatched parentheses" from a line that
    looks fine. That happened while documenting a column choice. The comments are
    for the reader of this file, not for the server, so they are stripped before
    splitting rather than being a hazard for whoever edits the DDL next.
    """
    stripped = "\n".join(
        line for line in ddl.splitlines() if not line.lstrip().startswith("--"))
    return [s for s in stripped.split(";") if s.strip()]

TABLES = ["marts.usage_daily", "crm.dim_account",
          "crm.fct_opportunity", "crm.fct_support_case"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", type=Path, default=DB_PATH)
    ap.add_argument("--force", action="store_true", help="delete and regenerate")
    args = ap.parse_args()

    if args.db_path.exists():
        if not args.force:
            print(f"{args.db_path} already exists; pass --force to regenerate")
            return
        shutil.rmtree(args.db_path)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    sess = chdb_session.Session(str(args.db_path))
    for stmt in _statements(DDL):
        sess.query(stmt)
    for t in TABLES:
        csv = CSV_DIR / (t.replace(".", "__") + ".csv")
        if not csv.exists():
            raise SystemExit(f"missing {csv}. The example data is committed; "
                             f"regenerate it with generate_example_data.py only if "
                             f"you mean to change the fixture.")
        sess.query(f"INSERT INTO {t} SELECT * FROM file('{csv}', 'CSVWithNames')")
        n = sess.query(f"SELECT count() FROM {t}", "CSV").bytes().decode().strip()
        print(f"  {t:26s} {n:>6s} rows")
    # JSONCompactEachRow, not CSV: tenant names carry commas ("Moon, Ford and
    # Hanson"), so a CSV result cannot be split back apart by hand.
    rows = sess.query("SELECT DISTINCT tenant_name FROM marts.usage_daily "
                      "ORDER BY tenant_name", "JSONCompactEachRow").bytes().decode()
    sess.close()
    names = [json.loads(line)[0] for line in rows.splitlines() if line.strip()]
    print(f"\nseeded {args.db_path} from {CSV_DIR.name}/")
    print("tenants: " + " | ".join(names))


if __name__ == "__main__":
    main()
