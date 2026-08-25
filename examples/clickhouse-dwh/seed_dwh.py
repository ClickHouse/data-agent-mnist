"""Seed the cloud-database-vendor example warehouse from committed CSVs.

Real table and column names, wholly synthetic rows. See generate_dwh_data.py for
how the rows were made and why the data is committed rather than generated here.

TIMEZONE. `timestamp_hour` is DateTime, matching the real grain, and chDB parses
AND renders DateTime in the session timezone. Seeding under one zone and reading
under another shifts every value by the offset, which silently moves questions
across day and month boundaries and invalidates ground truth without any error.
The session is pinned to UTC below. That is NOT inherited: Warehouse opens its
own chDB session, so every stage has to be told, which is what
`--session-timezone UTC` in the README's commands is for. Measured on this
fixture, max(timestamp_hour) reads "2026-06-30 00:00:00" under UTC and
"2026-06-29 17:00:00" under America/Los_Angeles, moving the snapshot date a full
day and with it every relative window the questions use.

    uv run examples/clickhouse-dwh/seed_dwh.py [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from chdb import session as chdb_session  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "warehouse"
CSV_DIR = Path(__file__).resolve().parent / "data"

DDL = """
CREATE DATABASE IF NOT EXISTS dbt_marts_general;
CREATE DATABASE IF NOT EXISTS dbt_dds;

-- the wide denormalized daily mart: organization attributes on every row, so
-- most questions need no join at all
CREATE TABLE IF NOT EXISTS dbt_marts_general.usage_history (
    timestamp_hour               DateTime,
    organization__id             String,
    organization__name           String,
    organization__email_domain   String,
    organization__tier           LowCardinality(String),
    organization__billing_model  LowCardinality(String),
    organization__cloud_provider LowCardinality(String),
    organization__region         LowCardinality(String),
    organization__country        LowCardinality(String),
    -- Float64 for money, which is NOT the recommendation: Decimal is exact to
    -- the cent and Float cannot represent most decimal fractions. It stays
    -- because this instance mirrors the real column types of the warehouse the
    -- harness was built against, and that warehouse uses Float64. The sibling
    -- instance under examples/saas uses Decimal64(2), so the pair shows both the
    -- recommended type and what a real warehouse often has.
    organization__dollar_usage   Float64,
    organization__MRR            Float64,
    organization__MRR_PAYG       Float64,
    organization__MRR_Committed  Float64,
    -- UInt64 is wider than these values need (a daily count peaks around a
    -- million here), and narrower is the general advice. Kept for the same
    -- reason as the Float64 above: it is the real column type.
    organization__query_count    UInt64,
    organization__error_count    UInt64
-- ORDER BY (timestamp_hour, organization__id), not the other way round. A sparse
-- primary index only helps a query filtering on a PREFIX of the sort key, and
-- half the questions here filter a time window without naming an organization,
-- so leading with organization__id would make those full scans. Caught in review
-- of this file, and it was wrong in the sibling instance too.
--
-- Which order is right depends on the query mix, and a production warehouse
-- serving mostly per-organization lookups would reasonably invert this. The
-- point for an example is that the choice follows the documented questions
-- rather than being inherited by accident.
--
-- 900 rows, so nothing here is measurable. It is written this way because a
-- published ClickHouse example gets copied.
) ENGINE = MergeTree ORDER BY (timestamp_hour, organization__id);

-- the CRM mirror. Reaching a case from an organization name is two hops:
-- name -> account__key (via the org dim) -> case__account_key.
CREATE TABLE IF NOT EXISTS dbt_dds.dim_organization_current (
    organization__key            String,
    organization__id             String,
    organization__account_key    String,
    organization__tier           LowCardinality(String),
    organization__billing_model  LowCardinality(String),
    organization__cloud_provider LowCardinality(String),
    organization__region         LowCardinality(String),
    organization__country        LowCardinality(String)
) ENGINE = MergeTree ORDER BY organization__key;

CREATE TABLE IF NOT EXISTS dbt_dds.dim_account_current (
    account__key             String,
    account__id              String,
    account__name            String,
    account__status          LowCardinality(String),
    account__industry        LowCardinality(String),
    account__region          LowCardinality(String),
    account__billing_country LowCardinality(String)
) ENGINE = MergeTree ORDER BY account__key;

CREATE TABLE IF NOT EXISTS dbt_dds.fct_case (
    case__case_number String,
    case__account_key String,
    case__subject     String,
    case__category    LowCardinality(String),
    case__status      LowCardinality(String),
    case__is_closed   UInt8,
    case__priority    LowCardinality(String),
    case__origin      LowCardinality(String),
    case__owner_name  String,
    case__created_date Date,
    -- Nullable against the general advice to prefer a DEFAULT: here NULL is the
    -- fact, that the case has not closed, and the questions ask exactly that.
    case__closed_date  Nullable(Date)
) ENGINE = MergeTree ORDER BY (case__account_key, case__created_date);
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

TABLES = ["dbt_marts_general.usage_history",
          "dbt_dds.dim_organization_current",
          "dbt_dds.dim_account_current",
          "dbt_dds.fct_case"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", type=Path, default=DB_PATH)
    ap.add_argument("--force", action="store_true", help="delete and reseed")
    args = ap.parse_args()

    if args.db_path.exists():
        if not args.force:
            print(f"{args.db_path} already exists; pass --force to reseed")
            return
        shutil.rmtree(args.db_path)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    sess = chdb_session.Session(str(args.db_path))
    # See the module docstring: DateTime is rendered in the session zone, so this
    # is what keeps a fixture built anywhere answering the same everywhere.
    sess.query("SET session_timezone = 'UTC'")
    for stmt in _statements(DDL):
        sess.query(stmt)
    for t in TABLES:
        csv = CSV_DIR / (t.replace(".", "__") + ".csv")
        if not csv.exists():
            raise SystemExit(f"missing {csv}. The data is committed; regenerate "
                             f"with generate_dwh_data.py only to change the fixture.")
        sess.query(f"INSERT INTO {t} SELECT * FROM file('{csv}', 'CSVWithNames')")
        n = sess.query(f"SELECT count() FROM {t}", "CSV").bytes().decode().strip()
        print(f"  {t:38s} {n:>6s} rows")

    rows = sess.query("SELECT DISTINCT organization__name FROM "
                      "dbt_marts_general.usage_history ORDER BY organization__name",
                      "JSONCompactEachRow").bytes().decode()
    span = sess.query("SELECT min(timestamp_hour), max(timestamp_hour) FROM "
                      "dbt_marts_general.usage_history",
                      "JSONCompactEachRow").bytes().decode().strip()
    sess.close()
    names = [json.loads(l)[0] for l in rows.splitlines() if l.strip()]
    print(f"\nseeded {args.db_path} from {CSV_DIR.name}/")
    print(f"span (UTC): {span}")
    print("organizations: " + " | ".join(names))


if __name__ == "__main__":
    main()
