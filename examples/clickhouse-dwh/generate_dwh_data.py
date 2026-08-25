"""Regenerate examples/clickhouse-dwh/data/*.csv, the fixture for this instance.

You almost certainly do not need to run this. The CSVs are committed and
seed_dwh.py loads them; this records how they were made.

WHAT THIS IS. The same harness, the same method, against a warehouse shaped like
a cloud-database vendor's rather than a generic SaaS app: a wide denormalized
daily mart carrying organization attributes on every row, plus a Salesforce
mirror that has to be joined through. Real table and column names, so a reader
sees the naming conventions the questions were written against
(`organization__id`, `case__account_key`, the double underscore). The DATA is
generated and wholly synthetic. No customer, no revenue figure and no support
case here corresponds to anything.

WHY A SECOND INSTANCE. The other one under examples/ is deliberately generic, and
two instances of the same shape would only show the harness is reproducible. This
one differs on the axis that actually breaks portability: naming convention, join
depth, a DateTime rather than Date grain, and columns whose names collide across
tables. If the harness handles both, the schema really is configuration.

    uv run examples/clickhouse-dwh/generate_dwh_data.py
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from faker import Faker

CSV_DIR = Path(__file__).resolve().parent / "data"
SEED = 20260824
SNAPSHOT = date(2026, 6, 30)
DAYS = 90
N_ORGS = 10

TIERS = ["Enterprise", "Growth", "Free"]
BILLING = ["Committed", "PAYG"]
CLOUDS = ["aws", "gcp", "azure"]
REGIONS = ["us-east-1", "eu-central-1", "ap-east-1", "sa-east-1", "me-south-1"]
COUNTRIES = ["United States", "Germany", "Singapore", "Brazil",
             "United Arab Emirates"]
INDUSTRIES = ["Technology", "Financial Services", "Retail", "Healthcare", "Media"]
CASE_PRIORITY = ["P1", "P2", "P3", "P4"]
CASE_STATUS = ["Open", "In Progress", "Closed"]
CASE_CATEGORY = ["Performance", "Billing", "Ingestion", "Access", "Upgrade"]

HEADERS = {
    "dbt_marts_general__usage_history": [
        "timestamp_hour", "organization__id", "organization__name",
        "organization__email_domain", "organization__tier",
        "organization__billing_model", "organization__cloud_provider",
        "organization__region", "organization__country",
        "organization__dollar_usage", "organization__MRR",
        "organization__MRR_PAYG", "organization__MRR_Committed",
        "organization__query_count", "organization__error_count"],
    "dbt_dds__dim_organization_current": [
        "organization__key", "organization__id", "organization__account_key",
        "organization__tier", "organization__billing_model",
        "organization__cloud_provider", "organization__region",
        "organization__country"],
    "dbt_dds__dim_account_current": [
        "account__key", "account__id", "account__name", "account__status",
        "account__industry", "account__region", "account__billing_country"],
    "dbt_dds__fct_case": [
        "case__case_number", "case__account_key", "case__subject",
        "case__category", "case__status", "case__is_closed", "case__priority",
        "case__origin", "case__owner_name", "case__created_date",
        "case__closed_date"],
}


def _rows(fake: Faker, rng: random.Random):
    names: list[str] = []
    while len(names) < N_ORGS:
        n = fake.company()
        if n not in names:
            names.append(n)

    usage, org_dim, accounts, cases = [], [], [], []
    for i, name in enumerate(names):
        # UUID-shaped ids, because the real ones are and the questions ask for them
        oid = str(fake.uuid4())
        akey = f"001{i:015d}"          # Salesforce-shaped 18-char key
        tier = TIERS[i % 3]
        billing = BILLING[i % 2]
        cloud, region = CLOUDS[i % 3], REGIONS[i % len(REGIONS)]
        country = COUNTRIES[i % len(COUNTRIES)]
        domain = name.lower().replace(" ", "").replace(",", "")[:14] + ".example"
        base_q = {"Enterprise": 900_000, "Growth": 90_000, "Free": 4_000}[tier]
        mrr = {"Enterprise": 41_000.0, "Growth": 5_200.0, "Free": 0.0}[tier]

        accounts.append((akey, str(fake.uuid4()), name,
                         rng.choice(["Customer", "Customer", "Prospect"]),
                         INDUSTRIES[i % len(INDUSTRIES)], region, country))
        org_dim.append((f"org-{i + 1:04d}", oid, akey, tier, billing, cloud,
                        region, country))

        for d in range(DAYS):
            day = SNAPSHOT - timedelta(days=DAYS - 1 - d)
            weekday = 0.6 if day.weekday() >= 5 else 1.0
            q = int(base_q * weekday * rng.uniform(0.75, 1.25))
            payg = round(mrr * (1.0 if billing == "PAYG" else 0.15), 2)
            usage.append((
                # DateTime at midnight, matching the real grain
                datetime(day.year, day.month, day.day).strftime("%Y-%m-%d %H:%M:%S"),
                oid, name, domain, tier, billing, cloud, region, country,
                round(mrr / 30 * rng.uniform(0.9, 1.1), 2),
                round(mrr, 2), payg, round(mrr - payg, 2),
                q, int(q * rng.uniform(0.0005, 0.004))))

        for j in range(rng.randint(0, 5)):
            created = SNAPSHOT - timedelta(days=rng.randint(1, DAYS - 1))
            closed = (created + timedelta(days=rng.randint(1, 20))
                      if rng.random() < 0.7 else None)
            cases.append((
                f"{5_100_000 + i * 100 + j}", akey,
                f"{rng.choice(CASE_CATEGORY)} question from {name}",
                rng.choice(CASE_CATEGORY),
                "Closed" if closed else rng.choice(["Open", "In Progress"]),
                1 if closed else 0,
                rng.choice(CASE_PRIORITY),
                rng.choice(["Web", "Email", "Phone"]),
                f"support-{rng.randint(1, 4)}",
                created.isoformat(),
                closed.isoformat() if closed else None))
    return names, usage, org_dim, accounts, cases


def _write(out_dir: Path, name: str, rows: list[tuple]) -> None:
    path = out_dir / f"{name}.csv"
    with path.open("w", newline="") as fh:
        # QUOTE_MINIMAL and a bare \N, which is what ClickHouse reads as NULL. A
        # quoted "\N" parses as the two-character string and fails against a
        # Nullable column partway through the load.
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(HEADERS[name])
        for r in rows:
            w.writerow(["\\N" if v is None else v for v in r])
    print(f"  {name + '.csv':44s} {len(rows):>5d} rows  "
          f"{path.stat().st_size / 1024:>6.1f} KB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=CSV_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fake = Faker("en_US")
    Faker.seed(SEED)
    rng = random.Random(SEED)
    names, usage, org_dim, accounts, cases = _rows(fake, rng)

    _write(args.out_dir, "dbt_marts_general__usage_history", usage)
    _write(args.out_dir, "dbt_dds__dim_organization_current", org_dim)
    _write(args.out_dir, "dbt_dds__dim_account_current", accounts)
    _write(args.out_dir, "dbt_dds__fct_case", cases)

    print(f"\nwrote {args.out_dir} (seed {SEED}, faker "
          f"{__import__('faker').VERSION}, snapshot {SNAPSHOT})")
    print("organizations: " + " | ".join(names))
    print("\nIf those names changed, questions.jsonl is stale: it names them.")


if __name__ == "__main__":
    main()
