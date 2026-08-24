"""Regenerate example/data/*.csv, the frozen fixture behind the example warehouse.

You almost certainly do not need to run this. The CSVs are committed, and
seed_example.py loads them; this script exists only to record how they were made
and to let the fixture be changed deliberately.

The data is committed rather than generated at seed time for two reasons. It is
reviewable: 70 KB of CSV shows exactly what ships, where a chDB store is 60 opaque
files that cannot be diffed. And it is stable: Faker's output changes between
releases, and example/questions.jsonl names tenants directly, so a Faker upgrade
would silently rewrite the warehouse under questions that still reference the old
names, breaking two of eight and leaving the rest passing.

So: RUNNING THIS CAN INVALIDATE example/questions.jsonl. Any change to the seed,
the tenant count or the Faker version reshuffles the names. Re-read the questions
afterwards and rebuild ground truth.

Faker rather than hand-invented names. Inventing was tried first and is a trap:
screening a list of plausible company names turned up real firms behind most of
them, several in the same sector, and even coined-looking tokens were registered
businesses. Faker guarantees nothing either, but nothing here is concealed, so
there is no assurance to falsify. Account owners are identifiers (`analyst-1`)
rather than person names, because no invented person name is nobody's real name.

    uv run example/generate_example_data.py
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

CSV_DIR = Path(__file__).resolve().parent / "data"
SEED = 20260821
SNAPSHOT = date(2026, 6, 30)
DAYS = 120
N_TENANTS = 8

PLANS = {"starter": 250.0, "growth": 1200.0, "enterprise": 5400.0}
REGIONS = ["eu-west-1", "us-east-1", "eu-central-1", "ap-south-1", "us-west-2"]
STAGES = ["discovery", "evaluation", "negotiation", "closed_won", "closed_lost"]

HEADERS = {
    "marts__usage_daily": ["day", "tenant_id", "tenant_name", "plan", "region",
                           "queries", "storage_gb", "compute_minutes", "daily_spend"],
    "crm__dim_account": ["account_key", "tenant_id", "account_name", "owner",
                         "signed_on"],
    "crm__fct_opportunity": ["opportunity_id", "account_key", "stage", "amount",
                             "opened_on", "closed_on"],
    "crm__fct_support_case": ["case_id", "account_key", "priority", "opened_on",
                              "closed_on"],
}


def _rows(fake: Faker, rng: random.Random):
    usage, accounts, opps, cases = [], [], [], []
    # Unique tenant names: Faker can repeat, and a duplicate would make
    # name-resolution questions ambiguous rather than merely synthetic.
    names: list[str] = []
    while len(names) < N_TENANTS:
        n = fake.company()
        if n not in names:
            names.append(n)

    for i, name in enumerate(names):
        tid, akey = f"t{i + 1:03d}", f"acct-{i + 1:03d}"
        plan = ["starter", "growth", "enterprise"][i % 3]
        region = REGIONS[i % len(REGIONS)]
        base_q = {"starter": 400, "growth": 3_000, "enterprise": 18_000}[plan]
        accounts.append((akey, tid, name, f"analyst-{i % 4 + 1}",
                         (SNAPSHOT - timedelta(days=rng.randint(200, 900))).isoformat()))
        for d in range(DAYS):
            day = SNAPSHOT - timedelta(days=DAYS - 1 - d)
            # a mild weekday cycle plus noise, so time-grain questions have signal
            weekday = 0.55 if day.weekday() >= 5 else 1.0
            q = int(base_q * weekday * rng.uniform(0.7, 1.3))
            usage.append((day.isoformat(), tid, name, plan, region, q,
                          round(base_q / 90 * rng.uniform(0.9, 1.1), 2),
                          round(q / 45 * rng.uniform(0.8, 1.2), 2),
                          round(PLANS[plan] / 30 * rng.uniform(0.95, 1.05), 2)))
        for j in range(rng.randint(1, 4)):
            opened = SNAPSHOT - timedelta(days=rng.randint(10, 300))
            stage = rng.choice(STAGES)
            closed = (opened + timedelta(days=rng.randint(5, 60))
                      if stage.startswith("closed") else None)
            opps.append((f"opp-{i + 1:03d}-{j + 1}", akey, stage,
                         round(PLANS[plan] * rng.uniform(2, 12), 2),
                         opened.isoformat(), closed.isoformat() if closed else None))
        for j in range(rng.randint(0, 6)):
            opened = SNAPSHOT - timedelta(days=rng.randint(1, 180))
            closed = (opened + timedelta(days=rng.randint(1, 21))
                      if rng.random() < 0.75 else None)
            cases.append((f"case-{i + 1:03d}-{j + 1}", akey,
                          rng.choice(["low", "medium", "high", "urgent"]),
                          opened.isoformat(), closed.isoformat() if closed else None))
    return names, usage, accounts, opps, cases


def _write(out_dir: Path, name: str, rows: list[tuple]) -> None:
    path = out_dir / f"{name}.csv"
    with path.open("w", newline="") as fh:
        # QUOTE_MINIMAL, not QUOTE_ALL: ClickHouse reads a bare \N as NULL but a
        # quoted "\N" as the two-character string, which fails to parse against
        # Nullable(Date). Minimal quoting also keeps the file readable, which is
        # the reason it is committed at all.
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(HEADERS[name])
        for r in rows:
            w.writerow(["\\N" if v is None else v for v in r])
    print(f"  {name + '.csv':26s} {len(rows):>5d} rows  "
          f"{path.stat().st_size / 1024:>5.1f} KB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=CSV_DIR)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fake = Faker("en_US")
    Faker.seed(SEED)
    rng = random.Random(SEED)
    names, usage, accounts, opps, cases = _rows(fake, rng)

    _write(out_dir, "marts__usage_daily", usage)
    _write(out_dir, "crm__dim_account", accounts)
    _write(out_dir, "crm__fct_opportunity", opps)
    _write(out_dir, "crm__fct_support_case", cases)

    print(f"\nwrote {out_dir} (seed {SEED}, faker {__import__('faker').VERSION}, "
          f"snapshot {SNAPSHOT})")
    print("tenants: " + " | ".join(names))
    print("\nIf the tenant names above changed, example/questions.jsonl is now "
          "stale: it names tenants directly.")


if __name__ == "__main__":
    main()
