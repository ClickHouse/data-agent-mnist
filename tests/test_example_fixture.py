"""Guards for the example fixture (AI-1862).

The example is the only thing an adopter runs before deciding whether the harness
works, so it fails in the most expensive way: silently and on someone else's
machine. These tests need no API keys and no benchmark data.

Two failures worth naming, because both were live during development.

A malformed CSV half-seeds. `INSERT ... SELECT * FROM file(...)` is one query per
table, so a parse error on table three leaves tables one and two populated and the
rest empty. The seeder reports the row counts it loaded and exits non-zero, but a
warehouse that answers four of eight questions looks like a model problem, not a
fixture problem. That is what happened with quoted `\\N`, which ClickHouse reads as
a two-character string rather than NULL.

Tenant names drift. Two questions name a tenant, so regenerating the CSVs with a
different Faker version rewrites the warehouse under them: those two go
unanswerable and the other six keep passing, which reads as a model regression.
test_questions_reference_tenants_that_exist ties the questions to the data.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parent.parent
EXAMPLE = DAM / "example"
CSV_DIR = EXAMPLE / "data"

TABLES = {
    "marts.usage_daily": 960,
    "crm.dim_account": 8,
    "crm.fct_opportunity": 15,
    "crm.fct_support_case": 28,
}


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    """Seed a throwaway warehouse from the committed CSVs.

    Out of tree on purpose: chDB rewrites its store directory on open, so seeding
    example/warehouse/ here would leave a developer's own copy modified.
    """
    db = tmp_path_factory.mktemp("wh") / "warehouse"
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE / "seed_example.py"), "--db-path", str(db)],
        capture_output=True, text=True, cwd=DAM)
    assert proc.returncode == 0, f"seed failed:\n{proc.stdout}\n{proc.stderr}"
    return db


def test_csvs_are_committed():
    """The fixture ships as data, not as a generator invocation."""
    missing = [t for t in TABLES
               if not (CSV_DIR / f"{t.replace('.', '__')}.csv").exists()]
    assert not missing, f"missing committed CSVs for {missing}"


def test_csvs_are_text_and_readable():
    """Being readable is the whole reason these are committed rather than the store."""
    for t in TABLES:
        raw = (CSV_DIR / f"{t.replace('.', '__')}.csv").read_bytes()
        assert b"\x00" not in raw, f"{t}: not text"
        raw.decode("utf-8")


def test_nulls_are_unquoted():
    """Quoted "\\N" parses as a string and fails against Nullable(Date) mid-seed."""
    for t in ("crm.fct_opportunity", "crm.fct_support_case"):
        text = (CSV_DIR / f"{t.replace('.', '__')}.csv").read_text()
        assert '"\\N"' not in text, f"{t}: quoted null would half-seed the warehouse"


def test_seed_loads_every_table_completely(seeded):
    """Every table, not just the first: a parse error mid-run is the failure mode."""
    from chdb import session as chdb_session
    sess = chdb_session.Session(str(seeded))
    try:
        for table, expected in TABLES.items():
            got = int(sess.query(f"SELECT count() FROM {table}",
                                 "CSV").bytes().decode().strip())
            assert got == expected, f"{table}: {got} rows, expected {expected}"
    finally:
        sess.close()


def test_nulls_round_trip_as_nulls(seeded):
    """Open opportunities and cases are the only thing NULL encodes here."""
    from chdb import session as chdb_session
    sess = chdb_session.Session(str(seeded))
    try:
        for table in ("crm.fct_opportunity", "crm.fct_support_case"):
            n = int(sess.query(f"SELECT count() FROM {table} WHERE closed_on IS NULL",
                               "CSV").bytes().decode().strip())
            assert n > 0, (f"{table}: no NULL closed_on, so either the fixture lost "
                           f"its open rows or \\N was read as a string")
    finally:
        sess.close()


def test_questions_reference_tenants_that_exist(seeded):
    """The questions name tenants; a regenerated fixture renames them.

    Asserted against the seeded warehouse rather than the CSV text so it also
    covers a name mangled on the way in (the comma in "Moon, Ford and Hanson" is
    exactly the sort of thing a quoting change breaks).
    """
    from chdb import session as chdb_session
    sess = chdb_session.Session(str(seeded))
    try:
        rows = sess.query("SELECT DISTINCT tenant_name FROM marts.usage_daily",
                          "JSONCompactEachRow").bytes().decode()
        tenants = {json.loads(line)[0] for line in rows.splitlines() if line.strip()}
    finally:
        sess.close()

    questions = [json.loads(line)["nl_question"]
                 for line in (EXAMPLE / "questions.jsonl").read_text().splitlines()
                 if line.strip()]
    # Any tenant a question names has to exist, or that question is unanswerable
    # and scores as a model failure.
    named = [(q, t) for q in questions for t in tenants if t in q]
    assert named, ("no question names any tenant in the warehouse. Either the "
                   "fixture was regenerated with a different Faker version and the "
                   "names drifted, or the name-resolution questions were dropped.")


def test_generator_reproduces_the_committed_csvs(tmp_path):
    """The generator is documentation only if it still emits what is committed."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE / "generate_example_data.py"),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=DAM)
    assert proc.returncode == 0, f"generator failed:\n{proc.stdout}\n{proc.stderr}"
    for t in TABLES:
        name = f"{t.replace('.', '__')}.csv"
        assert (tmp_path / name).read_text() == (CSV_DIR / name).read_text(), (
            f"{name} differs from the committed fixture. If this change is "
            f"intended, re-read example/questions.jsonl: the tenant names may "
            f"have moved.")
