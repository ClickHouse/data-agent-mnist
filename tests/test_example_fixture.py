"""Guards for the example fixture.

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


class Instance:
    """One example warehouse: where it is, how to seed it, what it should hold.

    Parametrised over both instances rather than written for one. The saas
    fixture had these tests and clickhouse-dwh had none, so a fixture that
    shipped with the second instance was covered by nothing while CI seeded it
    and reported success.
    """

    def __init__(self, name, seeder, tables, nullable, question_key,
                 generator, timezone=None):
        self.name = name
        self.dir = DAM / "examples" / name
        self.csv_dir = self.dir / "data"
        self.seeder = seeder
        self.tables = tables
        self.nullable = nullable          # (table, nullable date column) pairs
        self.question_key = question_key  # column the questions name entities by
        self.generator = generator
        self.timezone = timezone          # needed when the grain is DateTime

    def __repr__(self):
        return self.name


INSTANCES = [
    Instance("saas", "seed_example.py",
             {"marts.usage_daily": 960, "crm.dim_account": 8,
              "crm.fct_opportunity": 15, "crm.fct_support_case": 28},
             (("crm.fct_opportunity", "closed_on"),
              ("crm.fct_support_case", "closed_on")),
             ("marts.usage_daily", "tenant_name"),
             "generate_example_data.py"),
    Instance("clickhouse-dwh", "seed_dwh.py",
             {"dbt_marts_general.usage_history": 900,
              "dbt_dds.dim_organization_current": 10,
              "dbt_dds.dim_account_current": 10,
              "dbt_dds.fct_case": 30},
             (("dbt_dds.fct_case", "case__closed_date"),),
             ("dbt_marts_general.usage_history", "organization__name"),
             "generate_dwh_data.py", timezone="UTC"),
]


@pytest.fixture(scope="module", params=INSTANCES, ids=repr)
def inst(request) -> Instance:
    return request.param


@pytest.fixture(scope="module")
def seeded(inst, tmp_path_factory) -> Path:
    """Seed a throwaway warehouse from the committed CSVs.

    Out of tree on purpose: chDB rewrites its store directory on open, so seeding
    the in-tree copy would leave a developer's own warehouse modified.
    """
    db = tmp_path_factory.mktemp("wh") / "warehouse"
    proc = subprocess.run(
        [sys.executable, str(inst.dir / inst.seeder), "--db-path", str(db)],
        capture_output=True, text=True, cwd=DAM)
    assert proc.returncode == 0, f"seed failed:\n{proc.stdout}\n{proc.stderr}"
    return db


def _query(db, inst, sql, fmt="CSV") -> str:
    """Run one query against a seeded warehouse, in a subprocess.

    A subprocess per query rather than a Session held open, because chDB allows
    ONE session path per process. Opening a second one, even after closing the
    first, fails on Linux with "recursive_mutex lock failed" while working fine
    on macOS. Parametrising these tests over two instances put two paths in one
    pytest process, so it passed locally and failed in CI.

    The seeding fixture already shells out for the same reason; this makes the
    reads agree with it.

    A DateTime grain is rendered in the session zone, so an instance that has one
    pins it here, or the test measures the runner's locale.
    """
    tz = (f's.query("SET session_timezone = \'{inst.timezone}\'")\n'
          if inst.timezone else "")
    code = ("import sys\n"
            "from chdb import session as cs\n"
            f"s = cs.Session({str(db)!r})\n"
            f"{tz}"
            f"sys.stdout.write(s.query({sql!r}, {fmt!r}).bytes().decode())\n"
            "s.close()\n")
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=DAM)
    assert proc.returncode == 0, f"query failed:\n{sql}\n{proc.stderr}"
    return proc.stdout


def test_csvs_are_committed(inst):
    """The fixture ships as data, not as a generator invocation."""
    missing = [t for t in inst.tables
               if not (inst.csv_dir / f"{t.replace('.', '__')}.csv").exists()]
    assert not missing, f"{inst}: missing committed CSVs for {missing}"


def test_csvs_are_text_and_readable(inst):
    """Being readable is the whole reason these are committed rather than the store."""
    for t in inst.tables:
        raw = (inst.csv_dir / f"{t.replace('.', '__')}.csv").read_bytes()
        assert b"\x00" not in raw, f"{inst}/{t}: not text"
        raw.decode("utf-8")


def test_nulls_are_unquoted(inst):
    """Quoted "\\N" parses as a string and fails against Nullable(Date) mid-seed."""
    for t, _col in inst.nullable:
        text = (inst.csv_dir / f"{t.replace('.', '__')}.csv").read_text()
        assert '"\\N"' not in text, (
            f"{inst}/{t}: quoted null would half-seed the warehouse")


def test_seed_loads_every_table_completely(seeded, inst):
    """Every table, not just the first: a parse error mid-run is the failure mode."""
    for table, expected in inst.tables.items():
        got = int(_query(seeded, inst, f"SELECT count() FROM {table}").strip())
        assert got == expected, f"{inst}/{table}: {got} rows, expected {expected}"


def test_nulls_round_trip_as_nulls(seeded, inst):
    """An open opportunity or case is the only thing NULL encodes here."""
    for table, col in inst.nullable:
        n = int(_query(seeded, inst,
                       f"SELECT count() FROM {table} WHERE {col} IS NULL").strip())
        assert n > 0, (f"{inst}/{table}: no NULL {col}, so either the fixture "
                       f"lost its open rows or \\N was read as a string")


def test_questions_reference_entities_that_exist(seeded, inst):
    """The questions name entities; a regenerated fixture renames them.

    Asserted against the seeded warehouse rather than the CSV text so it also
    covers a name mangled on the way in (the comma in "Moon, Ford and Hanson" is
    exactly the sort of thing a quoting change breaks).
    """
    table, column = inst.question_key
    rows = _query(seeded, inst, f"SELECT DISTINCT {column} FROM {table}",
                  "JSONCompactEachRow")
    names = {json.loads(line)[0] for line in rows.splitlines() if line.strip()}

    questions = [json.loads(line)["nl_question"]
                 for line in (inst.dir / "questions.jsonl").read_text().splitlines()
                 if line.strip()]
    # Any entity a question names has to exist, or that question is unanswerable
    # and scores as a model failure.
    named = [(q, n) for q in questions for n in names if n in q]
    assert named, (
        f"{inst}: no question names any {column} in the warehouse. Either the "
        f"fixture was regenerated with a different Faker version and the names "
        f"drifted, or the name-resolution questions were dropped.")


def test_generator_reproduces_the_committed_csvs(inst, tmp_path):
    """The generator is documentation only if it still emits what is committed."""
    proc = subprocess.run(
        [sys.executable, str(inst.dir / inst.generator), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=DAM)
    assert proc.returncode == 0, f"generator failed:\n{proc.stdout}\n{proc.stderr}"
    for t in inst.tables:
        name = f"{t.replace('.', '__')}.csv"
        assert (tmp_path / name).read_text() == (inst.csv_dir / name).read_text(), (
            f"{inst}/{name} differs from the committed fixture. If this change is "
            f"intended, re-read {inst.dir.name}/questions.jsonl: the entity names "
            f"may have moved.")
