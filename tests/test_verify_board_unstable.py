"""Guards for verify_board.py's handling of results that change while the data does not.

A catalog query (system.*, information_schema, SHOW) returns the list of tables,
so its cached result stops reproducing as soon as tables are added around the
board. LIMIT without ORDER BY may return any of the matching rows. With
--skip-unstable-queries those queries leave the comparison; every other query
must still be compared. Float sums change in their last digits between runs, so
floats compare after rounding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAM))

vb = pytest.importorskip("verify_board")

CATALOG = [
    "SELECT name FROM system.tables WHERE database = 'marts'",
    "select table_name from INFORMATION_SCHEMA.columns",
    "SELECT * FROM information_schema . tables",
    "SHOW TABLES FROM marts",
    "  show columns from marts.orders",
    "EXISTS TABLE marts.orders",
]
DATA = [
    "SELECT * FROM marts.orders",
    "SELECT system_name FROM marts.orders",
    "DESCRIBE marts.orders",
    "WITH x AS (SELECT 1) SELECT * FROM x",
]


UNORDERED_LIMIT = [
    "SELECT id FROM marts.orders WHERE name ILIKE '%acme%' LIMIT 1",
    "select * from marts.orders limit 10",
]


@pytest.mark.parametrize("q", CATALOG + UNORDERED_LIMIT)
def test_unstable_queries_are_recognised(q):
    assert vb.unstable(q)


@pytest.mark.parametrize("q", DATA + ["SELECT id FROM marts.orders ORDER BY id LIMIT 5"])
def test_stable_queries_are_not(q):
    assert not vb.unstable(q)


def test_skip_unstable_drops_only_unstable_queries():
    sqls = [CATALOG[0], UNORDERED_LIMIT[0], DATA[0]]
    stored = ['{"name":"orders"}', '{"id":2}', '{"id":1}']
    assert [q for q, _ in vb._comparable(sqls, stored)] == sqls
    assert [q for q, _ in vb._comparable(sqls, stored, skip_unstable=True)] == [DATA[0]]


def test_float_sums_compare_after_rounding():
    # The same parallel sum() on the same data, two runs.
    a = vb._rowset('{"month":"2025-06-01","mrr":246821089.35630006}')
    b = vb._rowset('{"month":"2025-06-01","mrr":246821089.35630056}')
    assert a == b
    assert vb._rowset('{"mrr":246821089.4}') != a
