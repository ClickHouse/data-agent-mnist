"""Guards for the decidable schema-prompt rule engine, on the shipped example rules.

These facts feed Jev's state and stand in as a free oracle for validating Jev, so a
check that silently flips (a violation read as compliant, or the reverse) corrupts
both the labels and the validation. The engine is schema-independent: the rules are
data, so this exercises it against config/rules.example.yaml (the saas example). Pure
logic: regex over query text, no SDK, no warehouse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parents[1]  # experiments/data-agent-mnist
sys.path.insert(0, str(DAM))

import rule_checks as rc  # noqa: E402

RULES = rc.load_rules(DAM / "config/rules.example.yaml")


def _v(sqls):
    return rc.decidable_verdicts(rc.check_sql_rules(sqls, RULES))


def test_reference_date_now_is_violation_todate_ok_absent_na():
    assert _v(["SELECT x FROM marts.usage_daily WHERE day >= now() - INTERVAL 7 DAY"]
              )["reference_date"] == rc.VIOLATION
    assert _v(["SELECT x FROM marts.usage_daily WHERE day >= toDate('2026-06-30')"]
              )["reference_date"] == rc.OK
    assert _v(["SELECT count() FROM crm.dim_account"])["reference_date"] == rc.NA


def test_usage_mart_needs_a_day_or_tenant_filter():
    bad = _v(["SELECT sum(daily_spend) FROM marts.usage_daily"])
    assert bad["default_to_usage_mart"] == rc.VIOLATION
    good = _v(["SELECT sum(daily_spend) FROM marts.usage_daily WHERE day = toDate('2026-06-30')"])
    assert good["default_to_usage_mart"] == rc.OK
    absent = _v(["SELECT account_name FROM crm.dim_account"])
    assert absent["default_to_usage_mart"] == rc.NA


def test_per_statement_scope_catches_one_bad_statement():
    """A trigger statement that lacks the required token is a violation even when a
    different statement carries it: the scope is per_statement, not global."""
    mixed = _v([
        "SELECT max(day) FROM marts.usage_daily WHERE tenant_id = 't1'",   # ok
        "SELECT sum(daily_spend) FROM marts.usage_daily",                   # no filter
    ])
    assert mixed["default_to_usage_mart"] == rc.VIOLATION


def test_output_column_naming_rename_is_a_violation():
    assert _v(["SELECT tenant_name AS t FROM marts.usage_daily WHERE day = toDate('2026-06-30')"]
              )["output_column_naming"] == rc.VIOLATION
    assert _v(["SELECT tenant_name FROM marts.usage_daily WHERE day = toDate('2026-06-30')"]
              )["output_column_naming"] == rc.NA


def test_generic_counts_are_schema_independent():
    f = rc.check_sql_rules(["SELECT 1", "select   1", "SELECT 2"], RULES)
    assert f["n_queries"] == 3 and f["has_repeated_query"] is True


def test_decidable_verdicts_cover_exactly_the_declared_keys():
    """The labeler splits rules on the decidable keys, so decidable_verdicts must return
    exactly those or a rule falls through both scoring paths."""
    f = rc.check_sql_rules(["SELECT 1"], RULES)
    assert set(rc.decidable_verdicts(f)) == set(rc.decidable_rule_keys(RULES))


def test_a_rule_in_both_groups_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("semantic_rules:\n  foo: a semantic rule\n"
                 "decidable_rules:\n  foo:\n    kind: forbidden_when\n    forbidden_regex: 'x'\n")
    with pytest.raises(ValueError):
        rc.load_rules(p)
