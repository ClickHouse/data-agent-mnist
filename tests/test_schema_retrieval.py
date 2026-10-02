"""Guards for the schema-retrieval ranker's pure parts: the configurable table-name
matchers and the ranking metrics. No SDK, no network — the Jev call is not exercised
here. The namespaces are the saas example's, so this names no board table.
"""
from __future__ import annotations

import sys
from pathlib import Path

DAM = Path(__file__).resolve().parents[1]  # experiments/data-agent-mnist
sys.path.insert(0, str(DAM))

import schema_retrieval as sr  # noqa: E402

TABLE_RE, FROMJOIN_RE = sr.table_regexes(["marts", "crm"])


def test_gold_tables_parses_from_and_join():
    sql = ("SELECT u.daily_spend FROM marts.usage_daily u "
           "JOIN crm.dim_account a ON a.tenant_id = u.tenant_id")
    assert sr.gold_tables(sql, FROMJOIN_RE) == {"marts.usage_daily", "crm.dim_account"}


def test_gold_tables_accepts_a_statement_list_and_ignores_other_namespaces():
    assert sr.gold_tables(["SELECT * FROM crm.fct_opportunity", "SELECT * FROM other.foo"],
                          FROMJOIN_RE) == {"crm.fct_opportunity"}


def test_load_universe_reads_declared_tables_from_the_schema_text():
    schema = ("Table `marts.usage_daily` stores ...\n"
              "Table `crm.dim_account` ...\n"
              "join crm.fct_support_case on ...")
    assert set(sr.load_universe(schema, TABLE_RE)) == {
        "marts.usage_daily", "crm.dim_account", "crm.fct_support_case"}


def test_recall_and_ndcg():
    ranked = ["marts.usage_daily", "crm.dim_account", "crm.fct_support_case"]
    gold = {"marts.usage_daily", "crm.fct_support_case"}
    assert sr.recall_at(ranked, gold, 1) == 0.5
    assert sr.recall_at(ranked, gold, 3) == 1.0
    assert sr.ndcg_at(ranked, gold, 3) > 0.0
    assert sr.ndcg_at(["x", "y"], {"z"}, 2) == 0.0
