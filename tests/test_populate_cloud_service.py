"""Guards for maintenance/populate_cloud_service.py.

The live path talks to a ClickHouse Cloud service, which CI has none of, so these
cover the parts that decide correctness without a network: the DDL rewrite, the
source-table filter, and that the service password only ever comes from the
environment (never the command line).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAM / "maintenance"))
sys.path.insert(0, str(DAM))

import populate_cloud_service as pcs  # noqa: E402


def test_to_cloud_ddl_adds_if_not_exists():
    out = pcs.to_cloud_ddl("CREATE TABLE db.t (a Int64) ENGINE = MergeTree ORDER BY a",
                           "db", "t")
    assert out.startswith("CREATE TABLE IF NOT EXISTS `db`.`t` (")


def test_to_cloud_ddl_strips_server_uuid():
    src = ("CREATE TABLE dbt_marts_general.usage_history UUID 'a1b2-c3d4' "
           "(a Int64) ENGINE = MergeTree ORDER BY a")
    out = pcs.to_cloud_ddl(src, "dbt_marts_general", "usage_history")
    assert "UUID" not in out
    assert out.startswith("CREATE TABLE IF NOT EXISTS `dbt_marts_general`.`usage_history` (")


def test_to_cloud_ddl_retargets_database():
    src = "CREATE TABLE dbt_marts_general.usage_history (a Int64) ENGINE = MergeTree ORDER BY a"
    out = pcs.to_cloud_ddl(src, "dwh_demo", "usage_history")
    assert out.startswith("CREATE TABLE IF NOT EXISTS `dwh_demo`.`usage_history` (")
    assert "dbt_marts_general" not in out


def test_export_query_plain_by_default():
    assert pcs.export_query("dbt_dds", "dim_account_current", enrich=False) == \
        "SELECT * FROM `dbt_dds`.`dim_account_current`"


def test_export_query_enriches_only_configured_tables():
    # An enrichable table gets a REPLACE for each configured column.
    q = pcs.export_query("dbt_dds", "dim_account_current", enrich=True)
    assert q.startswith("SELECT * REPLACE (")
    assert q.endswith(" FROM `dbt_dds`.`dim_account_current`")
    for col in ("account__competitor_migration", "account__competitor_expertise",
                "account__use_case_details"):
        assert f"AS `{col}`" in q
    # A table with no enrichment spec stays a plain copy even with enrich on.
    assert pcs.export_query("dbt_marts_general", "usage_history", enrich=True) == \
        "SELECT * FROM `dbt_marts_general`.`usage_history`"


def test_enrichment_covers_the_expected_dds_tables_and_columns():
    # Every enriched target is a dbt_dds dimension, and each column's expression names
    # a per-row array pick, so no column is filled with a single constant value.
    for table, cols in pcs.DEMO_ENRICHMENT.items():
        assert table.startswith("dbt_dds.")
        for expr in cols.values():
            assert "cityHash64(" in expr and "modulo(" in expr
    assert "lead__use_case" in pcs.DEMO_ENRICHMENT["dbt_dds.dim_lead_current"]
    assert "opportunity__primary_use_case" in pcs.DEMO_ENRICHMENT["dbt_dds.dim_opportunity_current"]


class _Res:
    def __init__(self, payload: str):
        self._b = payload.encode()

    def bytes(self) -> bytes:
        return self._b


class _Sess:
    """Minimal chDB-session stand-in: records the SQL and replays a canned result."""

    def __init__(self, payload: str):
        self.payload = payload
        self.sql: str | None = None

    def query(self, sql: str, fmt: str) -> _Res:
        self.sql = sql
        return _Res(self.payload)


def test_iter_source_tables_parses_rows_and_filters_system_and_views():
    payload = (
        '{"database":"dbt_marts_general","name":"usage_history","engine":"MergeTree",'
        '"create_table_query":"CREATE TABLE dbt_marts_general.usage_history (a Int64) '
        'ENGINE = MergeTree ORDER BY a"}\n'
        '{"database":"raw_google_analytics","name":"events","engine":"MergeTree",'
        '"create_table_query":"CREATE TABLE raw_google_analytics.events (b Int64) '
        'ENGINE = MergeTree ORDER BY b"}\n'
    )
    sess = _Sess(payload)
    rows = pcs.iter_source_tables(sess)
    assert [r["name"] for r in rows] == ["usage_history", "events"]
    # The filter is in the SQL, so assert the SQL carries it.
    assert "NOT LIKE '%View%'" in sess.sql
    assert "'system'" in sess.sql and "'default'" in sess.sql


def test_cloud_client_kwargs_reads_password_from_env():
    args = pcs.build_parser().parse_args(["--host", "svc.clickhouse.cloud"])
    kwargs = pcs.cloud_client_kwargs(args, {pcs.PASSWORD_ENV: "s3cret"})
    assert kwargs["host"] == "svc.clickhouse.cloud"
    assert kwargs["port"] == 8443
    assert kwargs["username"] == "default"
    assert kwargs["password"] == "s3cret"
    assert kwargs["secure"] is True


def test_cloud_client_kwargs_requires_the_env_var():
    args = pcs.build_parser().parse_args(["--host", "svc.clickhouse.cloud"])
    with pytest.raises(SystemExit):
        pcs.cloud_client_kwargs(args, {})


def test_password_is_not_a_command_line_flag():
    parser = pcs.build_parser()
    options = [opt for action in parser._actions for opt in action.option_strings]
    assert not any("password" in opt.lower() for opt in options)
    with pytest.raises(SystemExit):
        parser.parse_args(["--host", "h", "--password", "leaked"])
