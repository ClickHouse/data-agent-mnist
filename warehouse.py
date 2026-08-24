"""
Synthetic ClickHouse warehouse + schema context for the data-agent-mnist bench.

DB/schema concerns live here, separate from the model/judge infrastructure in
bench.py. The chDB warehouse is seeded by notebooks/03_seed_synthetic_db.ipynb;
this module opens it read-only and exposes the `run_select_query` tool body
(`Warehouse.query`) and the agent system prompt the agents see (prompts/system.md).

Extracted from notebooks/05_synthetic_benchmark.ipynb (cells 05 + 07) so the
annotate/eval stages can run as scripts (`uv run`) instead of notebook cells.
"""
import json
import re
import threading
from pathlib import Path

from chdb import session as chdb_session
from paths import DATA
# BOARD DEFAULTS. Every one of these names our benchmark, and the harness does not
# ship any of them: they resolve under DAM_DATA_ROOT, which an adopter points at
# their own datasets. They stay as defaults so our own runs need no flags, and
# every path built from them fails with the flag to pass instead of a bare
# missing-file error (AI-1858).
DATA_DIR       = DATA / "text2sqlbench-synthetic"
DB_PATH        = DATA_DIR / "chdb"
QUERY_TIMEOUT_S = 30          # per-query wall-clock ceiling; every
                              # worker blocks on the shared lock while
                              # one query runs, so this bounds the
                              # damage as well as the query
QUERY_MEMORY_LIMIT = 4 << 30  # 4 GiB; the whole warehouse is ~180 MB
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"


class Warehouse:
    """Read-only chDB session over the seeded synthetic DWH.

    `query` is the body of the `run_select_query` tool handed to every agent;
    pass `wh.query` wherever bench.run_candidate expects `ch_query`.

    The chDB session is a single embedded ClickHouse instance and is NOT safe for
    concurrent queries, so `query` serializes on a lock. The annotate/eval loops
    parallelize the (I/O-bound) model calls; the DB queries they interleave are
    fast local calls, so serializing them behind the lock costs little.
    """

    # The board's fact table, used to check the DB is populated and to derive the
    # snapshot date. Both are corpus-specific, so both are parameters: another
    # warehouse has other tables, and hardcoding these made the class unusable
    # against anything but ours (AI-1862).
    PROBE_TABLE = "dbt_marts_general.usage_history"
    SNAPSHOT_COLUMN = "timestamp_hour"

    def __init__(self, db_path: Path = DB_PATH,
                 system_prompt_path: Path = SYSTEM_PROMPT_PATH,
                 probe_table: str | None = None,
                 snapshot_column: str | None = None):
        """The schema prompt is a parameter because it describes THIS warehouse.

        It was a module constant, which is fine while there is one warehouse and
        wrong the moment there are two: the runnable example ships its own small
        warehouse with a different schema, and handing a model our table names for
        someone else's tables produces confident queries against columns that do
        not exist (AI-1862).
        """
        if not db_path.exists():
            raise RuntimeError(
                f"warehouse not found at {db_path}.\n"
                f"  This default is the board's, which ships with no data. Pass "
                f"db_path= (or --db-path) for your own warehouse, or run "
                f"example/seed_example.py to build the small one that comes with "
                f"the harness.")
        self._lock = threading.Lock()
        self.sess = chdb_session.Session(str(db_path))

        # Bound query execution. Every query runs under a single process-wide lock
        # (chDB allows one session path per process), so an unbounded query does not
        # just stall its own turn, it blocks EVERY concurrent worker behind the
        # lock. Observed on the 60-turn board sweep: one runaway query pegged a core
        # and froze all 24 workers for 20+ minutes with zero completions. The risk
        # grows with the turn budget, because a model given more turns explores more
        # and writes heavier SQL.
        #
        # 60s is far above any legitimate query here (the ground-truth set runs in
        # milliseconds), so this only catches pathological ones. A timeout surfaces
        # to the model as an ordinary tool error, which it can react to, rather than
        # hanging the run.
        # A memory limit is not redundant with the time limit: an unbounded join
        # blows memory long before it blows the clock, and chDB ABORTS THE PROCESS
        # on allocation failure rather than raising, which would take down the whole
        # sweep. With a limit it raises, and query() turns it into a tool error.
        self.sess.query(f"SET max_execution_time = {QUERY_TIMEOUT_S}")
        self.sess.query(f"SET max_memory_usage = {QUERY_MEMORY_LIMIT}")

        probe = probe_table or self.PROBE_TABLE
        try:
            n = int(self.sess.query(f"SELECT count() FROM {probe}", "CSV")
                    .bytes().decode().strip())
        except Exception as e:
            raise RuntimeError(
                f"cannot read {probe} in {db_path}: {e}\n"
                f"  If this is not the board warehouse, pass probe_table= and "
                f"snapshot_column= for its own fact table.") from e
        if n == 0:
            raise RuntimeError(f"{db_path} is empty ({probe} has no rows) — seed it first.")
        self.n_rows = n

        # Fixed snapshot date (max timestamp in the DB). Both annotation and eval
        # use it so time-relative queries ("last 90 days") resolve to identical
        # windows regardless of when the bench is run.
        _col = snapshot_column or self.SNAPSHOT_COLUMN
        _raw = self.query(f"SELECT max({_col}) AS d FROM {probe}")
        # query() reports failures as an "Error: ..." STRING rather than raising,
        # because that is what a model on the other end of the tool needs. Here
        # there is no model, so an unparsed error surfaced as JSONDecodeError
        # ("Expecting value: line 1 column 1") from json.loads, several frames from
        # the actual problem and naming neither the column nor the table. The probe
        # check above says what to pass; this has to as well (AI-1862).
        try:
            self.snapshot_date = json.loads(_raw.splitlines()[0])["d"][:10]  # YYYY-MM-DD
        except Exception as e:
            raise RuntimeError(
                f"cannot read a snapshot date from {probe}.{_col}: "
                f"{_raw.splitlines()[0][:200] if _raw else '(no result)'}\n"
                f"  snapshot_column must name a date or datetime column on "
                f"{probe}. It defaults to {self.SNAPSHOT_COLUMN!r}, which is the "
                f"board warehouse's.") from e

        # Same shape as the probe check: the default is the board's schema prompt,
        # which the harness does not ship, so say which flag supplies one.
        try:
            self._system_template = Path(system_prompt_path).read_text()
        except FileNotFoundError as e:
            raise RuntimeError(
                f"no schema prompt at {system_prompt_path}.\n"
                f"  It describes YOUR warehouse to the model, so there is no useful "
                f"default. Pass system_prompt_path= (or --system-prompt); "
                f"example/schema.md is a worked one.") from e

    def close(self):
        try:
            self.sess.close()
        except Exception:
            pass

    def query(self, query: str, max_chars: int = 30_000) -> str:
        # Strip leading SQL comments (-- ...) before checking query type.
        stripped   = re.sub(r"(^\s*--[^\n]*\n)+", "", query.lstrip(), flags=re.MULTILINE).lstrip()
        normalized = stripped.upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            return "Error: only SELECT queries are allowed."
        try:
            with self._lock:
                result = self.sess.query(query, "JSONEachRow")
                text   = result.bytes().decode().strip()
            if len(text) > max_chars:
                return text[:max_chars] + "\n... (truncated)"
            return text or "(empty result)"
        except Exception as e:
            return f"Error: {e}"

    def system_prompt(self) -> str:
        # .replace (not .format) — the template is SQL-heavy and may carry literal braces.
        return self._system_template.replace("{snapshot_date}", self.snapshot_date)
