"""Guard for the ceiling summary's token categories.

The per-model `tokens` block and the board `tokens_total_mtok` block once carried
two hand-maintained key tuples. cache_write_tokens was dropped from the totals
tuple alone, so the board categories could not add back up to their own
total_tokens (total_tokens sums the write in; the breakdown hid it).

Both blocks now iterate one shared `_TOKEN_KEYS`, so they cannot drift. This pins
that the tuple carries the write and that the summable categories account for
total_tokens exactly the way TokenUsage.as_dict defines it: prompt + completion +
cache_read + cache_write, with reasoning nested inside completion, not added.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("DAM_DATA_ROOT", "/tmp/dam-ceiling-summary-test")
# 18_ceiling_summary.py does `from paths import DATA`. Running it as a script puts
# its directory on sys.path automatically; importing it here does not, so put the
# experiment root on the path first, the same way test_token_usage.py does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "18_ceiling_summary.py"
_spec = importlib.util.spec_from_file_location("ceiling_summary", _MODULE_PATH)
ceiling_summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ceiling_summary)


def test_token_keys_carry_both_cache_terms():
    keys = ceiling_summary._TOKEN_KEYS
    assert "cache_write_tokens" in keys, "the write category is what went missing"
    assert "cache_read_tokens" in keys


def test_token_categories_reconcile_with_total():
    # total_tokens = prompt + completion + cache_read + cache_write (as_dict).
    # reasoning is reported but nests inside completion, so it is not a component.
    keys = set(ceiling_summary._TOKEN_KEYS)
    components = keys - {"total_tokens", "reasoning_tokens"}
    assert components == {"prompt_tokens", "completion_tokens",
                          "cache_read_tokens", "cache_write_tokens"}
    assert {"total_tokens", "reasoning_tokens"} <= keys
