"""Guards for the per-run token accumulator.

The accumulator is the only record of what a run cost, and it is written once at
call time: usage cannot be recovered from a transcript afterwards, because hidden
thinking is billed as output and never appears there. So a defect here is not a
wrong number on a dashboard, it is a paid run that has to be repeated.

These tests pin four properties that are easy to break and expensive to notice.

THE VECTOR IS NOT DERIVABLE FROM THE SUMS. `per_turn` answers questions the totals
cannot: how close a turn came to the per-call output cap, which is what truncates a
run and therefore fails it, and how input grows across the transcript, which is
what makes cache behaviour legible. A mean over turns can sit far below a peak that
nearly hit the ceiling, so per-turn data has to be captured as it happens or not at
all.

REASONING NESTS INSIDE COMPLETION. Every provider that reports reasoning tokens
counts them within its completion figure, so anything that adds the two together
double-counts exactly the models it is meant to illuminate. The emitted total
excludes reasoning for this reason, and the vector keeps it as a separate element
rather than folding it in.

A MISSED CALL IS NOT A ZERO CALL. A provider that omits usage, or a gateway that
drops the details object, must leave the totals and the vector untouched and
increment the miss counter, so a gap can never be read as a genuine zero. This is
why `len(per_turn)` can be below `api_calls`.

ORDER IS MEANINGFUL. The vector is in call order, because growth across turns is
one of the things it exists to show.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DAM_DATA_ROOT", "/tmp/dam-token-usage-test")
os.environ.setdefault("DAM_MODELS_CONFIG",
                      str(Path(__file__).resolve().parents[1] / "config" / "models.example.yaml"))
os.environ.setdefault("OPENAI_API_KEY", "placeholder")
os.environ.setdefault("ANTHROPIC_API_KEY", "placeholder")
os.environ.setdefault("FIREWORKS_API_KEY", "placeholder")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bench = pytest.importorskip("bench")


def _usage(**kw):
    """Minimal stand-in for a provider usage object, addressed by attribute."""
    return type("U", (), kw)()


def test_per_turn_preserves_each_turn_and_its_order():
    u = bench.TokenUsage()
    for prompt, completion in ((1000, 100), (2000, 1900), (3000, 300)):
        u._record(_usage(x=1), prompt=prompt, completion=completion)
    out = u.as_dict()
    assert [t[0] for t in out["per_turn"]] == [1000, 2000, 3000], "input growth, in call order"
    assert [t[1] for t in out["per_turn"]] == [100, 1900, 300]
    assert out["completion_tokens"] == 2300
    assert out["api_calls"] == 3


def test_the_peak_is_recoverable_and_the_mean_would_have_hidden_it():
    u = bench.TokenUsage()
    for completion in (100, 1900, 300):
        u._record(_usage(x=1), prompt=1000, completion=completion)
    out = u.as_dict()
    peak = max(t[1] for t in out["per_turn"])
    mean = out["completion_tokens"] / out["api_calls"]
    assert peak == 1900
    assert mean < peak, "the mean is why the vector is needed rather than the sums"


def test_reasoning_is_kept_separate_not_folded_into_completion():
    u = bench.TokenUsage()
    u._record(_usage(x=1), prompt=1000, completion=1500, reasoning=1200)
    out = u.as_dict()
    prompt, completion, reasoning, c_read, c_write = out["per_turn"][0]
    assert (completion, reasoning) == (1500, 1200), "not summed into 2700"
    assert out["total_tokens"] == out["prompt_tokens"] + out["completion_tokens"] \
        + out["cache_read_tokens"] + out["cache_write_tokens"]


def test_a_call_with_no_usage_appends_nothing_and_counts_as_missing():
    u = bench.TokenUsage()
    u._record(_usage(x=1), prompt=1000, completion=900)
    u._record(None, prompt=0, completion=0)
    u._record(_usage(x=1), prompt=0, completion=0)
    out = u.as_dict()
    assert len(out["per_turn"]) == 1
    assert out["calls_missing_usage"] == 2
    assert out["api_calls"] == 3, "a missed call still happened and still cost money"
    assert len(out["per_turn"]) < out["api_calls"]


def test_fresh_accumulator_has_an_empty_vector():
    out = bench.TokenUsage().as_dict()
    assert out["per_turn"] == []
    assert out["api_calls"] == 0


def test_cache_fields_are_per_turn_too():
    u = bench.TokenUsage()
    u._record(_usage(x=1), prompt=500, completion=50, c_read=0, c_write=4000)
    u._record(_usage(x=1), prompt=20, completion=60, c_read=4400, c_write=0)
    out = u.as_dict()
    assert [t[3] for t in out["per_turn"]] == [0, 4400], "cache reads start on turn 2"
    assert [t[4] for t in out["per_turn"]] == [4000, 0], "the write happens on turn 1"
