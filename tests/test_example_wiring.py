"""Guards for the behaviours AI-1862 changed to make the harness portable.

Every one exists so a warehouse that is not ours can be evaluated, and every one
fails the same way: on an adopter's machine, with an error that points at the model
rather than at the wiring. None is exercised by our own board runs, because our own
runs use the defaults these changes introduced parameters for. That is the whole
reason to test them here.

  * `EFFORT_CAPABLE` gates the Anthropic thinking/effort parameters. Public Claude
    models reject `output_config.effort` with a 400, so sending it unconditionally
    made the native path usable only by the models that accept it, which are ours.
  * `judge_token_budget` decides which token kwarg a judge gets and floors the
    limit for reasoning models. The old test was an OpenAI name prefix, so a
    thinking judge on another provider spent its budget reasoning and returned
    nothing. On the two-seat example panel that takes the question down, because
    scoring needs two live votes.
  * `probe_table` / `snapshot_column` name the fact table used for the populated
    check and the snapshot date. Hardcoded, they made every warehouse have to be
    shaped like ours.
  * `system_prompt_path` is per-warehouse. Handing a model our table names for
    someone else's tables produces confident queries against columns that do not
    exist, which scores as a model failure.
  * Vertex credentials resolve on use, not at import. Resolved eagerly at module
    scope they made `import bench` need a GCP project, which the documented
    three-API-key path does not have.
  * the linker stats take their ground truth alongside `--results`. Left on the
    board default they crash when the board data is absent, and report a clean
    0.0 when it is present, because an empty trace_id intersection skips every
    pair rather than failing.

No credentials and no benchmark data. The Anthropic client is stubbed, so the
effort gate is checked on the request that would have been sent.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAM))

# bench.py constructs its provider clients at import, and the OpenAI client
# raises without a key, so importing it at all needs one. Placeholders: no test
# here reaches a provider (the Anthropic client is stubbed and the rest are
# unused), and a real key in the environment is left alone.
for _var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FIREWORKS_API_KEY"):
    os.environ.setdefault(_var, "test-placeholder-not-a-key")

import bench  # noqa: E402
import registry  # noqa: E402
import warehouse as warehouse_mod  # noqa: E402

EXAMPLE = DAM / "example"


# ── importability on the documented credentials ─────────────────────────────────

def test_bench_imports_on_the_three_documented_api_keys():
    """No cloud credentials to import the harness. Only to call a cloud provider.

    bench.py builds every provider client at module scope, so anything resolved
    eagerly becomes a hard import requirement for everyone. Vertex ADC was: it
    made `import bench` fail for an adopter following the example, who never calls
    Gemini. It passed locally regardless, because a developer machine has ADC, and
    only CI caught it. Hence a subprocess with the environment actually scrubbed
    rather than an in-process import.

    The three API keys are placeholders: importing must not validate them, and the
    example documents needing all three anyway.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AWS_", "GOOGLE_", "GCLOUD_", "CLOUDSDK_"))}
    env.update({"OPENAI_API_KEY": "placeholder", "ANTHROPIC_API_KEY": "placeholder",
                "FIREWORKS_API_KEY": "placeholder",
                # No ADC file, and no metadata server to fall back to.
                "GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent",
                "GCE_METADATA_HOST": "127.0.0.1:1",
                "DAM_DATA_ROOT": os.environ.get("DAM_DATA_ROOT", "/tmp/dam-nonexistent")})
    import subprocess
    proc = subprocess.run([sys.executable, "-c", "import bench"],
                          capture_output=True, text=True, cwd=DAM, env=env)
    assert proc.returncode == 0, (
        f"importing bench needs more than the documented API keys:\n{proc.stderr}")


# ── judge_token_budget ──────────────────────────────────────────────────────────

def test_openai_reasoning_judge_gets_completion_kwarg_and_a_floor():
    kwarg, limit = bench.judge_token_budget("o4-mini", 64)
    assert kwarg == "max_completion_tokens"
    assert limit >= 2048, "a 64-token budget is spent reasoning, leaving no verdict"


def test_plain_judge_keeps_the_ordinary_kwarg_and_limit():
    kwarg, limit = bench.judge_token_budget("gpt-4.1", 64)
    assert (kwarg, limit) == ("max_tokens", 64)


def test_reasoning_flag_is_honoured_for_non_openai_ids():
    """The regression: reasoning was inferred from an OpenAI name prefix.

    Any model the registry flags `reasoning: true` needs the floor, whatever it is
    called. The example config flags two Fireworks models precisely because their
    ids look nothing like OpenAI's.
    """
    assert registry.BEDROCK_REASONING, "no model carries reasoning: true"
    for model_id in registry.BEDROCK_REASONING:
        kwarg, limit = bench.judge_token_budget(model_id, 64)
        assert kwarg == "max_completion_tokens", f"{model_id} judged without a floor"
        assert limit >= 2048


def test_the_floor_never_lowers_a_larger_budget():
    _, limit = bench.judge_token_budget("o4-mini", 8192)
    assert limit == 8192


# ── the Anthropic effort gate ───────────────────────────────────────────────────

class _Block:
    type = "text"
    text = "done"

    def model_dump(self):
        return {"type": "text", "text": "done"}


class _Resp:
    stop_reason = "end_turn"
    model = "stub"
    content = [_Block()]
    usage = types.SimpleNamespace(input_tokens=1, output_tokens=1,
                                  cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0)


@pytest.fixture
def captured_request(monkeypatch):
    """Run one native-Anthropic turn against a stub and return its kwargs."""
    seen: dict = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _Resp()

    monkeypatch.setattr(bench, "anthropic_native",
                        types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))
    return seen


def _run(model_id: str) -> None:
    bench.run_candidate_messages_api(
        "q", model_id, ch_query=lambda q: "[]", system_prompt="s",
        thinking="on", effort="high")


def test_effort_is_withheld_from_models_that_reject_it(captured_request):
    """A public Claude id is not in EFFORT_CAPABLE, so the request carries neither."""
    model_id = "claude-sonnet-4-5"
    assert model_id not in registry.EFFORT_CAPABLE
    _run(model_id)
    extra = captured_request["extra_body"]
    assert extra == {}, f"would 400 on public Anthropic: {extra}"


def test_effort_is_sent_to_models_that_accept_it(captured_request):
    """And the gate does not silently disable the sweep for our own models."""
    if not registry.EFFORT_CAPABLE:
        pytest.skip("no effort-capable model in this registry")
    model_id = sorted(registry.EFFORT_CAPABLE)[0]
    _run(model_id)
    extra = captured_request["extra_body"]
    assert extra.get("output_config", {}).get("effort") == "high", extra


# ── per-warehouse probe table, snapshot column and schema prompt ────────────────

@pytest.fixture(scope="module")
def example_db(tmp_path_factory) -> Path:
    """Seed the example warehouse out of tree (chDB rewrites its store on open)."""
    import subprocess
    db = tmp_path_factory.mktemp("wh") / "warehouse"
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE / "seed_example.py"), "--db-path", str(db)],
        capture_output=True, text=True, cwd=DAM)
    assert proc.returncode == 0, f"seed failed:\n{proc.stdout}\n{proc.stderr}"
    return db


def test_a_foreign_warehouse_opens_with_its_own_probe_and_snapshot(example_db):
    wh = warehouse_mod.Warehouse(
        db_path=example_db, system_prompt_path=EXAMPLE / "schema.md",
        probe_table="marts.usage_daily", snapshot_column="day")
    try:
        assert wh.n_rows == 960
        # Read off the data, not off a constant: the snapshot date is what makes
        # "last 30 days" resolve identically in annotation and eval.
        assert wh.snapshot_date == "2026-06-30"
    finally:
        wh.close()


def test_the_defaults_fail_loudly_on_a_foreign_warehouse(example_db):
    """Our table names are absent, and the error has to say what to pass."""
    with pytest.raises(RuntimeError) as e:
        warehouse_mod.Warehouse(db_path=example_db,
                                system_prompt_path=EXAMPLE / "schema.md")
    assert "probe_table" in str(e.value), (
        f"error does not name the parameter to set: {e.value}")


def test_the_schema_prompt_comes_from_the_warehouse_not_a_constant(example_db):
    wh = warehouse_mod.Warehouse(
        db_path=example_db, system_prompt_path=EXAMPLE / "schema.md",
        probe_table="marts.usage_daily", snapshot_column="day")
    try:
        prompt = wh.system_prompt()
    finally:
        wh.close()
    assert "marts.usage_daily" in prompt, "example schema not used"
    assert "dbt_marts_general" not in prompt, "board table names leaked into the example"
    assert "{snapshot_date}" not in prompt, "snapshot placeholder left unrendered"
    assert wh.snapshot_date in prompt


# ── the board's ground-truth path ───────────────────────────────────────────────

def _stats_module():
    """08_results_stats.py is not an importable name, so load it by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "results_stats", DAM / "08_results_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_linker_stats_refuse_a_mismatched_ground_truth(tmp_path):
    """Results scored against another run's ground truth must not report 0.0.

    Every unmatched pair counts as "not fired", so a mismatch produced a clean
    zero rather than an error: the section read as a measurement when the
    trace_id intersection was empty. `08_results_stats.py` had no --annot flag at
    all, so pointing --results at the example while the board data was present
    did exactly this (AI-1862).
    """
    mod = _stats_module()
    annot = tmp_path / "annotated.jsonl"
    annot.write_text(json.dumps({"trace_id": "board-001", "gt_results": ["[]"]}) + "\n")
    rows = [{"trace_id": "ex-001", "candidates": {}}]
    with pytest.raises(SystemExit) as e:
        mod.linker_firing(rows, annot)
    assert "--annot" in str(e.value), f"error does not say how to fix it: {e.value}"


def test_linker_stats_accept_a_matching_ground_truth(tmp_path):
    """And the guard does not fire on a legitimately empty candidate set."""
    mod = _stats_module()
    annot = tmp_path / "annotated.jsonl"
    annot.write_text(json.dumps({"trace_id": "ex-001", "gt_results": ["[]"]}) + "\n")
    rows = [{"trace_id": "ex-001", "candidates": {}}]
    assert isinstance(mod.linker_firing(rows, annot), dict)


# ── budgets and errors that only a foreign warehouse or registry reaches ─────────

def test_a_flagged_reasoning_model_gets_the_larger_candidate_budget():
    """The same flag has to mean the same thing to a candidate and to a judge.

    The candidate runner sized its budget from a list of id substrings, written
    when "emits inline reasoning" had to be guessed from a name. A registry-flagged
    model matching none of them (glm-5p2 in the example config) got the reasoning
    floor as a judge and the 2048 cap as a candidate, from one flag.
    """
    src = (DAM / "bench.py").read_text()
    i = src.index("def run_candidate_openai_compat")
    body = src[i:i + 3000]
    assert "model_id in BEDROCK_REASONING" in body, (
        "candidate budget ignores the registry reasoning flag")


def test_a_bad_snapshot_column_says_so(example_db):
    """query() returns errors as a string, so an unparsed one used to surface as
    JSONDecodeError several frames from the cause, naming neither table nor column.
    """
    with pytest.raises(RuntimeError) as e:
        # probe_table given, snapshot_column left on the board default
        warehouse_mod.Warehouse(db_path=example_db,
                                system_prompt_path=EXAMPLE / "schema.md",
                                probe_table="marts.usage_daily")
    msg = str(e.value)
    assert "snapshot_column" in msg, f"does not name the parameter: {msg}"
    assert "marts.usage_daily" in msg, f"does not name the table: {msg}"


# ── the column linker ───────────────────────────────────────────────────────────

def test_the_linker_resolves_from_the_registry():
    """Not a hardcoded provider. It is a scoring component, so a hardcoded Bedrock
    id made correct-but-differently-named answers score wrong for anyone without
    an AWS account, with nothing said. The example config reaches it on an API key.
    """
    assert (registry.LINKER in registry.MODELS
            or registry.LINKER in registry.JUDGE_MODEL_IDS), (
        f"linker {registry.LINKER!r} resolves to nothing, so it would silently "
        f"become the default judge")
    src = (DAM / "bench.py").read_text()
    i = src.index("def _link_columns")
    body = src[i:i + 2500]
    assert "bedrock.converse" not in body, "linker still calls Bedrock directly"
    assert "_judge_complete(LINKER" in body, "linker does not route through the registry"


def test_the_linker_fallback_is_loud(monkeypatch, capsys):
    """Silent degradation is the actual defect: the fallback changes the scoring
    rule mid-run, so it has to announce itself."""
    monkeypatch.setattr(bench, "_COL_LINK_WARNED", False)
    monkeypatch.setattr(bench, "_COL_LINK_CACHE", {})

    def boom(*a, **k):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(bench, "_judge_complete", boom)
    assert bench._link_columns(["spend_usd"], ["total_spend"]) == {}
    err = capsys.readouterr().err
    assert "column linker" in err and "score as failures" in err, (
        f"degraded silently: {err!r}")


def test_the_linker_pins_temperature_zero(monkeypatch):
    """Scoring has to be reproducible, so the same column sets map the same way.

    The old direct Bedrock call passed temperature 0. Routing through
    _judge_complete dropped it, and every provider default is 1.0, so aliased
    result sets could map differently between runs while the docstring still
    promised determinism.
    """
    monkeypatch.setattr(bench, "_COL_LINK_CACHE", {})
    seen = {}

    def fake(name, prompt, max_tokens=512, temperature=None):
        seen["temperature"] = temperature
        return '{"mapping": {}}'

    monkeypatch.setattr(bench, "_judge_complete", fake)
    bench._link_columns(["spend_usd"], ["total_spend"])
    assert seen["temperature"] == 0, "linker no longer pins temperature"


def test_temperature_is_withheld_from_reasoning_judges(monkeypatch):
    """They reject anything but their default, so an explicit 0 is a 400."""
    seen = {}

    def create(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(bench, "openai_client", types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
    reasoning = [n for n in ("o4-mini", "o3-mini") if n in registry.MODELS]
    if not reasoning:
        pytest.skip("no reasoning judge in this registry")
    with pytest.raises(Exception):
        bench._judge_complete(reasoning[0], "hi", 256, temperature=0)
    assert "temperature" not in seen, "would 400 on a reasoning model"
