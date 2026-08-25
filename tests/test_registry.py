"""Guards for the config-driven model registry.

Two classes of guard here, protecting different things.

STRUCTURAL. Moving the registry from thirteen hand-maintained collections into
one entry per model removed the possibility of them disagreeing, but only if the
views stay derived. These tests pin the derivation: providers partition the
models, retired entries stay out of the live set, and every capability set
follows from a flag rather than a parallel list.

REFERENTIAL. A config file can be internally inconsistent in ways a literal could
not: a judge seat naming a model that was renamed, an annotator that no longer
exists. Those fail at scoring time, deep in a paid run, which is the worst place
to discover them. They are cheap to catch here.

The example config gets the same treatment plus a publish check, since it is the
file an adopter copies and the one most likely to drift unnoticed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

DAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAM))

import registry  # noqa: E402

EXAMPLE = DAM / "config/models.example.yaml"
PROVIDERS = {"bedrock", "mantle", "openai", "gemini", "fireworks", "anthropic", "gateway"}


# ── structural ────────────────────────────────────────────────────────────────

def test_providers_partition_the_live_models():
    """Every live model appears in exactly one provider view. An overlap would
    make routing depend on dict ordering; a gap makes a model unroutable."""
    views = {
        "bedrock": registry.CANDIDATES, "mantle": registry.MANTLE_CANDIDATES,
        "mantle_responses": registry.MANTLE_RESPONSES_CANDIDATES,
        "openai": registry.OPENAI_CANDIDATES, "gemini": registry.GEMINI_CANDIDATES,
        "fireworks": registry.FIREWORKS_CANDIDATES,
        "anthropic": registry.ANTHROPIC_CANDIDATES, "gateway": registry.GATEWAY_CANDIDATES,
    }
    seen: dict[str, str] = {}
    for view, d in views.items():
        for k in d:
            assert k not in seen, f"{k} is in both {seen[k]} and {view}"
            seen[k] = view
    assert set(seen) == set(registry.ALL_CANDIDATES), (
        f"unrouted: {set(registry.ALL_CANDIDATES) - set(seen)}")


def test_retired_models_are_resolvable_but_never_live():
    """Retired ids stay resolvable on purpose: verify_board replays every cached
    candidate and the bundle leak scan uses the ids as forbidden markers.

    Having any retired entry is a property of the registry in use, not of the
    code. Our own config has several; a fresh one (config/models.example.yaml,
    or an adopter's) has none, and asserting otherwise failed the suite for
    exactly the reader the example is for.
    """
    if not registry.RETIRED_CANDIDATES:
        pytest.skip("registry has no retired entries; nothing to separate")
    overlap = set(registry.RETIRED_CANDIDATES) & set(registry.ALL_CANDIDATES)
    assert not overlap, f"retired models leaked into the live set: {overlap}"
    for k, mid in registry.RETIRED_CANDIDATES.items():
        assert mid, f"{k} has no id, so it cannot be resolved or scanned for"


def test_capability_sets_follow_from_flags():
    assert registry.OPENAI_RESPONSES_ONLY == {
        k for k, e in registry.MODELS.items()
        if e["provider"] == "openai" and e.get("api") == "responses"}
    assert registry.GEMINI_GLOBAL == {
        k for k, e in registry.MODELS.items() if e.get("vertex_location") == "global"}
    assert registry.BEDROCK_REASONING == frozenset(
        e["id"] for e in registry.MODELS.values() if e.get("reasoning"))


def test_bedrock_reasoning_matches_exactly_not_by_substring():
    """The old constant was the fragment "claude-opus-5" matched with `in`. Exact
    ids cannot silently capture a future model whose id contains a flagged one."""
    for mid in registry.BEDROCK_REASONING:
        assert mid in {e["id"] for e in registry.MODELS.values()}, (
            f"{mid} is not a declared model id, so the match is a substring again")


def test_every_provider_is_one_the_runner_can_route():
    unknown = {e["provider"] for e in registry.MODELS.values()} - PROVIDERS
    assert not unknown, f"no runner path for provider(s): {unknown}"


# ── referential ───────────────────────────────────────────────────────────────

def _check_references(cfg: dict, label: str):
    declared = set(cfg["models"])
    for seat, models in cfg["judges"]["seats"].items():
        for m in models:
            assert m in declared or m in cfg["judges"].get("extra_ids", {}), (
                f"{label}: judge seat '{seat}' names undeclared model '{m}'")
    for a in cfg.get("annotators", []):
        assert a in declared, f"{label}: annotator '{a}' is not a declared model"


def test_live_config_references_resolve():
    _check_references(registry.load(), "config/models.yaml")


def test_judge_seats_hold_one_provider_each():
    """The panel's whole guarantee is that no provider can hold a majority."""
    for seat, models in registry.JUDGE_SEATS.items():
        provs = {registry.MODELS[m]["provider"] for m in models if m in registry.MODELS}
        assert len(provs) <= 1, f"seat '{seat}' mixes providers: {provs}"


def test_panel_seat_count_is_odd():
    """An even panel has no majority, so disagreements degrade to `tie`. Allowed,
    but it should be a decision rather than an accident."""
    assert len(registry.JUDGE_SEATS) % 2 == 1, (
        f"{len(registry.JUDGE_SEATS)} seats: ties cannot be broken. Intentional for "
        f"a two-provider example, but not for the board.")


# ── the example an adopter copies ─────────────────────────────────────────────

def test_example_config_loads_and_references_resolve():
    cfg = yaml.safe_load(EXAMPLE.read_text())
    _check_references(cfg, "models.example.yaml")
    for k, e in cfg["models"].items():
        assert {"id", "provider"} <= set(e), f"example model '{k}' is incomplete"
        assert e["provider"] in PROVIDERS, f"example model '{k}': bad provider"


def test_example_contains_no_internal_identifier():
    """The example is the file most likely to be published. Catch a paste here
    rather than in the publish gate, or after.

    The markers come from DAM_LEAK_MARKERS rather than a literal list, because
    this test travels with the harness and the list is exactly what it must not
    carry: it named an internal hostname, a service name, a key prefix and an AWS
    ACCOUNT ID. Hardcoded, a leak scan published four of the five things it
    exists to keep unpublished.

    Set the variable in the private tree (CI does). Skipped without it, so the
    harness ships a working check with no list attached.
    """
    raw = os.environ.get("DAM_LEAK_MARKERS", "").strip()
    if not raw:
        pytest.skip("DAM_LEAK_MARKERS unset; supply the markers to scan for")
    txt = EXAMPLE.read_text().lower()
    for marker in (m.strip().lower() for m in raw.split(",") if m.strip()):
        assert marker not in txt, f"example config leaks an internal identifier: {marker}"


# Providers an adopter can reach with a plain API key, no cloud account.
KEY_ONLY = {"openai", "anthropic", "fireworks", "gateway"}


def test_example_is_runnable_without_a_cloud_account():
    """Every active provider must be a direct API-key signup. A Bedrock or Vertex
    entry here silently hands a reader with only API keys an unroutable model."""
    cfg = yaml.safe_load(EXAMPLE.read_text())
    provs = {e["provider"] for e in cfg["models"].values()}
    assert provs <= KEY_ONLY, (
        f"example activates provider(s) needing a cloud account: {provs - KEY_ONLY}")


def test_example_supports_the_methodology_it_demonstrates():
    """Three providers is not cosmetic. Ground truth is the result set at least
    two of three annotators agree on, and the panel seats one model per provider
    so none can hold a majority. With two providers both degrade: every annotator
    must agree (dropping questions on any disagreement), and a split judge vote
    scores `tie` instead of resolving. The example exists to demonstrate the
    method, so it has to be able to run it.
    """
    cfg = yaml.safe_load(EXAMPLE.read_text())
    seats = cfg["judges"]["seats"]
    assert len(seats) >= 3 and len(seats) % 2 == 1, (
        f"{len(seats)} judge seats: need an odd number, at least 3, for a "
        f"majority that no single provider can hold")
    annotator_provs = {cfg["models"][a]["provider"] for a in cfg["annotators"]}
    assert len(cfg["annotators"]) >= 3, (
        f"{len(cfg['annotators'])} annotators: majority-vote ground truth needs 3, "
        f"or '>=2 must agree' collapses to 'all must agree'")
    assert len(annotator_provs) == len(cfg["annotators"]), (
        f"annotators share a provider {annotator_provs}: correlated errors defeat "
        f"the point of a majority vote")


# ── failure modes ─────────────────────────────────────────────────────────────

def _load_in_subprocess(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import registry" % str(DAM)],
        capture_output=True, text=True, env={"PATH": "", "DAM_MODELS_CONFIG": str(path)})


def test_missing_file_fails_loudly(tmp_path):
    r = _load_in_subprocess(tmp_path / "nope.yaml")
    assert r.returncode != 0 and "model registry not found" in r.stderr


def test_model_without_an_id_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("models:\n  x:\n    provider: openai\nendpoints: {}\n"
                 "judges: {seats: {}, default_id: z}\n")
    r = _load_in_subprocess(p)
    assert r.returncode != 0 and "is missing ['id']" in r.stderr


def test_missing_section_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("models: {}\n")
    r = _load_in_subprocess(p)
    assert r.returncode != 0 and "missing required section" in r.stderr


def test_every_judge_has_a_reachable_client():
    """A judge whose provider has no client raises only when it is first called,
    which is mid-run. Worse, on a two-seat panel one unroutable judge leaves fewer
    than two live votes, so every question scores `error` rather than failing
    visibly. Both configs are checked because the example is the one an adopter
    runs first.
    """
    ROUTABLE = {"bedrock", "anthropic", "openai", "google", "gemini",
                "mantle", "fireworks", "gateway"}
    for label, cfg in (("config/models.yaml", registry.load()),
                       ("models.example.yaml", yaml.safe_load(EXAMPLE.read_text()))):
        for seat, judges in cfg["judges"]["seats"].items():
            for j in judges:
                prov = (cfg["models"][j]["provider"] if j in cfg["models"] else seat)
                assert prov in ROUTABLE, (
                    f"{label}: judge '{j}' in seat '{seat}' resolves to provider "
                    f"'{prov}', which has no judge client")


def test_example_judges_need_no_cloud_account():
    """The example advertises an API-key-only path. A judge on bedrock or gemini
    silently breaks that promise, and on a two-seat panel it breaks scoring."""
    cfg = yaml.safe_load(EXAMPLE.read_text())
    for seat, judges in cfg["judges"]["seats"].items():
        for j in judges:
            prov = cfg["models"][j]["provider"] if j in cfg["models"] else seat
            assert prov in KEY_ONLY, (
                f"example judge '{j}' needs a cloud account (provider '{prov}')")


def test_bench_imports_every_registry_name_it_uses():
    """bench.py imports specific names from registry, so using one it did not
    import is a NameError raised at call time, not at import. That is how a judge
    routing fix shipped broken through 16 passing tests: they exercised the
    registry, never bench's use of it. Checked statically so it needs no
    credentials, which importing bench would.
    """
    import ast
    src = (DAM / "bench.py").read_text()
    tree = ast.parse(src)
    imported = {a.asname or a.name
                for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                and n.module == "registry" for a in n.names}
    assigned = {t.id for n in ast.walk(tree)
                for t in ([n.targets[0]] if isinstance(n, ast.Assign)
                          and isinstance(n.targets[0], ast.Name)
                          else [n.target] if isinstance(n, ast.AnnAssign)
                          and isinstance(n.target, ast.Name) else [])}
    exported = {n for n in dir(registry) if n.isupper()}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    missing = sorted((used & exported) - imported - assigned)
    assert not missing, (
        f"bench.py uses registry name(s) it does not import: {missing}")


def test_a_mistyped_linker_is_rejected(tmp_path, monkeypatch):
    """Not silently promoted to the default judge.

    _judge_complete falls back to JUDGE_MODEL on an unknown name, so an
    unvalidated typo in judges.linker would run the scoring linker on a model
    nobody chose, on whichever client the fallback picks.
    """
    cfg = yaml.safe_load((DAM / "config" / "models.example.yaml").read_text())
    cfg["judges"]["linker"] = "no-such-model"
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(cfg))

    proc = subprocess.run(
        [sys.executable, "-c", "import registry"],
        capture_output=True, text=True, cwd=DAM,
        env={**os.environ, "DAM_MODELS_CONFIG": str(path),
             "DAM_DATA_ROOT": os.environ.get("DAM_DATA_ROOT", "/tmp/dam-nonexistent")})
    assert proc.returncode != 0, "a mistyped linker was accepted"
    assert "judges.linker" in proc.stderr, proc.stderr
