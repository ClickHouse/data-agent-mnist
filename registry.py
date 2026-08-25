"""Model registry, provider endpoints and judge seats, loaded from configuration.

Why this is not a dict in bench.py. Three reasons, in order of how much
they bite:

1. Publishing bench.py publishes the catalog. The registry names every board model
   with its provider id, plus our internal hosts and Bedrock inference-profile ids.
   Gate 4 of the partner export exists specifically to keep that out of a bundle;
   shipping the module would hand the same information to everyone.
2. Membership IS routing. `run_candidate` dispatches on which dict a key appears
   in, so nine provider dicts and four side-sets encoded the routing table as
   set membership. A model's provider and its capabilities are properties OF the
   model, so they belong in one entry per model, not scattered across thirteen
   collections that have to stay mutually consistent by hand.
3. Reading the registry required credentials. Importing bench constructs six
   provider clients (OpenAI-compatible, Bedrock, GCP auth), so any tool wanting to
   know which models exist had to either hold full provider credentials or parse
   the source with `ast`, which is what export/bundle.py does today. A config file
   makes the registry readable without importing anything.

The config path comes from DAM_MODELS_CONFIG, defaulting to config/models.yaml
beside this module. The repository ships config/models.example.yaml with public
model ids only.

Schema, one entry per model:

    models:
      <key>:
        id: <provider's model id>
        provider: bedrock | mantle | openai | gemini | fireworks | anthropic | gateway
        api: responses                 # optional; OpenAI + mantle Responses API
        vertex_location: global        # optional; Gemini 3.x is global-only
        reasoning: true                # optional; larger output budget on Bedrock
        adaptive_thinking_only: true   # optional; Mythos-class, no "disabled" mode
        supports_effort: true          # optional; accepts thinking/effort controls
        retired: true                  # optional; kept resolvable, never run

`retired` models stay in the registry deliberately. verify_board replays every
cached candidate in results.jsonl, and export/bundle.py's resolve_models fails
closed on an unknown model, so dropping their ids would break board reproduction
and the bundle leak scan (which uses those ids as forbidden markers).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(os.environ.get(
    "DAM_MODELS_CONFIG", Path(__file__).resolve().parent / "config/models.yaml"))


def load(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or CONFIG_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"model registry not found at {p}. Copy config/models.example.yaml to "
            f"config/models.yaml, or point DAM_MODELS_CONFIG at your own.")
    cfg = yaml.safe_load(p.read_text()) or {}
    for section in ("models", "endpoints", "judges"):
        if section not in cfg:
            raise ValueError(f"{p}: missing required section '{section}'")
    for key, e in cfg["models"].items():
        missing = {"id", "provider"} - set(e or {})
        if missing:
            raise ValueError(f"{p}: model '{key}' is missing {sorted(missing)}")
    return cfg


_CFG = load()
MODELS: dict[str, dict[str, Any]] = _CFG["models"]
ENDPOINTS: dict[str, str] = _CFG["endpoints"]


def _by(**match) -> dict[str, str]:
    """-> {key: id} for models whose entry matches every given field.

    `None` matches "field absent or falsy", which is how the optional flags read.
    """
    out = {}
    for k, e in MODELS.items():
        if all((e.get(f) == v) if v is not None else (not e.get(f))
               for f, v in match.items()):
            out[k] = e["id"]
    return out


def _flagged(flag: str) -> set[str]:
    return {k for k, e in MODELS.items() if e.get(flag)}


# ── the shapes bench.py consumes ──────────────────────────────────────────────
# Provider dicts exclude retired models, matching the previous literals: retired
# entries lived in their own dict and were never merged into ALL_CANDIDATES.
CANDIDATES: dict[str, str] = _by(provider="bedrock", retired=None)
MANTLE_CANDIDATES: dict[str, str] = _by(provider="mantle", retired=None, api=None)
MANTLE_RESPONSES_CANDIDATES: dict[str, str] = _by(provider="mantle", retired=None,
                                                  api="responses")
OPENAI_CANDIDATES: dict[str, str] = _by(provider="openai", retired=None)
GEMINI_CANDIDATES: dict[str, str] = _by(provider="gemini", retired=None)
FIREWORKS_CANDIDATES: dict[str, str] = _by(provider="fireworks", retired=None)
ANTHROPIC_CANDIDATES: dict[str, str] = _by(provider="anthropic", retired=None)
GATEWAY_CANDIDATES: dict[str, str] = _by(provider="gateway", retired=None)
RETIRED_CANDIDATES: dict[str, str] = {k: e["id"] for k, e in MODELS.items()
                                      if e.get("retired")}

ALL_CANDIDATES: dict[str, str] = {k: e["id"] for k, e in MODELS.items()
                                  if not e.get("retired")}

# Capability sets. These were separate literals that had to be kept in step with
# the provider dicts by hand; now they are views over the same entries.
OPENAI_RESPONSES_ONLY: set[str] = {k for k in OPENAI_CANDIDATES
                                   if MODELS[k].get("api") == "responses"}
GEMINI_GLOBAL: set[str] = {k for k, e in MODELS.items()
                           if e.get("vertex_location") == "global"}
ADAPTIVE_THINKING_ONLY: set[str] = {MODELS[k]["id"] for k in _flagged("adaptive_thinking_only")}
# Anthropic's thinking/effort controls are not universal: public Claude models
# reject `output_config.effort` outright ("This model does not support the effort
# parameter"). Sending them unconditionally made the native Anthropic path usable
# only by models that happen to accept them, which is ours. Opt in.
EFFORT_CAPABLE: frozenset[str] = frozenset(
    MODELS[k]["id"] for k in _flagged("supports_effort"))
# Exact model ids, not substrings. The old constant was the fragment
# "claude-opus-5" matched with `in`, which was a proxy for "is this a reasoning
# model" back when the answer had to be inferred from the id. The flag states it
# directly, so the match can be exact: same set today (verified), and it cannot
# silently capture a future id that happens to contain the fragment.
BEDROCK_REASONING: frozenset[str] = frozenset(
    MODELS[k]["id"] for k in _flagged("reasoning"))

_J = _CFG["judges"]
JUDGE_SEATS: dict[str, list[str]] = _J["seats"]
JUDGE_MODEL_IDS: dict[str, str] = _J.get("extra_ids", {})
JUDGE_MODEL: str = _J["default_id"]
JUDGE_PROVIDER: dict[str, str] = {m: prov for prov, ms in JUDGE_SEATS.items() for m in ms}


def _default_linker() -> str:
    """Cheapest seat model, deterministically. Seats are listed best-first."""
    for provider in sorted(JUDGE_SEATS):
        if JUDGE_SEATS[provider]:
            return JUDGE_SEATS[provider][-1]
    raise KeyError("no judge seat holds a model, so no column linker can be chosen")


# The column linker's model. It was a hardcoded Bedrock id calling converse
# directly, which quietly made a SCORING component require an AWS account: with
# no credentials the call raised, the fallback matched identical column names
# only, and equivalent answers written with different column names scored wrong.
# Nothing reported it. On the runnable example that was 24 of 56 compared pairs.
#
# One cached call per distinct pair of differing column sets, so this wants the
# cheapest capable model rather than the panel flagship. Set `judges.linker` to
# pin one; the default is the last model of the first seat.
LINKER: str = _J.get("linker") or _default_linker()
if LINKER not in MODELS and LINKER not in JUDGE_MODEL_IDS:
    # Unvalidated, a typo here resolves to the default judge on the anthropic
    # client, because that is _judge_complete's fallback for an unknown name. The
    # linker would then quietly run on a model nobody chose, in the scoring path.
    raise KeyError(
        f"judges.linker is {LINKER!r}, which is not a model in this registry and "
        f"not in judges.extra_ids. Use one of: {', '.join(sorted(MODELS))}")

ANNOTATORS: dict[str, str] = {k: MODELS[k]["id"] for k in _CFG.get("annotators", [])}
