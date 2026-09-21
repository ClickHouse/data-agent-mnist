"""Real name lists for the anonymiser, loaded from configuration.

02b_anonymize.py needs hand-curated backstops the LLM extractor misses: real
customer and employee names, service instance names, and per-question post-scrub
edits. Those are the one part of the anonymiser that cannot ship, so they live in a
file outside the tree instead of as literals in the module.

The config path comes from DAM_NAMES_CONFIG, defaulting to config/names.yaml beside
this module. The repository ships config/names.example.yaml with invented names.

A missing file is a hard error, not a silent fallback. Running the anonymiser
without its backstops does not fail; it anonymises strictly worse while reporting
success, which is the failure this module exists to prevent. The generic half of the
anonymiser (safe-word allowlists, strip terms) is not sensitive and stays in
02b_anonymize.py.

Schema, all seven sections required:

    orgs:                  [str, ...]   org names, scrubbed whole-word
    services:              [str, ...]   service instance names, scrubbed whole-word
    case_sensitive_orgs:   [str, ...]   orgs that are also common words
    people:                [str, ...]   person names the extractor misses
    case_sensitive_people: [str, ...]   short/common-word person names
    scale_figures:         [str, ...]   trace ids whose figures are scaled
    question_overrides:    {trace_id: [[pattern, replacement], ...]}
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(os.environ.get(
    "DAM_NAMES_CONFIG", Path(__file__).resolve().parent / "config/names.yaml"))

_LIST_SECTIONS = ("orgs", "people", "case_sensitive_orgs", "case_sensitive_people",
                  "services", "scale_figures")


def load(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or CONFIG_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"anonymiser name lists not found at {p}. Copy config/names.example.yaml "
            f"to config/names.yaml, or point DAM_NAMES_CONFIG at your own.")
    cfg = yaml.safe_load(p.read_text()) or {}
    for section in (*_LIST_SECTIONS, "question_overrides"):
        if section not in cfg:
            raise ValueError(f"{p}: missing required section '{section}'")
    for section in _LIST_SECTIONS:
        if not isinstance(cfg[section], list):
            raise ValueError(f"{p}: section '{section}' must be a list")
    if not isinstance(cfg["question_overrides"], dict):
        raise ValueError(f"{p}: section 'question_overrides' must be a mapping")
    return cfg
