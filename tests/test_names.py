"""Guards for the config-driven anonymiser name lists.

The lists are the one part of the anonymiser that cannot ship, so they moved out of
the module into a file supplied per operator. These tests pin the two properties
that keeps that safe: the loader fails loudly when the file or a section is missing
(a silent fallback would anonymise worse while reporting success), and the shipped
example carries no real identifier.

Everything here runs against the example or a temp file, so it passes in the public
mirror where the real config/names.yaml is absent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

DAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAM))

import names  # noqa: E402

EXAMPLE = DAM / "config/names.example.yaml"
LIST_SECTIONS = ("orgs", "people", "case_sensitive_orgs", "case_sensitive_people",
                 "services", "scale_figures")


# ── the example an adopter copies ─────────────────────────────────────────────

def test_example_loads_with_every_section():
    cfg = names.load(EXAMPLE)
    for section in (*LIST_SECTIONS, "question_overrides"):
        assert section in cfg, f"example missing section '{section}'"
    for section in LIST_SECTIONS:
        assert isinstance(cfg[section], list) and all(isinstance(x, str) for x in cfg[section])
    assert isinstance(cfg["question_overrides"], dict)


def test_example_carries_no_real_trace_id():
    """The example is the file most likely to be published. A trace id is a bare
    32-hex token; the board-specific sections ship empty so none can appear."""
    text = EXAMPLE.read_text()
    assert re.search(r"\b[0-9a-f]{32}\b", text) is None, "example carries a 32-hex id"
    cfg = names.load(EXAMPLE)
    assert cfg["scale_figures"] == [] and cfg["question_overrides"] == {}, (
        "scale_figures and question_overrides are board-specific; ship them empty")


# ── failure modes ─────────────────────────────────────────────────────────────

def test_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="anonymiser name lists not found"):
        names.load(tmp_path / "nope.yaml")


def test_missing_section_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("orgs: []\n")
    with pytest.raises(ValueError, match="missing required section"):
        names.load(p)


def test_a_list_section_given_as_a_scalar_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    doc = {s: [] for s in LIST_SECTIONS}
    doc["question_overrides"] = {}
    doc["orgs"] = "Globex"           # a bare string, not a list
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="must be a list"):
        names.load(p)


def test_question_overrides_given_as_a_list_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    doc = {s: [] for s in LIST_SECTIONS}
    doc["question_overrides"] = []   # a list, not a mapping
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="must be a mapping"):
        names.load(p)


def test_real_config_loads_if_present():
    """Tree-dependent: the real file is private, so this skips in the mirror."""
    if not names.CONFIG_PATH.exists():
        pytest.skip("config/names.yaml absent (expected in the public mirror)")
    cfg = names.load()
    for section in (*LIST_SECTIONS, "question_overrides"):
        assert section in cfg
