"""Guards for the entity-recovery output-shape bucketing.

The entity probe scores 1 if the exact id appears in the reply and 0 otherwise,
so its headline result is a floor: no model recovered a synthetic id. That floor
only means something if the models actually answered, and for 10 of 26 they did
not. `entity_shape` is what makes the difference visible in the saved rows, so
the buckets are the contract and they are tested here rather than inferred
later by re-reading generations.

The module is a numbered script, so it is loaded by path rather than imported.
Only the pure classifier is exercised: nothing here makes a provider call.
"""
from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest

DAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAM))


def _load_probe_bits():
    """Pull the classifier out of the script without importing `bench`.

    `12_contamination_probe.py` imports bench at module scope, which constructs
    provider clients and needs credentials. The classifier has no dependency on
    any of that, so it is exec'd from the source slice instead. If the source
    layout changes this fails loudly rather than skipping.
    """
    src = (DAM / "12_contamination_probe.py").read_text()
    start = src.index("def _id_signature(")
    end = src.index("def similarity(")
    ns: dict = {"uuid": uuid}
    exec(compile(src[start:end], "12_contamination_probe.py", "exec"), ns)  # noqa: S102
    return ns["entity_shape"], ns["build_id_profile"], ns["SHAPES"]


entity_shape, build_id_profile, SHAPES = _load_probe_bits()

UUID = "266dbbc7-60f5-44b4-813a-c0f99ffed99e"   # 36 chars
SFID = "001sdXZQLFZmFEqbwa"                     # 18 chars, the other id shape in the warehouse
# The board column mixes shapes: 250 uuids, 24 mixed-case alphanumeric keys and
# one 8-character hex string. MIXED is that column; UUID_ONLY is the homogeneous
# case a different warehouse is likely to have.
MIXED = build_id_profile([UUID, SFID, "3de5fbad"])
UUID_ONLY = build_id_profile([UUID, "6ad05292-6c78-44c1-8566-3fa2cc845139"])


@pytest.mark.parametrize("gen,answer,expected", [
    # Silence. gpt-5.5 returned this on all 25 entity calls before the fix,
    # having spent its whole 2048-token budget reasoning.
    ("", UUID, "empty"),
    ("   \n ", UUID, "empty"),
    # Recovery, bare and embedded. Both score 1, so both must bucket as a hit or
    # the shape column would contradict the score column.
    (UUID, UUID, "hit"),
    (f"The id is {UUID}.", UUID, "hit"),
    # Wrong but drawn from an alphabet the ids use: the model attempted an answer.
    ("a1b2c3d4e5f6g7h8i9", SFID, "id_shaped"),
    (UUID.replace("266", "999"), UUID, "id_shaped"),
    # Right length, wrong alphabet. No id in this column is all digits, and the
    # length rule this replaced called it an attempt.
    ("582941735301954688", SFID, "degenerate"),
    # Far too short to be an id. These are the observed replies from gpt-4.1,
    # gemini-3.1-pro and deepseek-v3.2, and they are guesses, not truncation:
    # finish_reason was "stop" with two completion tokens.
    ("100010", UUID, "degenerate"),
    ("320", SFID, "degenerate"),
    ("42", SFID, "degenerate"),
    # Refusals, long and short. The short one is why the rule splits on
    # whitespace rather than on length alone.
    ("I do not have access to that benchmark, so I cannot look up the id.", UUID, "prose"),
    ("I don't know.", SFID, "prose"),
])
def test_buckets(gen, answer, expected):
    assert entity_shape(gen, answer, MIXED) == expected


def test_every_bucket_is_declared():
    """SHAPES drives the summary table's columns, so a bucket the classifier can
    return but SHAPES omits would be silently dropped from the report."""
    produced = {entity_shape(g, a, MIXED) for g, a in [
        ("", UUID), (UUID, UUID), ("a1b2c3d4e5f6g7h8i9", SFID),
        ("42", SFID), ("I don't know.", SFID)]}
    assert produced <= set(SHAPES)
    assert produced == set(SHAPES), f"declared but unreachable here: {set(SHAPES) - produced}"


def test_one_threshold_for_the_whole_run():
    """The same reply must bucket the same way whichever row drew it.

    The column mixes id lengths (36, 18 and 8 characters here), so deriving the
    threshold from the row's own answer meant a ten-character reply was
    id_shaped against a Salesforce key and degenerate against a uuid, from one
    model behaving one way. The run passes the shortest id instead.
    """
    assert (entity_shape("a1b2c3d4e5", UUID, MIXED)
            == entity_shape("a1b2c3d4e5", SFID, MIXED) == "id_shaped")


def test_the_profile_follows_the_warehouse_not_a_constant():
    """The probe's table and id column are configurable, so a warehouse with
    short ids must not have every correct-length reply called degenerate."""
    short = build_id_profile(["ab12", "cd34"])
    assert entity_shape("ef56", "ab12", short) == "id_shaped"
    assert entity_shape("ef56", UUID, MIXED) == "degenerate"


def test_hit_wins_over_shape():
    """A reply containing the id is a hit even when it is wrapped in prose that
    would otherwise bucket as a refusal."""
    assert entity_shape(f"I am not sure, but it may be {UUID}", UUID, MIXED) == "hit"


def test_digits_are_not_id_shaped_when_no_id_is_all_digits():
    """The length rule this replaced accepted eighteen digits against an
    18-character key. No id in the column uses that alphabet."""
    assert entity_shape("582941735301954688", SFID, MIXED) == "degenerate"


def test_uuid_case_does_not_change_the_bucket():
    """Hex is case-insensitive, so an uppercase reply is the same uuid.

    The signature test alone read case as a different alphabet, so a valid
    mixed-case uuid bucketed as degenerate against a lowercase sample. uuid
    parsing now runs first wherever the column holds uuids at all, not only
    where it holds nothing else.
    """
    other = "6ad05292-6c78-44c1-8566-3fa2cc845139"
    for reply in (other, other.upper(), "6AD05292-6c78-44C1-8566-3FA2CC845139"):
        assert entity_shape(reply, UUID, MIXED) == "id_shaped", reply


def test_a_column_without_uuids_does_not_accept_one():
    """The uuid branch is conditional on the column, not a blanket rule."""
    no_uuid = build_id_profile([SFID, "3de5fbad"])
    assert entity_shape("6ad05292-6c78-44c1-8566-3fa2cc845139", SFID, no_uuid) == "degenerate"
    assert entity_shape("001abcDEF123456789", SFID, no_uuid) == "id_shaped"


def test_a_uuid_column_is_checked_exactly():
    """Where every sampled id is a uuid there is no need to guess at shape:
    uuid.UUID either parses the reply or it does not."""
    assert UUID_ONLY["all_uuid"] is True
    assert entity_shape("6ad05292-6c78-44c1-8566-3fa2cc845139", UUID, UUID_ONLY) == "id_shaped"
    # right alphabet, right length, still not a uuid
    assert entity_shape("6ad05292-6c78-44c1-8566-3fa2cc84513z", UUID, UUID_ONLY) == "degenerate"
    assert MIXED["all_uuid"] is False


def test_a_well_formed_but_wrong_uuid_is_an_attempt():
    """gpt-5.5 invents these. It is a different observation from a two-digit
    stub and the buckets have to keep them apart."""
    assert entity_shape("6ad05292-6c78-44c1-8566-3fa2cc845139", UUID, MIXED) == "id_shaped"
    assert entity_shape("39", UUID, MIXED) == "degenerate"
