"""Guards for the Jev failure-mode labeler.

The labeler is offline analysis, but three things silently corrupt its output if
they drift, so they are pinned here:

  * the fail filter and the ground-truth join, because labeling a passing cell or
    dropping the reference SQL feeds Jev the wrong material,
  * the taxonomy/rule wiring, because a sub-mode with no family or a rule question
    missing the 'none' escape hatch skews every aggregate,
  * response parsing, because the family must follow from the chosen sub-mode.

All pure logic: no SDK, no warehouse, no network. The primitives and the Jev
response are stubbed, so this asserts the shapes the real SDK must satisfy. The rule
set is the shipped example (config/rules.example.yaml).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DAM = Path(__file__).resolve().parents[1]  # experiments/data-agent-mnist
sys.path.insert(0, str(DAM))

import label_failure_modes as lfm  # noqa: E402
import rule_checks as rc  # noqa: E402

RULES = rc.load_rules(DAM / "config/rules.example.yaml")


# ── stubs standing in for the typesafe_sdk primitives and response ───────────────
class _Choice:
    def __init__(self, instructions, criteria):
        self.instructions, self.criteria = instructions, criteria


class _Noul:
    def __init__(self, instructions):
        self.instructions = instructions


class _Ans:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, choices, nouls):
        self.choices, self.nouls = choices, nouls


PRIMS = (_Choice, _Noul)


def _write(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ── loading ──────────────────────────────────────────────────────────────────────
def test_iter_failed_keeps_fail_joins_ground_truth(tmp_path):
    """Only failed cells are yielded, and each carries the joined reference SQL/answer."""
    _write(tmp_path / "annotated.jsonl",
           [{"trace_id": "t1", "gt_sql": ["SELECT 1"], "gt_answer": "one"}])
    _write(tmp_path / "results.jsonl", [{
        "trace_id": "t1", "nl_question": "how many?",
        "candidates": {
            "m_fail": {"result_score": {"outcome": "fail"}, "sqls": ["SELECT 2"],
                       "sql_results": ["2"], "final_answer": "two", "turns": 3, "error": None},
            "m_pass": {"result_score": {"outcome": "pass"}, "sqls": ["SELECT 1"]},
            "m_tie":  {"result_score": {"outcome": "tie"}, "sqls": ["SELECT 3"]},
        }}])
    gt = lfm.load_ground_truth(tmp_path / "annotated.jsonl")

    cells = list(lfm.iter_failed(tmp_path / "results.jsonl", gt))
    assert [c["model"] for c in cells] == ["m_fail"]
    c = cells[0]
    assert c["gt_sql"] == ["SELECT 1"] and c["gt_answer"] == "one"
    assert c["nl_question"] == "how many?" and c["candidate_sqls"] == ["SELECT 2"]

    with_tie = {c["model"] for c in lfm.iter_failed(tmp_path / "results.jsonl", gt,
                                                    include_tie=True)}
    assert with_tie == {"m_fail", "m_tie"}


def test_sample_failed_is_deterministic_subset():
    """Reservoir sampling returns n items from the stream, the same set for a seed,
    and all of them when the stream is shorter than n."""
    items = [{"i": i} for i in range(100)]
    a = lfm.sample_failed(iter(items), 10, seed=0)
    b = lfm.sample_failed(iter(list(items)), 10, seed=0)
    assert len(a) == 10 and a == b and all(x in items for x in a)
    assert lfm.sample_failed(iter(items[:5]), 10, seed=0) == items[:5]


def test_missing_ground_truth_does_not_crash(tmp_path):
    """A results row with no matching annotation still yields, with empty reference."""
    _write(tmp_path / "results.jsonl", [{
        "trace_id": "orphan", "nl_question": "q",
        "candidates": {"m": {"result_score": {"outcome": "fail"}}}}])
    cells = list(lfm.iter_failed(tmp_path / "results.jsonl", {}))
    assert cells[0]["gt_sql"] == [] and cells[0]["gt_answer"] == ""


# ── request shape ──────────────────────────────────────────────────────────────
def test_build_state_truncates_result_strings():
    cell = {"nl_question": "q", "gt_sql": [], "gt_answer": "", "candidate_sqls": [],
            "sql_results": ["x" * 5000], "final_answer": "", "turns": 1, "error": None}
    state = lfm.build_state(cell, max_result_chars=100)
    assert len(state["candidate_results"][0]) < 200
    assert state["candidate_results"][0].endswith("...(truncated)")


def test_questions_cover_taxonomy_and_rules():
    q = lfm.build_questions(PRIMS, RULES)
    assert set(q) == set(lfm.QUESTION_IDS)
    # every sub-mode is offered; there is no separate family question (family is derived)
    all_subs = {s for subs in lfm.TAXONOMY.values() for s in subs}
    assert set(q["sub_mode"].criteria) == all_subs
    assert "family" not in q
    # the rule question offers only the semantic rules plus a no-match escape hatch; the
    # decidable rules are scored in code and must never be offered to the model
    crit = set(q["prevented_by_rule"].criteria)
    assert lfm.NO_RULE in crit
    assert set(lfm.semantic_rules(RULES)) <= crit
    assert crit.isdisjoint(rc.decidable_rule_keys(RULES))


def test_every_sub_mode_has_a_family():
    all_subs = {s for subs in lfm.TAXONOMY.values() for s in subs}
    assert set(lfm.FAMILY_OF) == all_subs


def test_rules_partition_into_decidable_and_semantic():
    """Every schema-prompt rule is scored by exactly one path: code (decidable) or Jev
    (semantic). A rule in neither set silently drops out of the report."""
    semantic = set(lfm.semantic_rules(RULES))
    decidable = set(rc.decidable_rule_keys(RULES))
    assert semantic.isdisjoint(decidable)
    assert semantic | decidable == set(lfm.prompt_rules(RULES))


# ── response parsing ─────────────────────────────────────────────────────────────
def test_parse_response_derives_family_from_sub_mode():
    resp = _Resp(
        choices={
            "sub_mode": _Ans(choice="wrong_join", confidence=0.8,
                             probabilities={"wrong_join": 0.5, "wrong_grain": 0.2,
                                            "wrong_table": 0.3}),
            "prevented_by_rule": _Ans(choice="crm_join", confidence=0.7),
        },
        nouls={
            "schema_hint_would_help": _Ans(noul=0.9),
            "correct_despite_fail": _Ans(noul=0.1),
        })
    lab = lfm.parse_response(resp)
    # family is derived from the sub-mode, never asked, so it cannot contradict it
    assert lab["sub_mode"] == "wrong_join" and lab["family"] == "query_construction"
    # family confidence rolls up the sub-mode probabilities within that family
    # (wrong_join 0.5 + wrong_grain 0.2 are query_construction; wrong_table 0.3 is not)
    assert abs(lab["family_confidence"] - 0.7) < 1e-9
    assert lab["prevented_by_rule"] == "crm_join"
    assert lab["schema_hint_would_help"] == 0.9 and lab["correct_despite_fail"] == 0.1
