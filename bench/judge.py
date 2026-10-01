"""Blind cross-model judge: one verdict per seat, majority vote over the panel."""
import json
import random
import re
from collections import Counter

from registry import JUDGE_SEATS

from . import completion, scoring

_JUDGE_PROMPT = """\
You are evaluating whether two SQL agents answered a data warehouse question equivalently.

Question: {question}

A deterministic, schema-aware comparison of the two agents' returned result sets reports them as: {data_equiv}. This is the authoritative verdict on whether they retrieved the same DATA — trust it over your own reading of the rows below (the rows may be a truncated sample).

=== Answer {label_a} ===
SQL queries run: {n_a}
Result set (sample):
{results_a}
Conclusion: {answer_a}

=== Answer {label_b} ===
SQL queries run: {n_b}
Result set (sample):
{results_b}
Conclusion: {answer_b}

Decide whether the two answers are equivalent:
- For data-retrieval questions (lists, lookups, breakdowns) the result set IS the answer — defer to the result-set comparison above.
- For interpretive questions (yes/no, comparisons, "are they over X") the conclusion is the answer — two agents can reach the same correct conclusion from differently-shaped result sets, so weigh the conclusions even when the result sets differ.
Ignore style, formatting, and column-name differences.

Respond ONLY with JSON:
{{"verdict": "equivalent"|"not_equivalent"|"cannot_determine", "reason": "<one sentence>"}}

Use "cannot_determine" only if neither the result sets nor the conclusions give enough to judge."""


def _has_results(results: list) -> bool:
    return any(
        r and not r.startswith("Error:") and r != "(empty result)"
        for r in results
    )


def _fmt_results(results: list, max_chars: int = 2000, max_rows: int = 60) -> str:
    non_empty = [r for r in results if r and not r.startswith("Error:") and r != "(empty result)"]
    if not non_empty:
        return "(no results)"
    best     = max(non_empty, key=len)
    all_rows = [line for line in best.split("\n... (truncated)")[0].splitlines() if line.strip()]
    rows     = all_rows[:max_rows]
    text     = "\n".join(rows)
    if len(text) > max_chars:
        text = text[:max_chars] + " […]"
    # Tell the judge when WE truncated, so it never mistakes our sampling for the
    # agent answering incompletely.
    if len(rows) < len(all_rows):
        text += f"\n[showing {len(rows)} of {len(all_rows)} rows]"
    return text


def judge_score(
    question: str,
    gt_results: list, gt_answer: str,
    cand_results: list, cand_answer: str,
    judge_name: str = "opus47",
    data_equiv: bool | None = None,
) -> dict:
    """Single judge's verdict on whether a candidate matched ground truth.
    Blind (A/B randomized). `judge_name` selects the judge model (any provider).

    The data comparison is done deterministically (entity-linked `annotators_agree`,
    untruncated) and handed to the judge as a signal; the judge weighs it against the
    conclusions (authoritative for retrieval questions, advisory for interpretive
    ones). `data_equiv` is computed here if not supplied (the panel passes it once)."""
    gt_has   = _has_results(gt_results)
    cand_has = _has_results(cand_results)

    if not gt_has and not cand_has:
        return {"outcome": "tie", "verdict": "cannot_determine",
                "reasoning": "both sides returned no data", "judge": judge_name}
    if not gt_has:
        return {"outcome": "tie", "verdict": "cannot_determine",
                "reasoning": "ground truth returned no data", "judge": judge_name}
    if not cand_has:
        return {"outcome": "fail", "verdict": "not_equivalent",
                "reasoning": "candidate returned no data", "judge": judge_name}

    if data_equiv is None:
        data_equiv = scoring.annotators_agree(cand_results, gt_results)
    de_str = "EQUIVALENT" if data_equiv else "NOT EQUIVALENT"

    _swap    = random.random() < 0.5
    _a_res   = cand_results if _swap else gt_results
    _b_res   = gt_results   if _swap else cand_results
    _a_ans   = cand_answer  if _swap else gt_answer
    _b_ans   = gt_answer    if _swap else cand_answer
    _label_a, _label_b = ("B", "A") if _swap else ("A", "B")

    _prompt = _JUDGE_PROMPT.format(
        question=question, data_equiv=de_str,
        label_a=_label_a, n_a=len(_a_res), results_a=_fmt_results(_a_res),
        answer_a=(_a_ans or "(none)")[:4000],
        label_b=_label_b, n_b=len(_b_res), results_b=_fmt_results(_b_res),
        answer_b=(_b_ans or "(none)")[:4000],
    )
    _text  = completion._judge_complete(judge_name, _prompt, max_tokens=512)
    _match = re.search(r"\{.*\}", _text, re.DOTALL)
    try:
        _raw = json.loads(_match.group()) if _match else {}
    except Exception:
        _raw = {}
    _verdict = _raw.get("verdict", "cannot_determine")

    outcome = {"equivalent": "pass", "not_equivalent": "fail"}.get(_verdict, "tie")
    return {"outcome": outcome, "verdict": _verdict, "reasoning": _raw.get("reason", ""),
            "judge": judge_name, "data_equiv": de_str}


def select_panel(cand_model_name: str) -> list[str]:
    """One judge per provider seat, best-first, never the candidate itself
    (drops to a provider's #2 when its #1 is the candidate). Always 3 judges ->
    odd-sized -> a majority always resolves and no model judges itself."""
    panel: list[str] = []
    for models in JUDGE_SEATS.values():
        pick = next((m for m in models if m != cand_model_name), None)
        if pick is not None:
            panel.append(pick)
    return panel


def judge_panel(
    question: str,
    gt_results: list, gt_answer: str,
    cand_results: list, cand_answer: str,
    cand_model_name: str,
) -> dict:
    """Score a candidate with a provider-diverse panel. Majority vote over
    the judges' outcomes; a non-majority split (e.g. pass/tie/fail) scores 'tie'.
    Returns the aggregate plus each judge's individual vote."""
    panel = select_panel(cand_model_name)
    # Deterministic data-equivalence computed once (entity-linked, untruncated) and
    # shared across the panel, so all judges see the same authoritative data signal.
    data_equiv = (_has_results(cand_results) and _has_results(gt_results)
                  and scoring.annotators_agree(cand_results, gt_results))
    votes = []
    for jn in panel:
        try:
            votes.append(judge_score(question, gt_results, gt_answer,
                                     cand_results, cand_answer, judge_name=jn,
                                     data_equiv=data_equiv))
        except Exception as e:
            votes.append({"outcome": "error", "verdict": "error",
                          "reasoning": str(e), "judge": jn})

    ok_votes = [v for v in votes if v["outcome"] != "error"]
    tally  = Counter(v["outcome"] for v in ok_votes)
    ranked = tally.most_common()
    if len(ok_votes) < 2:
        outcome = "error"                    # need a real panel; <2 live judges isn't scorable
    elif len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        outcome = "tie"                      # no majority -> ambiguous
    else:
        outcome = ranked[0][0]

    return {"outcome": outcome, "panel": panel, "tally": dict(tally), "votes": votes}
