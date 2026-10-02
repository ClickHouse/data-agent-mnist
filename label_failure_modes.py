"""Label analytical-agent failure modes with TypeSafe Jev.

Offline, read-only analysis. Reads a corpus's recorded runs (`results.jsonl`) and
ground truth (`annotated.jsonl`), keeps the candidate cells the judge scored `fail`,
and asks Jev a batch of typed questions about each failed trajectory:

  * a coarse failure family and a fine sub-mode (the two-level taxonomy below),
  * which stated schema-prompt rule, if any, would have prevented it (this is what
    turns the distribution into a ranked list of prompt sections to improve),
  * two yes/no flags: would a schema-use hint have prevented it, and does the candidate
    look correct despite the fail verdict (a judge-disagreement signal).

Jev classifies, it does not generate, so this runs off the measured harness and adds
no benchmark confound. It never runs a model against the warehouse and never writes
into the corpus.

The schema-prompt rules are data, not code: they load from a rules config
(DAM_RULES_CONFIG / --rules, default config/rules.example.yaml), so this names no
warehouse. Jev is reached through the native TypeSafe API: set TYPESAFE_API_KEY, and
TYPESAFE_BASE_URL to override the host. The SDK is an optional extra (`typesafe-sdk`),
pulled in lazily so the loader, the question builder, and the tests import without it.

  export TYPESAFE_API_KEY=...
  uv run --with typesafe-sdk label_failure_modes.py --corpus . --limit 20
  uv run label_failure_modes.py --corpus . --dry-run   # no Jev call, no egress
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import rule_checks  # pure stdlib + yaml; lives at the package root

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"

# ── taxonomy (two level) ────────────────────────────────────────────────────────
# family -> {sub_mode: what it means, given to Jev as the Choice criteria}. Jev picks a
# sub-mode; FAMILY_OF derives its family (there is no separate family question). Families
# are failure stages, ordered from understanding the question to running the SQL, chosen
# to be mutually exclusive. Each sub-mode description states its boundary against the
# nearby ones, since overlapping definitions were what made the fine label low-confidence.
# The labels are only as good as these descriptions. This taxonomy names no warehouse.
TAXONOMY: dict[str, dict[str, str]] = {
    "comprehension": {
        "misread_question": "pursued a different question than the one asked, for example "
        "the wrong metric or the wrong subject; not a SQL mistake made while pursuing the "
        "right question",
        "ambiguous_question": "the question itself is underspecified or has no single "
        "correct answer; the candidate is not at fault",
    },
    "entity_resolution": {
        "wrong_identifier": "used the wrong id, key, or literal to identify the entity, "
        "for example an internal account id where an organization id is needed, so the "
        "filter matched nothing or the wrong row",
        "unresolved_lookup": "did not resolve a named entity to its row before querying "
        "facts, for example never looking the name up or skipping the dimension join",
    },
    "schema_choice": {
        "wrong_table": "read a table that does not hold the asked-for concept when another "
        "table does; not a wrong predicate on the right table",
        "wrong_column": "used a real column but misread what it measures, picking the "
        "wrong field for the metric",
    },
    "query_construction": {
        "wrong_join": "joined on the wrong key, or missed a join the answer requires",
        "wrong_grain": "counted or summed at the wrong grain (per row vs per entity vs per "
        "day), inflating or collapsing the result",
        "wrong_aggregation": "wrong aggregate or grouping (for example avg vs sum, or a "
        "missing GROUP BY) while the entities and filters are right",
        "wrong_filter": "the right tables, columns, and entities are used, but a predicate "
        "is wrong or missing (wrong category or comparison, or an absent qualifier); not a "
        "wrong identifier and not a time-window mistake",
        "wrong_time_window": "wrong reference date or time range (point-in-time vs range, "
        "or wrong window bounds)",
    },
    "execution": {
        "clickhouse_error": "the SQL failed to run and returned a ClickHouse error "
        "(syntax, unknown identifier, or an unsupported feature)",
    },
    "process": {
        "stopped_early": "returned a final answer before resolving the question, often "
        "after empty results, when more turns were available",
        "ran_out_of_budget": "hit the turn limit without a usable answer",
        "tool_loop": "repeated near-identical queries without making progress",
    },
    "grading": {
        "correct_but_judged_fail": "the final answer looks correct for the question "
        "despite the fail verdict",
    },
}
FAMILY_OF: dict[str, str] = {sub: fam for fam, subs in TAXONOMY.items() for sub in subs}

NO_RULE = "none"  # no semantic rule explains this failure

QUESTION_IDS = ("sub_mode", "prevented_by_rule", "schema_hint_would_help",
                "correct_despite_fail")


# ── rules (loaded from config) ────────────────────────────────────────────────────
def semantic_rules(rules: dict) -> dict[str, str]:
    """The judgment-call rules offered to Jev (decidable rules are scored in code)."""
    return dict(rules["semantic_rules"])


def prompt_rules(rules: dict) -> dict[str, str]:
    """Every schema-prompt rule, decidable and semantic, keyed to its description. Used
    for display in the report."""
    out = {k: r.get("description", k) for k, r in rules["decidable_rules"].items()}
    out.update(rules["semantic_rules"])
    return out


# ── loading ──────────────────────────────────────────────────────────────────────
def load_ground_truth(annotated_path: Path) -> dict[str, dict]:
    """trace_id -> annotated row (carries gt_sql, gt_answer)."""
    gt: dict[str, dict] = {}
    with annotated_path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gt[row["trace_id"]] = row
    return gt


def iter_failed(results_path: Path, gt: dict[str, dict],
                include_tie: bool = False) -> Iterator[dict]:
    """Yield one small record per failed candidate cell.

    Reads `results.jsonl` line by line (rows are large), keeps candidate cells whose
    judged outcome is `fail` (plus `tie` when asked), and joins ground truth by
    trace_id. `gt_sql` / `gt_answer` are included as the reference the failure is
    judged against; the oracle result set is deliberately left out.
    """
    keep = {"fail", "tie"} if include_tie else {"fail"}
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            g = gt.get(row["trace_id"], {})
            for model, cell in (row.get("candidates") or {}).items():
                cell = cell or {}
                score = cell.get("result_score") or {}
                if score.get("outcome") not in keep:
                    continue
                yield {
                    "trace_id": row["trace_id"],
                    "model": model,
                    "outcome": score.get("outcome"),
                    "nl_question": row.get("nl_question", ""),
                    "gt_sql": g.get("gt_sql") or [],
                    "gt_answer": g.get("gt_answer", ""),
                    "candidate_sqls": cell.get("sqls") or [],
                    "sql_results": cell.get("sql_results") or [],
                    "final_answer": cell.get("final_answer", ""),
                    "turns": cell.get("turns"),
                    "error": cell.get("error"),
                }


def sample_failed(cells: Iterator[dict], n: int, seed: int = 0) -> list[dict]:
    """Reservoir-sample n failed cells from the stream so the sample spans many
    questions, not just the first rows of results.jsonl. Constant memory."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, cell in enumerate(cells):
        if i < n:
            reservoir.append(cell)
        elif (j := rng.randint(0, i)) < n:
            reservoir[j] = cell
    return reservoir


# ── request shape ──────────────────────────────────────────────────────────────
def build_state(cell: dict, max_result_chars: int = 2000,
                checks: dict | None = None) -> dict:
    """Trajectory as Jev state. Result sets are truncated to keep under the 64k cap
    and to avoid sending large oracle-adjacent output. `checks` are the decidable
    rule facts (from rule_checks); passing them lets Jev ground its judgments on what
    the SQL did rather than re-derive it from the raw text."""
    def clip(items: list) -> list:
        out = []
        for it in items:
            s = it if isinstance(it, str) else json.dumps(it, ensure_ascii=False)
            out.append(s[:max_result_chars] + " ...(truncated)" if len(s) > max_result_chars else s)
        return out

    state = {
        "question": cell["nl_question"],
        "reference_sql": cell["gt_sql"],
        "reference_answer": cell["gt_answer"],
        "candidate_sqls": cell["candidate_sqls"],
        "candidate_results": clip(cell["sql_results"]),
        "candidate_final_answer": cell["final_answer"],
        "candidate_turns": cell["turns"],
        "candidate_error": cell["error"],
    }
    if checks is not None:
        state["mechanical_checks"] = checks
    return state


def build_questions(primitives, rules: dict):
    """The batched Jev questions. `primitives` is the (Choice, Noul) pair, passed in
    so this stays importable without the SDK; `rules` carries the semantic rule set.
    Questions are independent and answered in parallel; each carries its full meaning
    because question ids are not sent."""
    Choice, Noul = primitives
    sub_modes = {sub: desc for subs in TAXONOMY.values() for sub, desc in subs.items()}
    rule_criteria = {**semantic_rules(rules), NO_RULE: "no semantic rule explains this failure"}
    return {
        "sub_mode": Choice(
            instructions="Given the question, the reference SQL and answer, and the "
            "candidate's SQL, results, and final answer, pick the single most specific "
            "sub-mode for why the candidate failed. Its family is derived from your "
            "choice, so pick the sub-mode that fits best.",
            criteria=sub_modes),
        "prevented_by_rule": Choice(
            instructions="Which of these judgment-call schema-prompt rules best explains "
            "why the candidate failed? The mechanically checkable rules are scored "
            "separately in code, so they are not listed here; the state field "
            "`mechanical_checks` shows what the SQL did. Choose 'none' if no listed rule "
            "explains the failure.",
            criteria=rule_criteria),
        "schema_hint_would_help": Noul(
            instructions="Would giving the candidate a correct schema-use hint (the "
            "right table, column, join, or grain) before it queried most likely have "
            "prevented this failure?"),
        "correct_despite_fail": Noul(
            instructions="Does the candidate's final answer actually look correct for "
            "the question, despite being scored a failure?"),
    }


def parse_response(resp) -> dict:
    """Read the typed answers off a Jev response into a flat label record.

    Family is derived from the chosen sub-mode, not asked separately, so it can never
    contradict it. Family confidence is the sub-mode Choice's probability mass on that
    family (the sum over the family's sub-modes), a real coarse confidence from the one
    fine question rather than a second question that can disagree."""
    ch, no = resp.choices, resp.nouls
    sub = ch["sub_mode"].choice
    family = FAMILY_OF.get(sub)
    probs = getattr(ch["sub_mode"], "probabilities", None) or {}
    family_confidence = (sum(p for opt, p in probs.items() if FAMILY_OF.get(opt) == family)
                         if probs else None)
    return {
        "sub_mode": sub,
        "sub_mode_confidence": ch["sub_mode"].confidence,
        "family": family,
        "family_confidence": family_confidence,
        "prevented_by_rule": ch["prevented_by_rule"].choice,
        "prevented_by_rule_confidence": ch["prevented_by_rule"].confidence,
        "schema_hint_would_help": no["schema_hint_would_help"].noul,
        "correct_despite_fail": no["correct_despite_fail"].noul,
    }


# ── Jev client ───────────────────────────────────────────────────────────────────
def make_client():
    """Build a Jev client. The key is read from TYPESAFE_API_KEY and never printed;
    TYPESAFE_BASE_URL overrides the host (None → the SDK default, api.typesafe.ai)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("set TYPESAFE_API_KEY in the environment")
    from typesafe_sdk import TypeSafeClient
    return TypeSafeClient(api_key=key, base_url=os.environ.get("TYPESAFE_BASE_URL"))


def label_cell(client, primitives, cell: dict, rules: dict, max_result_chars: int) -> dict:
    checks = rule_checks.check_sql_rules(cell["candidate_sqls"], rules)
    resp = client.system_one(state=build_state(cell, max_result_chars, checks=checks),
                             questions=build_questions(primitives, rules))
    return {"trace_id": cell["trace_id"], "model": cell["model"],
            "outcome": cell["outcome"], **parse_response(resp),
            "checks": checks, "decidable_verdicts": rule_checks.decidable_verdicts(checks)}


# ── reporting ────────────────────────────────────────────────────────────────────
def render_report(labels: list[dict], corpus: str, rules: dict) -> str:
    n = len(labels)
    descriptions = prompt_rules(rules)
    per_model_family: dict[str, Counter] = defaultdict(Counter)
    sub_modes: Counter = Counter()
    semantic_rule_hits: Counter = Counter()  # Jev's causal attribution (judgment-call rules)
    decidable_viol: Counter = Counter()       # rule violations decided in code
    idea_hint: Counter = Counter()            # sub-modes a schema hint would likely fix
    judge_disagreements = 0
    for lab in labels:
        per_model_family[lab["model"]][lab["family"]] += 1
        sub_modes[lab["sub_mode"]] += 1
        if lab["prevented_by_rule"] != NO_RULE:
            semantic_rule_hits[lab["prevented_by_rule"]] += 1
        if lab["schema_hint_would_help"] >= 0.5:
            idea_hint[lab["sub_mode"]] += 1
        if lab["correct_despite_fail"] >= 0.5:
            judge_disagreements += 1
        for rule, verdict in lab.get("decidable_verdicts", {}).items():
            if verdict == rule_checks.VIOLATION:
                decidable_viol[rule] += 1

    out = [f"# Failure-mode labels — {corpus}", "",
           f"Jev-labeled {n} failed candidate cells. Sub-mode is authoritative; family "
           f"is the sub-mode's family.", ""]
    out += ["## Semantic rules Jev judged would have prevented the failure (ranked)", "",
            "The judgment-call rules, attributed by Jev. The decidable rules are scored "
            "in code and listed separately below.", ""]
    for rule, cnt in semantic_rule_hits.most_common():
        out.append(f"- **{rule}** x{cnt} — {descriptions.get(rule, '')}")
    out += ["", f"({n - sum(semantic_rule_hits.values())} failures mapped to no semantic rule.)", ""]
    out += ["## Decidable rule violations in the failed runs (from code)", "",
            "Adherence gaps the checks found in these failures. They co-occur with the "
            "failure and are worth fixing in the prompt, but are not proven to cause it.", ""]
    for rule, cnt in decidable_viol.most_common():
        out.append(f"- **{rule}** x{cnt} — {descriptions.get(rule, '')}")
    if not decidable_viol:
        out.append("- (no decidable violations in this set)")
    out += ["", "## Sub-mode distribution", ""]
    for sub, cnt in sub_modes.most_common():
        out.append(f"- **{sub}** x{cnt} ({FAMILY_OF.get(sub, '?')})")
    out += ["", "## Candidates for an in-loop schema-hint gate", "",
            "Sub-modes Jev judged a schema-use hint would most often have prevented.", ""]
    for sub, cnt in idea_hint.most_common():
        out.append(f"- **{sub}** x{cnt}")
    out += ["", "## Per-model family distribution", ""]
    for model in sorted(per_model_family):
        dist = ", ".join(f"{fam} x{c}" for fam, c in per_model_family[model].most_common())
        out.append(f"- **{model}**: {dist}")
    out += ["", "## Judge disagreement", "",
            f"{judge_disagreements} of {n} failures Jev judged possibly correct despite "
            f"the fail verdict. Review these against ground truth before trusting them.", ""]
    return "\n".join(out) + "\n"


# ── main ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=".",
                    help="corpus dir name under the benchmark data root")
    ap.add_argument("--rules", type=Path, default=None,
                    help="rules config (semantic + decidable). Defaults to DAM_RULES_CONFIG "
                         "or config/rules.example.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="label at most the first N failed cells (quick smoke)")
    ap.add_argument("--sample", type=int, default=None,
                    help="reservoir-sample N failed cells across all questions; more "
                    "representative than --limit, which takes the first N")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--include-tie", action="store_true",
                    help="also label cells scored 'tie', not just 'fail'")
    ap.add_argument("--max-result-chars", type=int, default=2000,
                    help="truncate each candidate result string to this many chars")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "failure_modes.md",
                    help="markdown report path (written outside the corpus)")
    ap.add_argument("--labels", type=Path, default=OUT_DIR / "failure_labels.jsonl",
                    help="per-cell labels JSONL path (written outside the corpus)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load and count failed cells and print one sample state; no Jev call")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    rules = rule_checks.load_rules(args.rules or os.environ.get("DAM_RULES_CONFIG"))

    from paths import DATA  # resolved here so the module imports without a data root
    corpus_dir = DATA / args.corpus
    gt = load_ground_truth(corpus_dir / "annotated.jsonl")
    cells = iter_failed(corpus_dir / "results.jsonl", gt, include_tie=args.include_tie)

    if args.dry_run:
        sample = None
        count = 0
        for count, cell in enumerate(cells, 1):
            if sample is None:
                sample = build_state(cell, args.max_result_chars,
                                     checks=rule_checks.check_sql_rules(cell["candidate_sqls"], rules))
        print(f"{count} failed cells in {args.corpus}")
        if sample is not None:
            print("\nsample state (one cell):")
            print(json.dumps(sample, ensure_ascii=False, indent=2)[:4000])
        return

    from typesafe_sdk import Choice, Noul
    primitives = (Choice, Noul)
    client = make_client()

    if args.sample is not None:
        cells = sample_failed(cells, args.sample, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []
    with client, args.labels.open("w") as lf:
        for i, cell in enumerate(cells):
            if args.limit is not None and i >= args.limit:
                break
            lab = label_cell(client, primitives, cell, rules, args.max_result_chars)
            labels.append(lab)
            lf.write(json.dumps(lab, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"labeled {i + 1}")

    args.out.write_text(render_report(labels, args.corpus, rules))
    print(f"labeled {len(labels)} cells -> {args.labels}\nwrote {args.out}")


if __name__ == "__main__":
    main()
