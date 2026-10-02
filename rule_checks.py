"""Decidable schema-prompt rule checks over candidate SQL.

Several rules in a warehouse's schema prompt are facts about the SQL, not judgment
calls, so code decides them and no model question is spent. `check_sql_rules` returns
those facts; `decidable_verdicts` maps each decidable rule to ok/violation/n/a.

The facts feed two things: they go into Jev's state so its semantic judgments (was a
missing filter actually required, would a schema hint help) are grounded in what the
SQL did rather than re-derived from raw text, and the verdicts are a free oracle to
validate Jev against on those rules.

The rule set is data, not code: it is loaded from a rules config (DAM_RULES_CONFIG,
defaulting to config/rules.example.yaml beside this module), so this engine names no
warehouse's tables or columns. The example targets examples/saas; point
DAM_RULES_CONFIG (or pass --rules) at your own. Each decidable rule is one of two
kinds:

  * forbidden_when: VIOLATION if `forbidden_regex` appears; OK if `ok_regex` appears
    (when given); otherwise n/a. (A token the query must not use, e.g. now().)
  * required_when: applies when `applies_regex` appears; then OK if `required_regex`
    appears and VIOLATION otherwise; n/a when it does not apply. `scope: per_statement`
    (default) decides each statement on its own, so the required token must sit in the
    same statement as the trigger; `scope: global` is satisfied by the token appearing
    in any statement (e.g. a probe query that precedes the real one).

These are regex heuristics over query text, not a SQL parser: they read tokens, so an
unusual alias or a token inside a string literal can fool them. That is acceptable for
an aggregate signal and for grounding a judgment; it is not a correctness gate.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

OK, VIOLATION, NA = "ok", "violation", "n/a"

CONFIG_PATH = Path(os.environ.get(
    "DAM_RULES_CONFIG", Path(__file__).resolve().parent / "config/rules.example.yaml"))

_KINDS = ("forbidden_when", "required_when")


def load_rules(path: Path | None = None) -> dict:
    """Load and validate a rules config, compiling each rule's regexes once.

    A missing file is a hard error: a labeler run with no rule set does not fail, it
    labels strictly worse (no decidable grounding, no semantic rule attribution) while
    reporting success, which is the failure this indirection exists to prevent.
    """
    p = Path(path or CONFIG_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"rule set not found at {p}. Copy config/rules.example.yaml and edit it for "
            f"your warehouse, or point DAM_RULES_CONFIG (or --rules) at your own.")
    cfg = yaml.safe_load(p.read_text()) or {}
    semantic = cfg.get("semantic_rules") or {}
    decidable = cfg.get("decidable_rules") or {}
    if not isinstance(semantic, dict) or not isinstance(decidable, dict):
        raise ValueError(f"{p}: 'semantic_rules' and 'decidable_rules' must be mappings")
    overlap = set(semantic) & set(decidable)
    if overlap:
        raise ValueError(f"{p}: a rule is in both groups, so it would be scored twice "
                         f"and offered to Jev at once: {sorted(overlap)}")
    for key, rule in decidable.items():
        kind = rule.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"{p}: rule '{key}' has kind {kind!r}, not one of {_KINDS}")
        if kind == "forbidden_when":
            rule["_forbidden"] = re.compile(rule["forbidden_regex"], re.I)
            rule["_ok"] = re.compile(rule["ok_regex"], re.I) if rule.get("ok_regex") else None
        else:
            rule["_applies"] = re.compile(rule["applies_regex"], re.I)
            rule["_required"] = re.compile(rule["required_regex"], re.I)
            if rule.get("scope", "per_statement") not in ("per_statement", "global"):
                raise ValueError(f"{p}: rule '{key}' scope must be per_statement or global")
    return {"semantic_rules": semantic, "decidable_rules": decidable}


def decidable_rule_keys(rules: dict) -> tuple[str, ...]:
    """The decidable rule keys, in config order. The labeler splits rules on these, so
    whatever is here is scored in code and never offered to Jev."""
    return tuple(rules["decidable_rules"])


def _verdict(rule: dict, sqls: list[str], joined: str) -> str:
    if rule["kind"] == "forbidden_when":
        if rule["_forbidden"].search(joined):
            return VIOLATION
        ok = rule["_ok"]
        if ok is not None:
            return OK if ok.search(joined) else NA
        return NA
    applies, required = rule["_applies"], rule["_required"]
    if not any(applies.search(s) for s in sqls):
        return NA
    if rule.get("scope", "per_statement") == "global":
        return OK if any(required.search(s) for s in sqls) else VIOLATION
    # per_statement: a trigger statement that lacks the required token is a violation,
    # even if a different statement carries it.
    return VIOLATION if any(applies.search(s) and not required.search(s) for s in sqls) else OK


def check_sql_rules(sqls: list[str], rules: dict) -> dict:
    """Facts about a candidate's SQLs: a verdict per decidable rule plus two generic,
    schema-independent counts. Booleans/verdicts only; no judgment."""
    sqls = [s or "" for s in sqls]
    joined = "\n".join(sqls)
    norm = [re.sub(r"\s+", " ", s.strip().lower()) for s in sqls if s.strip()]
    return {
        "decidable": {key: _verdict(rule, sqls, joined)
                      for key, rule in rules["decidable_rules"].items()},
        "n_queries": len(norm),
        "has_repeated_query": len(norm) != len(set(norm)),
    }


def decidable_verdicts(facts: dict) -> dict[str, str]:
    """Pull the ok/violation/n/a verdict for each decidable rule out of the facts.
    `n/a` means the rule did not apply to the query, distinct from complying with it."""
    return dict(facts["decidable"])
