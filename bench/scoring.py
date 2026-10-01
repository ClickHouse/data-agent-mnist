"""Deterministic result-set comparison and majority-vote ground truth.

Parses result-set strings, links differently named columns through the registry's
linker model, and decides whether two runs retrieved the same data.
"""
import json
import math
import random
import re
import sys
import threading
from typing import NamedTuple

from registry import LINKER, SCORE_ABS_TOL, SCORE_REL_TOL, SCORE_ROUND_DECIMALS

from . import completion

# Result-set numeric agreement policy. Two numbers agree when they are within
# the absolute OR the relative tolerance (math.isclose). Values are first rounded
# to `round_decimals` places, or kept at full precision when it is None.
#
# The board is a currency warehouse, so the default rounds to cents and allows 5%
# relative drift: it catches 59K vs 70K and ignores rounding. That default is
# wrong for a warehouse of concentrations, p-values or dose-response, where a
# wrong answer by a factor of two sits inside 5% and two small values both round
# to 0.00. A non-currency operator overrides these in the `scoring` section of
# the model config (see config/models.example.yaml); the values come from
# registry.py so annotate-time and eval-time share one policy.
class ComparisonPolicy(NamedTuple):
    round_decimals: int | None
    rel_tol: float
    abs_tol: float


SCORING = ComparisonPolicy(SCORE_ROUND_DECIMALS, SCORE_REL_TOL, SCORE_ABS_TOL)

# Back-compat alias: the sensitivity scripts read bench.AGREEMENT_TOL.
AGREEMENT_TOL = SCORING.rel_tol

# ── Trajectory utilities ──────────────────────────────────────────────────────

def is_exploratory(sql: str) -> bool:
    s = (sql or "").strip().upper()
    if re.match(r"(DESCRIBE|DESC)\b", s):
        return True
    if re.match(r"SHOW\b", s):
        return True
    if any(k in s for k in ("SYSTEM.TABLES", "SYSTEM.COLUMNS", "SYSTEM.DATABASES",
                             "INFORMATION_SCHEMA")):
        return True
    m = re.search(r"\bLIMIT\s+(\d+)\b", s)
    if m and int(m.group(1)) <= 10:
        return True
    return False

def _parse_result(result_str, decimals: int | None = SCORING.round_decimals):
    """Parse a result-set string into normalized, sorted rows.

    Numbers are rounded to `decimals` places when it is not None (the board rounds
    to cents); None keeps full precision so small-magnitude values are not
    flattened before comparison. Non-numeric cells are kept as strings.
    """
    if not result_str or result_str.startswith("Error:") or result_str == "(empty result)":
        return []
    rows = []
    for _ln in result_str.strip().splitlines():
        try:
            _row, _norm = json.loads(_ln), {}
            for k, v in _row.items():
                try:    _norm[k.lower().strip()] = round(float(v), decimals) if decimals is not None else float(v)
                except: _norm[k.lower().strip()] = str(v)
            rows.append(_norm)
        except Exception:
            continue
    return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))


_COL_LINK_CACHE: dict = {}   # (tuple(keys_a), tuple(keys_b)) -> {a_col: b_col}
_COL_LINK_LOCK  = threading.Lock()
_COL_LINK_WARNED = False
_COL_LINK_EMPTY_WARNED = False   # parsed-but-empty mapping seen (not a linker error)


def _link_columns(keys_a: list, keys_b: list) -> dict:
    """Map each column in A to the column in B that means the same quantity, so
    aliasing (e.g. total_dollar_usage <-> monthly_spend) doesn't hide a real value
    comparison. Identity when one column set contains the other (no model call);
    otherwise a cached Haiku call (temp 0 -> reproducible). Falls back to shared
    names on error; logs a parsed-but-empty mapping instead of scoring it blind.

    The eval loop calls this from many worker threads. The Haiku call runs OUTSIDE
    the lock so column-linking stays concurrent, but the cache read and write are
    locked and the write is a first-writer-wins `setdefault` -> every thread that
    misses on the same key returns the one stored mapping (temp-0 output isn't
    byte-guaranteed across calls, so last-writer-wins could otherwise hand different
    threads different mappings for the same key)."""
    ka, kb = tuple(keys_a), tuple(keys_b)
    sa, sb = set(ka), set(kb)
    # Identity when one column set contains the other (no model call): the smaller
    # set's columns all exist on the other side, so mapping each to itself scores the
    # SELECT-subset case (candidate returns the ground truth's columns plus extras)
    # on the ground truth's columns alone, instead of sending a wide list to the
    # model and reading its truncated, valid-but-empty reply as "no columns match".
    if sa <= sb or sb <= sa:
        return {k: k for k in (ka if sa <= sb else kb)}
    with _COL_LINK_LOCK:
        if (ka, kb) in _COL_LINK_CACHE:
            return _COL_LINK_CACHE[(ka, kb)]
    prompt = (
        "Two SQL result sets answer the same question but may use different column "
        "names. Map each column in A to the column in B that represents the SAME "
        "quantity (same meaning and unit). Use null when there is no match.\n"
        f"A columns: {list(ka)}\nB columns: {list(kb)}\n"
        'Respond ONLY with JSON: {"mapping": {"<a_col>": "<b_col or null>"}}'
    )
    try:
        # Generous budget: a truncated reply parses as valid-but-empty JSON.
        text = completion._judge_complete(LINKER, prompt, max_tokens=1024, temperature=0)
        m   = re.search(r"\{.*\}", text, re.DOTALL)
        raw = json.loads(m.group()).get("mapping", {}) if m else {}
        mapping = {a: b for a, b in raw.items() if a in sa and b in sb}
        if not mapping:
            # Still a mismatch (no shared meaning), but say so once: on very wide
            # non-subset sets an empty mapping can be a truncation artifact.
            global _COL_LINK_EMPTY_WARNED
            if not _COL_LINK_EMPTY_WARNED:
                _COL_LINK_EMPTY_WARNED = True
                print(f"WARNING: column linker ({LINKER}) returned an empty mapping "
                      f"for {len(ka)}x{len(kb)} columns; scoring as a mismatch.",
                      file=sys.stderr)
    except Exception as e:
        # Loudly. The fallback matches identical names only, so aliased columns
        # stop linking and equivalent answers score WRONG. That is a silent change
        # to the scoring rule, and it used to happen to anyone without an AWS
        # account, because this call was hardcoded to Bedrock. Once per
        # process: it fires per column pair and the run should stay readable.
        global _COL_LINK_WARNED
        if not _COL_LINK_WARNED:
            _COL_LINK_WARNED = True
            print(f"WARNING: column linker ({LINKER}) failed: {e!r}\n"
                  f"  Falling back to exact column-name matching for the whole run. "
                  f"Answers that are correct but name their columns differently "
                  f"will score as failures. Set judges.linker in the model config "
                  f"to a model you can reach.", file=sys.stderr)
        mapping = {k: k for k in ka if k in set(kb)}   # fallback: shared names only
    with _COL_LINK_LOCK:
        return _COL_LINK_CACHE.setdefault((ka, kb), mapping)


def _results_match(res_a: str, res_b: str, policy: ComparisonPolicy = SCORING) -> bool:
    """Whether two result-set strings are equivalent under `policy` (rounding-safe).
    Columns are entity-linked first so differently-aliased value columns are still
    compared by value, not silently skipped. Two numbers agree when they are within
    the policy's absolute OR relative tolerance; the currency default is 5% relative."""
    a, b = _parse_result(res_a, policy.round_decimals), _parse_result(res_b, policy.round_decimals)
    if not a and not b:
        return True
    if not a or not b or len(a) != len(b):
        return False
    mapping = _link_columns(sorted({k for r in a for k in r}),
                            sorted({k for r in b for k in r}))
    if not mapping:
        return False
    # Project BOTH sides onto the linked columns (in a's namespace) so they share a
    # key set -> identical sort order -> rows align. Extra/unmapped columns (e.g. a
    # candidate that also returned id/mrr/tier) are ignored, not compared.
    a = [{ac: r[ac] for ac in mapping if ac in r} for r in a]
    b = [{ac: r[bc] for ac, bc in mapping.items() if bc in r} for r in b]
    _key = lambda r: json.dumps(r, sort_keys=True, default=str)
    for ra, rb in zip(sorted(a, key=_key), sorted(b, key=_key)):
        common = set(ra.keys()) & set(rb.keys())
        if not common:
            return False
        for k in common:
            x, y = ra[k], rb[k]
            try:
                fx, fy = float(x), float(y)
                if not math.isclose(fx, fy, rel_tol=policy.rel_tol, abs_tol=policy.abs_tol):
                    return False
            except (ValueError, TypeError):
                if str(x) != str(y):
                    return False
    return True


def annotators_agree(results_a: list, results_b: list) -> bool:
    """Two annotators agree if any of their non-empty result-set strings match.
    Both-empty does NOT count: a question no annotator can answer has no usable
    ground truth and is excluded, not asserted as 'no rows'."""
    a = [r for r in results_a if r and not r.startswith("Error:") and r != "(empty result)"]
    b = [r for r in results_b if r and not r.startswith("Error:") and r != "(empty result)"]
    if not a or not b:
        return False
    return any(_results_match(x, y) for x in a for y in b)


def majority_vote_gt(annotator_runs: dict) -> dict:
    """Build ground truth from independent annotator runs.

    `annotator_runs` is {annotator_name: run_candidate(...) result dict}. Annotators
    are clustered by result-set agreement; if a cluster of >=2 exists, its first
    member's run becomes the ground truth. If every annotator disagrees, the
    question is excluded. Replaces the v1 single-Opus stability filter and the
    manual exclusion list."""
    names = list(annotator_runs)
    res   = {n: annotator_runs[n].get("sql_results", []) for n in names}

    clusters: list[list[str]] = []
    for n in names:
        for cl in clusters:
            if annotators_agree(res[n], res[cl[0]]):
                cl.append(n)
                break
        else:
            clusters.append([n])
    clusters.sort(key=len, reverse=True)

    top = clusters[0] if clusters else []
    if len(top) >= 2:
        # Pick the representative agreer at random so the stored GT bytes don't skew
        # to one annotator's formatting. Deterministic in the agreed content (seeded
        # by agreer names + their result sets) -> reproducible across re-runs.
        _key   = "|".join(sorted(top)) + "::" + "|".join("".join(res[n]) for n in sorted(top))
        winner = random.Random(_key).choice(sorted(top))
        run    = annotator_runs[winner]
        return {"excluded": False, "agreers": top, "gt_from": winner,
                "gt_results": run.get("sql_results", []),
                "gt_answer":  run.get("final_answer", ""),
                "gt_sql":     run.get("sqls", []),
                "clusters":   clusters}
    return {"excluded": True, "agreers": [], "gt_from": None,
            "gt_results": [], "gt_answer": "", "gt_sql": [],
            "clusters": clusters, "reason": "annotators fully disagree"}


def bt_sigma_aggregate(*args, **kwargs):
    """Bradley-Terry + per-judge reliability (sigma) jury aggregation
    (arXiv 2602.16610). Deferred: only worthwhile once the benchmark
    scales past ~71 questions. Use majority_vote_gt / judge_panel until then."""
    raise NotImplementedError(
        "BT-sigma jury deferred until the benchmark scales past ~71 questions"
    )
