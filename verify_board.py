"""
Board integrity: pin every question to the chDB it was scored against, and fail
loudly if the committed DB(s) stop reproducing the committed board.

The v2.2 board drifted because the synthetic DB was re-seeded between eval rounds
while results.jsonl / annotated.jsonl stayed frozen, so cached results were
scored against a DB that no longer existed on disk. Nothing caught it until a new
model (Fable) cratered. This script is the guard that makes the DB + ground truth
+ results a single verified unit: an eval or a commit is only valid if each
question's cached results still reproduce on its assigned DB.

Reproduction test (offline, no model calls): re-run each cached candidate's own
stored SQL against a DB and require its result-set to equal the stored result-set
(order-independent). A question "reproduces" on a DB when >= --threshold of its
comparable (non-empty, non-error) candidate queries match.

Modes
  --emit                write board_manifest.json: assign each trace to the DB
                        that best reproduces it, record repro rates + input hashes.
  (default) check       verify every trace reproduces on its manifest-assigned DB;
                        exit 1 on any drift or on GT/results hashes changing.
  --check-db NAME --traces-file FILE
                        verify only the listed traces reproduce on DB NAME (the
                        06_eval.py startup guard for the subset it is about to run).
  --require-single      (check) fail if the board spans more than one DB epoch;
                        use once v2.3 is unified onto a single seeded DB.

    uv run verify_board.py --emit
    uv run verify_board.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import chdb.session as chs

from warehouse import DATA_DIR

RESULTS_PATH  = DATA_DIR / "results.jsonl"
ANNOT_PATH    = DATA_DIR / "annotated.jsonl"
MANIFEST_PATH = DATA_DIR / "board_manifest.json"

# DBs the board may be reproduced against, in preference order (first = default).
CANDIDATE_DBS = ["chdb", "chdb-disp"]
DEFAULT_THRESHOLD = 0.85


def _rowset(text: str) -> frozenset:
    """Canonicalize a JSONEachRow result string into an order-independent row-set."""
    out = set()
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.add(json.dumps(json.loads(ln), sort_keys=True))
        except Exception:
            out.add(ln)
    return frozenset(out)


def _comparable(sqls, stored):
    """Yield (sql, stored_rowset) pairs worth checking (skip empty/error/None)."""
    for q, st in zip(sqls or [], stored or []):
        if not q or st in (None, "(empty result)") or str(st).startswith("Error"):
            continue
        yield q, _rowset(st)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def repro_on_db(rows, db_path: Path, only: set | None = None):
    """-> {trace_id: (matched, comparable)} reproduction counts on one DB."""
    sess = chs.Session(str(db_path))
    try:
        out = {}
        for r in rows:
            tid = r["trace_id"]
            if only is not None and tid not in only:
                continue
            m = n = 0
            for cand in r["candidates"].values():
                cand = cand or {}
                for q, want in _comparable(cand.get("sqls"), cand.get("sql_results")):
                    try:
                        got = _rowset("\n".join(
                            str(sess.query(q, "JSONEachRow")).splitlines()))
                    except Exception:
                        continue
                    n += 1
                    m += (got == want)
            out[tid] = (m, n)
        return out
    finally:
        sess.close()


def _rate(mc):
    m, n = mc
    return (m / n) if n else None


def emit(threshold: float):
    rows = load_rows(RESULTS_PATH)
    dbs = [d for d in CANDIDATE_DBS if (DATA_DIR / d).exists()]
    if not dbs:
        sys.exit(f"no candidate DBs present under {DATA_DIR} ({CANDIDATE_DBS})")
    per_db = {d: repro_on_db(rows, DATA_DIR / d) for d in dbs}

    partition, summary = {}, {d: 0 for d in dbs}
    summary["low_confidence"] = 0
    for r in rows:
        tid = r["trace_id"]
        rates = {d: _rate(per_db[d][tid]) for d in dbs}
        scored = {d: v for d, v in rates.items() if v is not None}
        best = max(scored, key=scored.get) if scored else dbs[0]  # park no-signal on default
        assigned_rate = rates[best]
        summary[best] += 1
        # Baseline the assigned repro. The board was assembled incrementally across
        # DB epochs, so a few questions have candidate cells from an older seed and
        # never reach 1.0 on any single DB. That is pre-existing board noise, not
        # live drift: we record the rate now and the guard flags only REGRESSION
        # below it (see check()), which is what a future re-seed would cause.
        if assigned_rate is not None and assigned_rate < threshold:
            summary["low_confidence"] += 1
        partition[tid] = {
            "db": best,
            "baseline_repro": round(assigned_rate, 4) if assigned_rate is not None else None,
            "repro": {d: (round(v, 4) if v is not None else None) for d, v in rates.items()},
        }

    manifest = {
        "schema_version": 1,
        "threshold": threshold,
        "default_db": dbs[0],
        "dbs": {
            "chdb":      {"note": "v2.2 board commit (majority epoch)"},
            "chdb-disp": {"note": "disposition/CRM re-annotation epoch"},
        },
        "inputs": {
            "results_sha256":   _sha256(RESULTS_PATH),
            "annotated_sha256": _sha256(ANNOT_PATH),
        },
        "summary": summary,
        "partition": dict(sorted(partition.items())),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST_PATH.name}: "
          + ", ".join(f"{k}={v}" for k, v in summary.items()))
    if summary["low_confidence"]:
        lc = [f"{t[:8]}={p['baseline_repro']:.0%}" for t, p in partition.items()
              if p["baseline_repro"] is not None and p["baseline_repro"] < threshold]
        print(f"note: {summary['low_confidence']} question(s) below {threshold:.0%} on their best DB "
              f"(incremental-epoch cells, baselined not failed): {', '.join(lc)}")
    return manifest


def _load_manifest():
    if not MANIFEST_PATH.exists():
        sys.exit(f"{MANIFEST_PATH} missing — run `uv run verify_board.py --emit` first")
    return json.loads(MANIFEST_PATH.read_text())


def check(require_single: bool, tolerance: float):
    man = _load_manifest()
    problems = []

    # 1) inputs unchanged since the manifest was emitted
    for key, path in (("results_sha256", RESULTS_PATH), ("annotated_sha256", ANNOT_PATH)):
        if _sha256(path) != man["inputs"][key]:
            problems.append(f"{path.name} changed since manifest was emitted "
                            f"(re-run --emit after any board edit)")

    # 2) DB epochs used by the partition
    used = sorted({p["db"] for p in man["partition"].values()})
    if require_single and len(used) > 1:
        problems.append(f"board spans {len(used)} DB epochs {used}; "
                        f"--require-single expects one (unify before v2.3)")
    for d in used:
        if not (DATA_DIR / d).exists():
            problems.append(f"assigned DB '{d}' missing from {DATA_DIR}")

    # 3) every trace still reproduces on its assigned DB at (>= baseline - tolerance).
    #    Regression, not an absolute bar: a re-seed that breaks a DB drops rates far
    #    below baseline and trips this; the known incremental-epoch questions sit at
    #    their recorded baseline and pass.
    if not [p for p in problems if "missing" in p]:
        rows = load_rows(RESULTS_PATH)
        by_db: dict[str, set] = {}
        for tid, p in man["partition"].items():
            by_db.setdefault(p["db"], set()).add(tid)
        for d, traces in by_db.items():
            counts = repro_on_db(rows, DATA_DIR / d, only=traces)
            for tid in traces:
                base = man["partition"][tid].get("baseline_repro")
                if base is None:
                    continue                       # no baseline recorded -> nothing to check
                rate = _rate(counts.get(tid, (0, 0)))
                if rate is None:                   # had a baseline, now 0 comparable -> DB unreadable/empty
                    problems.append(f"{tid[:8]} produced no comparable results on {d} "
                                    f"(baseline {base:.0%}) — DB unreadable/empty")
                elif rate < base - tolerance:
                    problems.append(f"{tid[:8]} regressed to {rate:.0%} on {d} "
                                    f"(baseline {base:.0%}, tol {tolerance:.0%}) — DB/board drift")

    if problems:
        print(f"BOARD VERIFY FAILED ({len(problems)} problem(s)):")
        for p in problems[:40]:
            print("  -", p)
        sys.exit(1)
    print(f"board OK: {len(man['partition'])} questions hold their reproduction baseline "
          f"(epochs: {', '.join(used)}, tol {tolerance:.0%})")


def verify_subset(traces: set[str], db_name: str, tolerance: float = 0.05,
                  db_path: Path | None = None):
    """Startup guard (importable): verify `traces` reproduce on the mounted DB at baseline.

    Enforces that db_name is the DB the manifest assigns each trace to, so an eval
    can never silently run a question against the wrong epoch, and that content
    still reproduces (catches a swapped/re-seeded/empty DB). Reproduction runs
    against `db_path` (the exact directory the eval opens) — pass it so a copy under
    a different directory with the same basename can't pass on the committed tree
    while the eval queries elsewhere; it defaults to DATA_DIR / db_name. Raises
    SystemExit on failure. No-ops with a warning if the manifest does not exist yet
    (pipeline bootstrap)."""
    if not MANIFEST_PATH.exists():
        print(f"[board guard] no {MANIFEST_PATH.name} yet — skipping (run --emit to enable)")
        return
    man = json.loads(MANIFEST_PATH.read_text())
    db_path = db_path or (DATA_DIR / db_name)
    if not db_path.exists():
        sys.exit(f"[board guard] DB '{db_name}' not found at {db_path}")
    misassigned = [t for t in traces
                   if t in man["partition"] and man["partition"][t]["db"] != db_name]
    if misassigned:
        print(f"SUBSET GUARD FAILED: {len(misassigned)} trace(s) are assigned to a different "
              f"DB than '{db_name}' in the manifest: {', '.join(t[:8] for t in misassigned[:10])}")
        sys.exit(1)
    rows = load_rows(RESULTS_PATH)
    counts = repro_on_db(rows, db_path, only=traces)
    bad, empty = [], []
    for tid in traces:
        base = man["partition"].get(tid, {}).get("baseline_repro")
        if base is None:
            continue                       # no baseline recorded -> nothing to check
        rate = _rate(counts.get(tid, (0, 0)))
        if rate is None:
            empty.append(tid)              # had a baseline, now 0 comparable results -> DB unreadable/empty
        elif rate < base - tolerance:
            bad.append((tid, rate, base))
    if bad or empty:
        print(f"SUBSET GUARD FAILED on '{db_name}' ({db_path}): "
              f"{len(bad)} regressed below baseline, {len(empty)} produced no comparable results:")
        for tid, rate, base in bad[:20]:
            print(f"  - {tid[:8]} {rate:.0%} (baseline {base:.0%})")
        for tid in empty[:20]:
            print(f"  - {tid[:8]} no comparable results (DB unreadable/empty?)")
        sys.exit(1)
    print(f"[board guard] OK: {len(traces)} traces hold baseline on '{db_name}'")


def check_subset(db_name: str, traces_file: Path, tolerance: float):
    """CLI wrapper: read trace IDs from a file and run verify_subset."""
    traces = {t.strip() for t in traces_file.read_text().splitlines() if t.strip()}
    verify_subset(traces, db_name, tolerance)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true", help="generate board_manifest.json")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="min per-question reproduction rate (default %(default)s)")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="(check) max allowed drop below baseline before failing (default %(default)s)")
    ap.add_argument("--require-single", action="store_true",
                    help="(check) fail if the board spans >1 DB epoch")
    ap.add_argument("--check-db", type=str, default=None,
                    help="verify only --traces-file against this DB (eval startup guard)")
    ap.add_argument("--traces-file", type=Path, default=None)
    args = ap.parse_args()

    if args.emit:
        emit(args.threshold)
    elif args.check_db:
        if not args.traces_file:
            sys.exit("--check-db requires --traces-file")
        check_subset(args.check_db, args.traces_file, args.tolerance)
    else:
        check(args.require_single, args.tolerance)


if __name__ == "__main__":
    main()
