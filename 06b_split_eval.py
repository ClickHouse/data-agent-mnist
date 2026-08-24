"""
Split eval: score candidate(s) against the correct chDB epoch per question (AI-1474).

The v2.2 board is a two-epoch patchwork (see board_manifest.json / verify_board.py):
most questions were scored against the primary DB `chdb`, and the questions
re-annotated during the disposition/CRM pass were scored against `chdb-disp`. A
new model must run each question against the DB its ground truth was computed on,
or it is graded against a warehouse it never saw (which is what gave Fable a bogus
14.5%). This orchestrator partitions the questions by the manifest and runs
06_eval.py once per DB, merging into a single results.jsonl.

    uv run 06b_split_eval.py --models fable5
    uv run 06b_split_eval.py --models fable5,kimi-k3 --workers 8

Each 06_eval.py invocation runs its own board guard (verify_subset), so a partition
that no longer reproduces its DB aborts before spending model calls.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from warehouse import DATA_DIR

HERE          = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "board_manifest.json"
ANNOT_PATH    = DATA_DIR / "annotated.jsonl"
RESULTS_PATH  = DATA_DIR / "results.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated candidate keys")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--annot", type=Path, default=ANNOT_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap NEW questions per DB partition (debug/smoke)")
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(f"{MANIFEST_PATH} missing — run `uv run verify_board.py --emit` first")
    manifest = json.loads(MANIFEST_PATH.read_text())

    # Group the annotated trace_ids by their manifest-assigned DB. Route STRICTLY by
    # the manifest — never fall back to a default DB, or a question missing from the
    # manifest (a stale manifest, or an AI-997-excluded question) would be graded
    # against an unverified DB and slip past 06_eval's guard (which only checks the
    # traces it scores against their manifest assignment). Unrouted trace_ids are
    # reported and left out; re-emit the manifest if a graded question is among them.
    annot_lines = {json.loads(l)["trace_id"]: l
                   for l in args.annot.read_text().splitlines() if l.strip()}
    by_db = defaultdict(list)
    unrouted = []
    for tid, line in annot_lines.items():
        entry = manifest["partition"].get(tid)
        if entry is None:
            unrouted.append(tid)
            continue
        by_db[entry["db"]].append(line)

    print(f"split eval '{args.models}' over {sum(len(v) for v in by_db.values())} "
          f"manifest-routed questions: "
          + ", ".join(f"{db}={len(v)}" for db, v in sorted(by_db.items())))
    if unrouted:
        print(f"note: {len(unrouted)} annotated question(s) not in the manifest — left out "
              f"(excluded questions, or re-emit the manifest if they should be graded): "
              f"{', '.join(t[:8] for t in unrouted[:8])}{' ...' if len(unrouted) > 8 else ''}")

    tmp = Path(tempfile.mkdtemp(prefix="split_eval_"))
    try:
        for db in sorted(by_db):
            db_path = DATA_DIR / db
            if not db_path.exists():
                sys.exit(f"assigned DB '{db}' not found at {db_path}")
            # filtered annotated.jsonl for this partition (byte-identical GT lines)
            sub_annot = tmp / f"annotated_{db}.jsonl"
            sub_annot.write_text("\n".join(by_db[db]) + "\n")
            cmd = [
                "uv", "run", "06_eval.py",
                "--models", args.models,
                "--annot", str(sub_annot),
                "--db-path", str(db_path),
                "--out", str(args.out),
                "--workers", str(args.workers),
            ]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            print(f"\n=== partition '{db}' ({len(by_db[db])} questions) ===\n{' '.join(cmd)}")
            r = subprocess.run(cmd, cwd=HERE)
            if r.returncode != 0:
                sys.exit(f"partition '{db}' failed (exit {r.returncode}) — stopping; "
                         f"results.jsonl holds whatever merged before the failure")
    finally:
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()

    print(f"\nsplit eval complete — merged into {args.out}")


if __name__ == "__main__":
    main()
