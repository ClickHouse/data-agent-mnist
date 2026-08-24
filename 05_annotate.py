"""
Build majority-vote ground truth (AI-997) for the synthetic text2sql benchmark.

Each annotator in bench.ANNOTATORS (one per provider — Opus 4.7, GPT-4.1, Gemini
2.5 Pro) runs the agentic tool loop independently against the synthetic DWH. The
result set that >=2 annotators agree on becomes ground truth; questions where all
annotators disagree are excluded. This replaces the v1 single-Opus stability
filter and the manual exclusion list.

    uv run 05_annotate.py [--limit N] [--force] [--no-classify]

Writes data/benchmarks/text2sqlbench-synthetic/annotated.jsonl (resume-safe:
re-run to fill in only the questions not already annotated).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
import obs
from warehouse import DATA_DIR, Warehouse

ANNOT_PATH           = DATA_DIR / "annotated.jsonl"
SCHEMA_COMPAT_PROMPT = Path(__file__).resolve().parent / "prompts" / "schema_compat.md"
CLASSIFIER_MODEL     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Questions the schema-compat classifier wrongly rejects but which are answerable
# with the synthetic schema (was nb05 cell 12). The manual *exclusion* list (cell
# 11) is gone — its job is now done by the all-annotators-disagree filter (AI-997).
MANUAL_ADDITIONS = {
    "1da35f0750f2dbb864a8145396c3af45": "GCP estimated cost = dollar_usage filtered by cloud_provider='gcp'",
    "7391776c7c5991f2f46e59f72e8d174d": "instances created per day in production = services_history",
}


def _flag(v) -> bool:
    # robust truthiness for LLM JSON flags — the string "false" is NOT True
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"))


def load_questions(classify: bool = True, path: Path | None = None) -> list[dict]:
    """Curated questions, optionally filtered to those answerable with the
    synthetic schema (Haiku classifier, nb05 cell 10), plus manual additions."""
    # No board default here. The corpus this used to point at is ours, and naming
    # it in a public module both leaks the dataset name and hands an adopter a
    # path that cannot exist. DAM_QUESTIONS carries it for our own runs (AI-1858).
    src = path or (Path(os.environ["DAM_QUESTIONS"])
                   if os.environ.get("DAM_QUESTIONS") else None)
    if src is None:
        raise SystemExit(
            "no question set. Pass --questions, or set DAM_QUESTIONS.\n"
            "  example/questions.jsonl is a worked set.")
    if not src.exists():
        raise SystemExit(f"no question set at {src}.")
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if not classify:
        return rows

    prompt = SCHEMA_COMPAT_PROMPT.read_text()
    # Batch the classifier: one call for the whole set overflows maxTokens once the
    # curated set grows (the response JSON truncates and won't parse). Indices are
    # local to each chunk, offset back to the global row position.
    compatible: set[int] = set()
    BATCH = 50
    for start in range(0, len(rows), BATCH):
        chunk  = rows[start:start + BATCH]
        q_list = "\n".join(f"{i}. {q['nl_question']}" for i, q in enumerate(chunk))
        resp = bench.bedrock.converse(
            modelId=CLASSIFIER_MODEL,
            system=[{"text": prompt}],
            messages=[{"role": "user", "content": [{"text": q_list}]}],
            inferenceConfig={"maxTokens": 4096, "temperature": 0},
        )
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            parsed = json.loads(m.group()) if m else None
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            # Don't silently drop a whole batch on an unparseable classifier response —
            # keep these questions (annotation's all-disagree filter is the backstop).
            print(f"  ⚠ schema-compat: unparseable response for batch {start}-{start + len(chunk)}; keeping {len(chunk)}")
            compatible.update(range(start, start + len(chunk)))
            continue
        covered = set()
        for r in parsed.get("results", []):
            i = r.get("index")
            if isinstance(i, int) and 0 <= i < len(chunk):
                covered.add(i)
                if _flag(r.get("compatible")):
                    compatible.add(start + i)
        # Indices the parseable response omitted are ambiguous — keep them (same "keep on
        # ambiguity" rule as an unparseable batch) rather than silently dropping.
        missing = [i for i in range(len(chunk)) if i not in covered]
        if missing:
            print(f"  ⚠ schema-compat: response omitted {len(missing)}/{len(chunk)} indices in batch {start}; keeping them")
            compatible.update(start + i for i in missing)
    filtered   = [q for i, q in enumerate(rows) if i in compatible]

    by_id   = {q["trace_id"]: q for q in rows}
    present = {q["trace_id"] for q in filtered}
    added   = 0
    for tid in MANUAL_ADDITIONS:
        if tid in by_id and tid not in present:
            filtered.append(by_id[tid])
            added += 1
    print(f"Schema-compatible: {len(filtered) - added} / {len(rows)}  (+{added} manual additions)")
    return filtered


def annotate_one(wh: Warehouse, nl_question: str, langfuse=None) -> dict:
    """Run every annotator, then majority-vote the ground truth."""
    langfuse = langfuse or obs._NoopLangfuse()
    runs = {}
    for name, model_id in bench.ANNOTATORS.items():
        with langfuse.start_as_current_observation(
            name=f"annotator:{name}", as_type="span",
            input={"question": nl_question, "annotator": name},
        ) as span:
            r = bench.run_candidate(nl_question, name, model_id, wh.query, wh.system_prompt())
            runs[name] = r
            span.update(output={"sql_results": r.get("sql_results"), "final_answer": r.get("final_answer"),
                                "turns": r.get("turns"), "error": r.get("error")})
    gt = bench.majority_vote_gt(runs)
    return {
        "annotators": {
            name: {k: r.get(k) for k in ("sqls", "sql_results", "final_answer", "turns", "latency", "error")}
            for name, r in runs.items()
        },
        "excluded":   gt["excluded"],
        "agreers":    gt["agreers"],
        "gt_from":    gt["gt_from"],
        "gt_sql":     gt["gt_sql"],
        "gt_results": gt["gt_results"],
        "gt_answer":  gt["gt_answer"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="annotate at most N new questions")
    ap.add_argument("--force", action="store_true", help="re-annotate questions already present")
    ap.add_argument("--no-classify", action="store_true", help="skip the schema-compat filter")
    ap.add_argument("--out", type=Path, default=ANNOT_PATH, help="output annotated.jsonl path")
    ap.add_argument("--db-path", type=Path, default=None,
                    help="warehouse to run against (default: the board DB)")
    ap.add_argument("--questions", type=Path, default=None,
                    help="question set to annotate (default: the curated corpus)")
    ap.add_argument("--probe-table", default=None,
                    help="fact table used to check the DB is populated and to "
                         "derive the snapshot date (default: the board's)")
    ap.add_argument("--snapshot-column", default=None,
                    help="date column in --probe-table (default: the board's)")
    ap.add_argument("--system-prompt", type=Path, default=None,
                    help="schema context for that warehouse (default: prompts/system.md)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent questions in flight (1 = sequential)")
    args = ap.parse_args()

    out_path = args.out
    _wh_kw = {k: v for k, v in (('db_path', args.db_path),
                                ('system_prompt_path', args.system_prompt),
                                ('probe_table', args.probe_table),
                                ('snapshot_column', args.snapshot_column))
              if v is not None}
    wh = Warehouse(**_wh_kw)
    print(f"DB ready: {wh.n_rows:,} rows, snapshot {wh.snapshot_date}")
    langfuse = obs.get_langfuse()
    questions = load_questions(classify=not args.no_classify,
                               path=args.questions)

    # Always load existing rows so save() preserves them; --force re-annotates in
    # place (rather than starting from an empty file, which --force + --limit would
    # otherwise truncate to just the re-annotated rows).
    existing: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["trace_id"]] = r

    todo = [q for q in questions if args.force or q["trace_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"To annotate: {len(todo)}  (already present: {len(existing)})")

    def save():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in existing.values()) + "\n")

    def annotate_question(q: dict) -> dict:
        """Annotate one question (3 annotators + majority vote) in a worker thread.
        Opens its own Langfuse span so the nested annotator spans stay within this
        thread's context — the trace tree is intact per question."""
        nlq = q["nl_question"]
        with langfuse.start_as_current_observation(
            name="synthetic-annotation", as_type="span",
            input={"question": nlq, "trace_id": q["trace_id"]},
        ) as span:
            ann = annotate_one(wh, nlq, langfuse)
            span.update(output={"excluded": ann["excluded"], "agreers": ann["agreers"],
                                "gt_from": ann["gt_from"]})
        return ann

    SAVE_EVERY = 5
    completed  = 0
    try:
        for q, ann, exc in bench.map_concurrent(annotate_question, todo, args.workers):
            completed += 1
            nlq = q["nl_question"]
            if exc is not None:
                print(f"[{completed}/{len(todo)}] ERROR {nlq[:56]}: {exc}")
                continue
            existing[q["trace_id"]] = {**q, "trace_id": q["trace_id"], "nl_question": nlq, **ann}
            if ann["excluded"]:
                print(f"[{completed}/{len(todo)}] EXCLUDED       {nlq[:56]}")
            else:
                print(f"[{completed}/{len(todo)}] GT<-{ann['gt_from']:<13} {nlq[:52]}")
            if completed % SAVE_EVERY == 0:
                save()
    finally:
        save()

    langfuse.flush()
    n_excl = sum(1 for r in existing.values() if r.get("excluded"))
    print(f"\nDone. Total {len(existing)}  usable {len(existing) - n_excl}  excluded {n_excl}")
    wh.close()


if __name__ == "__main__":
    main()
