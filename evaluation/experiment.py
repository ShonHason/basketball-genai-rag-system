"""
2x2 ablation per collection: query rewriting x reranking.

Build the collections first, e.g.:
    uv run python -m pro_implementation.ingest --chunk-size 500
Validate the matrix without spending anything, then run:
    uv run python -m evaluation.experiment --dry-run
    uv run python -m evaluation.experiment
    uv run python -m evaluation.experiment --reuse   # copy matching completed full runs instead of re-running them
"""
import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from evaluation.run import RESULTS_DIR, execute, summarize, load_run, repair_run_file
from evaluation.test import load_tests
from evaluation import eval as ev
import pro_implementation.answer as rag

DEFAULT_COLLECTIONS = ["docs", "docs_cs300", "docs_cs600", "docs_cs800", "docs_cs1000", "docs_cs1200"]
VARIANTS = [  # (suffix, use_rewrite, use_rerank)
    ("rw_rr", True, True),
    ("rw_only", True, False),
    ("rr_only", False, True),
    ("none", False, False),
]
BASELINE = "docs__rw_rr"

TABLE_METRICS = [
    ("correct_%", "correct%", 1),
    ("faithful_%", "faithful%", 1),
    ("accuracy", "acc/5", 2),
    ("ctx_recall", "ctx_recall", 2),
    ("ctx_sufficient_%", "sufficient%", 1),
    ("judged_mrr", "jMRR", 2),
    ("precision@k", "P@k", 2),
    ("kw_coverage_%", "kw_cov%", 1),
    ("latency_p50", "p50_s", 1),
    ("pipeline_usd", "pipe_$", 3),
    ("judge_usd", "judge_$", 3),
]


def build_matrix(collections: list[str]) -> list[dict]:
    return [
        {"label": f"{c}__{suffix}", "collection": c, "use_rewrite": rw, "use_rerank": rr}
        for c in collections
        for suffix, rw, rr in VARIANTS
    ]


def validate(matrix: list[dict], n_tests: int) -> list[str]:
    problems = []
    labels = [m["label"] for m in matrix]
    if len(set(labels)) != len(labels):
        problems.append("duplicate configuration labels")
    existing = {c.name for c in rag.chroma.list_collections()}
    for c in sorted({m["collection"] for m in matrix}):
        if c not in existing:
            problems.append(f"collection {c!r} does not exist (build it with pro_implementation.ingest)")
        elif rag.chroma.get_collection(c).count() == 0:
            problems.append(f"collection {c!r} is empty")
    if n_tests == 0:
        problems.append("no tests loaded")
    return problems


def find_reusable(cfg: dict, n_tests: int) -> Path | None:
    """Latest completed full run (all tests, no filters) with exactly this configuration and models."""
    candidates = []
    for f in RESULTS_DIR.rglob("*.jsonl"):  # includes earlier experiment folders, so an interrupted grid can resume
        c = json.loads(f.open(encoding="utf-8").readline())["config"]
        if (
            c.get("collection") == cfg["collection"]
            and c.get("use_rewrite") == cfg["use_rewrite"]
            and c.get("use_rerank") == cfg["use_rerank"]
            and c.get("generator_model") == rag.MODEL
            and c.get("judge_model") == ev.JUDGE_MODEL
            and c.get("judge_reasoning_effort") == (ev.JUDGE_REASONING_EFFORT if "gpt-5" in ev.JUDGE_MODEL else None)
            and c.get("generator_reasoning_effort") == (rag.REASONING_EFFORT if "gpt-5" in rag.MODEL else None)
            and (not cfg["use_rerank"] or c.get("reranker") == (rag.LOCAL_RERANKER_MODEL if rag.RERANKER == "local" else rag.MODEL))
            and c.get("final_k") == rag.FINAL_K
            and c.get("with_judge")
            and not c.get("limit")
            and not c.get("categories")
        ):
            _, results = load_run(str(f))
            # Judge-only failures (answer and chunks saved) are repaired when the run is reused
            if len(results) == n_tests and all(r.generated_answer and r.contexts for r in results if r.error):
                candidates.append(f)
    return max(candidates) if candidates else None


def fmt(v, digits):
    return "-" if v is None else f"{v:.{digits}f}"


def consolidated_table(runs: list[dict]) -> str:
    width = max(len(r["label"]) for r in runs) + 2
    lines = [f"{'config':<{width}}" + "".join(f"{label:>12}" for _, label, _ in TABLE_METRICS)]
    lines.append("-" * len(lines[0]))
    for r in runs:
        lines.append(f"{r['label']:<{width}}" + "".join(f"{fmt(r['summary'].get(k), d):>12}" for k, _, d in TABLE_METRICS))
    return "\n".join(lines)


def delta(v, base, digits):
    if v is None or base is None:
        return "-", "-"
    rel = f"{(v - base) / base * 100:+.1f}%" if base else "-"
    return f"{v - base:+.{digits}f}", rel


def ranking_table(runs: list[dict]) -> str:
    base = next((r["summary"] for r in runs if r["label"] == BASELINE), None)
    ranked = sorted(
        runs,
        key=lambda r: (
            -(r["summary"].get("correct_%") or 0),
            -(r["summary"].get("faithful_%") or 0),
            r["summary"].get("latency_p50") or float("inf"),
            r["summary"].get("pipeline_usd") or float("inf"),
        ),
    )
    width = max(len(r["label"]) for r in runs) + 2
    header = (f"{'#':>3} {'config':<{width}}{'correct%':>9}{'Δabs':>8}{'Δrel':>9}"
              f"{'faithful%':>11}{'Δabs':>8}{'p50_s':>8}{'Δrel':>9}{'pipe_$':>9}{'Δrel':>9}")
    lines = [f"Baseline = {BASELINE}", header, "-" * len(header)]
    for i, r in enumerate(ranked, start=1):
        s = r["summary"]
        b = base or {}
        c_abs, c_rel = delta(s.get("correct_%"), b.get("correct_%"), 1)
        f_abs, _ = delta(s.get("faithful_%"), b.get("faithful_%"), 1)
        _, l_rel = delta(s.get("latency_p50"), b.get("latency_p50"), 1)
        _, p_rel = delta(s.get("pipeline_usd"), b.get("pipeline_usd"), 3)
        lines.append(
            f"{i:>3} {r['label']:<{width}}{fmt(s.get('correct_%'), 1):>9}{c_abs:>8}{c_rel:>9}"
            f"{fmt(s.get('faithful_%'), 1):>11}{f_abs:>8}{fmt(s.get('latency_p50'), 1):>8}{l_rel:>9}"
            f"{fmt(s.get('pipeline_usd'), 3):>9}{p_rel:>9}"
        )
    return "\n".join(lines)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", nargs="+", default=DEFAULT_COLLECTIONS)
    parser.add_argument("--judge-model", default=ev.JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate the matrix and exit without API calls")
    parser.add_argument("--reuse", action="store_true", help="Copy matching completed full runs instead of re-running")
    args = parser.parse_args()

    ev.JUDGE_MODEL = args.judge_model
    tests = load_tests()
    matrix = build_matrix(args.collections)
    reusable = {m["label"]: find_reusable(m, len(tests)) for m in matrix} if args.reuse else {}

    print(f"Configurations: {len(matrix)} ({len(args.collections)} collections x {len(VARIANTS)} variants)")
    print(f"Questions per configuration: {len(tests)}  ->  {len(matrix) * len(tests)} question runs")
    print(f"Generator: {rag.MODEL} ({rag.REASONING_EFFORT}) | Reranker: {rag.LOCAL_RERANKER_MODEL if rag.RERANKER == 'local' else rag.MODEL} "
          f"| Judge: {ev.JUDGE_MODEL} ({ev.JUDGE_REASONING_EFFORT}) | final k: {rag.FINAL_K} | retrieval k: {rag.RETRIEVAL_K}")
    for m in matrix:
        source = reusable.get(m["label"])
        note = f"reuse {source.name}" if source else "run"
        print(f"  {m['label']:<24} rewrite={m['use_rewrite']!s:<5} rerank={m['use_rerank']!s:<5} {note}")
    to_run = [m for m in matrix if not reusable.get(m["label"])]
    print(f"To run: {len(to_run)} configurations, reused: {len(matrix) - len(to_run)}")

    problems = validate(matrix, len(tests))
    if problems:
        print("\nValidation FAILED:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print("Validation OK")
    if args.dry_run:
        return

    exp_dir = RESULTS_DIR / f"experiment_{datetime.now():%Y%m%d_%H%M%S}"
    exp_dir.mkdir(parents=True)
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "generator_model": rag.MODEL,
        "generator_reasoning_effort": rag.REASONING_EFFORT,
        "reranker": rag.LOCAL_RERANKER_MODEL if rag.RERANKER == "local" else rag.MODEL,
        "judge_model": ev.JUDGE_MODEL,
        "judge_reasoning_effort": ev.JUDGE_REASONING_EFFORT,
        "embedding_model": rag.embedding_model,
        "retrieval_k": rag.RETRIEVAL_K,
        "final_k": rag.FINAL_K,
        "n_tests": len(tests),
        "seed": args.seed,
        "baseline": BASELINE,
        "matrix": [m | {"reused_from": str(reusable[m["label"]]) if reusable.get(m["label"]) else None} for m in matrix],
    }
    (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    runs = []
    for m in matrix:
        out = exp_dir / f"{m['label']}.jsonl"
        if reusable.get(m["label"]):
            shutil.copy(reusable[m["label"]], out)
            remaining = repair_run_file(out)  # repairs the copy; the original stays untouched
            if remaining:
                print(f"  {m['label']}: {remaining} tests still failed after re-judging")
        else:
            run_args = argparse.Namespace(
                label=m["label"], collection=m["collection"], judge_model=args.judge_model,
                no_rewrite=not m["use_rewrite"], no_rerank=not m["use_rerank"], no_judge=False,
                workers=args.workers, limit=None, category=None, seed=args.seed, out_dir=exp_dir,
            )
            execute(run_args)
        header = json.loads(out.open(encoding="utf-8").readline())
        _, results = load_run(str(out))
        runs.append({
            **m,
            "file": out.name,
            "summary": header["summary"],
            "by_category": {cat: summarize([r for r in results if r.category == cat]) for cat in sorted({r.category for r in results})},
        })
        print(f"  -> {m['label']}: correct {header['summary']['correct_%']:.1f}%")

    tables = "CONSOLIDATED\n" + consolidated_table(runs) + "\n\nRANKING\n" + ranking_table(runs)
    print("\n" + tables)
    (exp_dir / "summary.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (exp_dir / "tables.txt").write_text(tables + "\n", encoding="utf-8")
    print(f"\nSaved to {exp_dir}")


if __name__ == "__main__":
    main()
