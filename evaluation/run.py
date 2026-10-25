"""
Batch evaluation runner: runs the whole test set in parallel, saves every result, prints a summary.

    uv run python -m evaluation.run --label baseline
    uv run python -m evaluation.run --label no_rerank --no-rerank
    uv run python -m evaluation.run --label quick --limit 40 --category ranking multi_hop
    uv run python -m evaluation.run --label cs500 --collection docs_cs500
    uv run python -m evaluation.run --compare evaluation/results/A.jsonl evaluation/results/B.jsonl
"""
import argparse
import json
import random
import statistics
import subprocess
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import litellm
from tqdm import tqdm

from evaluation.test import load_tests
from evaluation import eval as ev
from evaluation.eval import TestResult, run_test
import pro_implementation.answer as rag

RESULTS_DIR = Path(__file__).parent / "results"

# --- Cost tracking: wrap the litellm calls made by the pipeline and the judge ---
_cost_lock = threading.Lock()
costs = defaultdict(float)


_openrouter_prices = None


def openrouter_price(model: str) -> tuple[float, float]:
    """Per-token (input, output) price from OpenRouter's public model list."""
    global _openrouter_prices
    if _openrouter_prices is None:
        data = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30))["data"]
        _openrouter_prices = {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}
    return _openrouter_prices.get(model.removeprefix("openrouter/"), (0.0, 0.0))


def _tracked(fn):
    def wrapper(*args, **kwargs):
        response = fn(*args, **kwargs)
        model = kwargs.get("model", "")
        try:
            if model.startswith("openrouter/"):
                price_in, price_out = openrouter_price(model)
                cost = response.usage.prompt_tokens * price_in + response.usage.completion_tokens * price_out
            else:
                cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        with _cost_lock:
            costs[kwargs.get("model", "?")] += cost or 0.0
        return response
    return wrapper


rag.completion = _tracked(rag.completion)
ev.completion = _tracked(ev.completion)


def cost_summary() -> dict:
    judge = sum(v for k, v in costs.items() if k == ev.JUDGE_MODEL)
    return {"pipeline_usd": sum(costs.values()) - judge, "judge_usd": judge}


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def pct(values):
    values = [v for v in values if v is not None]
    return 100 * sum(values) / len(values) if values else None


def summarize(results: list[TestResult]) -> dict:
    ok = [r for r in results if not r.error]
    retr = [r.retrieval for r in ok if r.retrieval]
    informative = [r for r in retr if r.informative_keywords]
    ans = [r.answer for r in ok if r.answer]
    latencies = sorted(r.latency_s for r in ok)
    return {
        "n": len(results),
        "errors": len(results) - len(ok),
        "correct_%": pct([a.correct for a in ans]),
        "faithful_%": pct([a.faithful for a in ans]),
        "accuracy": mean([a.accuracy for a in ans]),
        "completeness": mean([a.completeness for a in ans]),
        "relevance": mean([a.relevance for a in ans]),
        "ctx_recall": mean([r.context_recall for r in retr]),
        "ctx_sufficient_%": pct([r.context_sufficient for r in retr]),
        "precision@k": mean([r.precision_at_k for r in retr]),
        "judged_mrr": mean([r.judged_mrr for r in retr]),
        "kw_mrr": mean([r.mrr for r in informative]),
        "kw_coverage_%": mean([r.keyword_coverage for r in informative]),
        "latency_p50": latencies[len(latencies) // 2] if latencies else None,
        "latency_p95": latencies[int(len(latencies) * 0.95) - 1] if len(latencies) >= 20 else None,
    }


COLUMNS = ["n", "correct_%", "faithful_%", "accuracy", "ctx_recall", "ctx_sufficient_%", "judged_mrr", "kw_coverage_%", "latency_p50"]


def print_table(rows: dict[str, dict]):
    header = f"{'category':<16}" + "".join(f"{c:>17}" for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for name, s in rows.items():
        cells = []
        for c in COLUMNS:
            v = s.get(c)
            cells.append(f"{'-':>17}" if v is None else f"{v:>17.2f}" if isinstance(v, float) else f"{v:>17}")
        print(f"{name:<16}" + "".join(cells))


def report(results: list[TestResult]):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    rows = {cat: summarize(rs) for cat, rs in sorted(by_cat.items())}
    rows["ALL"] = summarize(results)
    print_table(rows)

    errors = [r for r in results if r.error]
    if errors:
        print(f"\n{len(errors)} tests errored, e.g. #{errors[0].index}: {errors[0].error}")
    failures = [r for r in results if r.answer and not r.answer.correct]
    if failures:
        print(f"\nIncorrect answers ({len(failures)}), first 10:")
        for r in failures[:10]:
            sufficient = r.retrieval.context_sufficient if r.retrieval else None
            # Retrieval failure vs generation failure tells you which stage to fix
            if r.category == "unanswerable":
                stage = "hallucination"
            else:
                stage = "retrieval" if sufficient is False else "generation" if sufficient else "?"
            print(f"  #{r.index:<4} [{r.category}] ({stage} failure) {r.question[:90]}")


def config_snapshot(args) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "generator_model": rag.MODEL,
        "generator_reasoning_effort": rag.REASONING_EFFORT if "gpt-5" in rag.MODEL else None,
        "reranker": (rag.LOCAL_RERANKER_MODEL if rag.RERANKER == "local" else rag.MODEL) if not args.no_rerank else None,
        "judge_model": ev.JUDGE_MODEL,
        "judge_reasoning_effort": ev.JUDGE_REASONING_EFFORT if "gpt-5" in ev.JUDGE_MODEL else None,
        "collection": rag.collection_name,
        "chunks_in_collection": rag.collection.count(),
        "embedding_model": rag.embedding_model,
        "retrieval_k": rag.RETRIEVAL_K,
        "final_k": rag.FINAL_K,
        "use_rewrite": not args.no_rewrite,
        "use_rerank": not args.no_rerank,
        "with_judge": not args.no_judge,
        "limit": args.limit,
        "categories": args.category,
        "seed": args.seed,
    }


def load_run(path: str) -> tuple[dict, list[TestResult]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    config = json.loads(lines[0])["config"]
    return config, [TestResult.model_validate_json(line) for line in lines[1:]]


def compare(path_a: str, path_b: str):
    config_a, results_a = load_run(path_a)
    config_b, results_b = load_run(path_b)
    sa, sb = summarize(results_a), summarize(results_b)
    print(f"A = {config_a['label']} ({config_a['timestamp']})\nB = {config_b['label']} ({config_b['timestamp']})\n")
    print(f"{'metric':<18}{'A':>10}{'B':>10}{'delta':>10}")
    for key in sa:
        a, b = sa[key], sb[key]
        delta = f"{b - a:+10.2f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else f"{'':>10}"
        fmt = lambda v: f"{'-':>10}" if v is None else f"{v:>10.2f}" if isinstance(v, float) else f"{v:>10}"
        print(f"{key:<18}{fmt(a)}{fmt(b)}{delta}")

    a_by_q = {r.question: r for r in results_a if r.answer}
    flips = defaultdict(list)
    for rb in results_b:
        ra = a_by_q.get(rb.question)
        if ra and rb.answer and ra.answer.correct != rb.answer.correct:
            flips["fixed" if rb.answer.correct else "broken"].append(rb)
    for kind in ("broken", "fixed"):
        print(f"\n{kind.upper()} in B ({len(flips[kind])}):")
        for r in flips[kind]:
            print(f"  #{r.index:<4} [{r.category}] {r.question[:90]}")


def main():
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite")
    parser.add_argument("--label", default="run", help="Name for this run (used in the results filename)")
    parser.add_argument("--limit", type=int, help="Evaluate a random sample of N tests")
    parser.add_argument("--category", nargs="+", help="Only these categories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-rewrite", action="store_true", help="Ablation: skip query rewriting")
    parser.add_argument("--no-rerank", action="store_true", help="Ablation: skip LLM reranking")
    parser.add_argument("--no-judge", action="store_true", help="Keyword metrics only (no judge cost)")
    parser.add_argument("--collection", help="Chroma collection to retrieve from (e.g. docs_cs500)")
    parser.add_argument("--judge-model", help=f"Override the judge model (default {ev.JUDGE_MODEL})")
    parser.add_argument("--reranker", choices=["local", "llm"], help=f"Reranker (default {rag.RERANKER})")
    parser.add_argument("--model", help=f"Override the RAG generator model (default {rag.MODEL}), e.g. openai/gpt-5-nano or ollama_chat/qwen3.5:9b")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two saved runs")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    out, results = execute(args)
    print()
    report(results)
    print(f"\nCost: {cost_summary()}")
    print(f"Saved to {out}")


def regrade_errors(results: list[TestResult]) -> list[TestResult]:
    """Second pass, one at a time: re-judge tests whose answer exists but whose judging failed."""
    tests = load_tests()
    failed = [r for r in results if r.error and r.generated_answer and r.contexts]
    if failed:
        print(f"Re-judging {len(failed)} tests whose grading failed...")
    return [ev.regrade_failed(r, tests[r.index]) if r in failed else r for r in results]


def repair_run_file(path: Path) -> int:
    """Re-judge failed tests in a saved run file in place; returns how many errors remain."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    results = [TestResult.model_validate_json(line) for line in lines[1:]]
    if not any(r.error for r in results):
        return 0
    costs.clear()
    results = regrade_errors(results)
    old = header["summary"]
    extra = cost_summary()
    header["summary"] = summarize(results) | {
        "pipeline_usd": old.get("pipeline_usd", 0),
        "judge_usd": (old.get("judge_usd") or 0) + extra["judge_usd"],
        "latency_p50": old.get("latency_p50"),
        "latency_p95": old.get("latency_p95"),
    }
    header["config"]["repaired"] = True
    path.write_text("\n".join([json.dumps(header)] + [r.model_dump_json() for r in results]) + "\n", encoding="utf-8")
    return sum(1 for r in results if r.error)


def select_tests(args) -> list:
    indexed = list(enumerate(load_tests()))
    if args.category:
        indexed = [(i, t) for i, t in indexed if t.category in args.category]
    if args.limit:
        random.Random(args.seed).shuffle(indexed)
        indexed = sorted(indexed[: args.limit], key=lambda it: it[0])
    return indexed


def execute(args) -> tuple[Path, list[TestResult]]:
    """Run one evaluation configuration and save it. Also used by evaluation/experiment.py."""
    if getattr(args, "model", None):
        rag.MODEL = args.model
    if getattr(args, "reranker", None):
        rag.RERANKER = args.reranker
    if args.collection:
        rag.use_collection(args.collection)
    if args.judge_model:
        ev.JUDGE_MODEL = args.judge_model
    rag.DEBUG = False
    costs.clear()

    indexed = select_tests(args)
    config = config_snapshot(args)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_test, i, t, not args.no_rewrite, not args.no_rerank, not args.no_judge)
            for i, t in indexed
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {args.label}"):
            results.append(future.result())
    results.sort(key=lambda r: r.index)
    if not args.no_judge:
        results = regrade_errors(results)

    summary = summarize(results) | cost_summary()
    out_dir = getattr(args, "out_dir", None)
    if out_dir:  # experiment runs are grouped in one folder, one file per configuration
        out = Path(out_dir) / f"{args.label}.jsonl"
    else:
        out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"config": config, "summary": summary}) + "\n")
        for r in results:
            f.write(r.model_dump_json() + "\n")
    return out, results


if __name__ == "__main__":
    main()
