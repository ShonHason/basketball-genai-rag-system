"""
Re-grade saved RAG answers with one or more judges, without re-running the RAG pipeline.
With --labels, measures how often each judge agrees with human grades (judge calibration).

    uv run python -m evaluation.rejudge evaluation/results/X.jsonl \
        --judges openai/gpt-4.1-mini openai/gpt-5-mini@minimal openrouter/anthropic/claude-haiku-5.5 \
        --labels evaluation/calibration_labels.json
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import litellm

from evaluation import eval as ev
from evaluation.run import RESULTS_DIR, load_run, openrouter_price
from evaluation.test import load_tests

_lock = threading.Lock()
usage = {"calls": 0, "usd": 0.0}
def _tracked(fn):
    def wrapper(*args, **kwargs):
        response = fn(*args, **kwargs)
        model = kwargs.get("model", "")
        if model.startswith("openrouter/"):
            price_in, price_out = openrouter_price(model)
            u = response.usage
            cost = u.prompt_tokens * price_in + u.completion_tokens * price_out
        else:
            try:
                cost = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                cost = 0.0
        with _lock:
            usage["calls"] += 1
            usage["usd"] += cost
        return response
    return wrapper


ev.completion = _tracked(ev.completion)


def parse_judge(spec: str) -> tuple[str, str | None]:
    model, _, effort = spec.partition("@")
    return model, effort or None


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="A saved run (.jsonl) whose results include retrieved chunk texts")
    parser.add_argument("--judges", nargs="+", required=True, help="model or model@reasoning_effort")
    parser.add_argument("--labels", help="Human labels JSON for calibration")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    _, results = load_run(args.results)
    tests = load_tests()  # current references, in case the dataset was corrected after the run
    results = [r for r in results if r.contexts and not r.error]
    labels = json.load(open(args.labels))["labels"] if args.labels else {}
    if labels:
        results = [r for r in results if str(r.index) in labels]
    print(f"Re-judging {len(results)} answers with {len(args.judges)} judges\n")

    report = []
    for spec in args.judges:
        model, effort = parse_judge(spec)
        ev.JUDGE_MODEL = model
        ev.JUDGE_REASONING_EFFORT = effort or "low"
        usage.update(calls=0, usd=0.0)

        def grade(r):
            docs = [SimpleNamespace(page_content=c, metadata={"source": s}) for c, s in zip(r.contexts, r.sources)]
            try:
                return r.index, ev.judge_answer(tests[r.index], r.generated_answer, docs), None
            except Exception as e:
                return r.index, None, f"{type(e).__name__}: {str(e)[:200]}"

        start = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            graded = list(pool.map(grade, results))
        elapsed = time.perf_counter() - start

        ok = [(i, a) for i, a, err in graded if a]
        errors = [(i, err) for i, _, err in graded if err]
        entry = {
            "judge": spec, "graded": len(ok), "errors": len(errors),
            "usd": usage["usd"], "usd_per_call": usage["usd"] / max(usage["calls"], 1), "seconds": elapsed,
            "verdicts": {str(i): {"correct": a.correct, "faithful": a.faithful, "feedback": a.feedback} for i, a in ok},
        }
        if errors:
            entry["first_error"] = errors[0][1]
        if labels and ok:
            human = [labels[str(i)]["correct"] for i, _ in ok]
            judge = [a.correct for _, a in ok]
            clear = [(h, j) for (i, _), h, j in zip(ok, human, judge) if not labels[str(i)].get("partial")]
            entry["agreement_%"] = 100 * sum(h == j for h, j in zip(human, judge)) / len(ok)
            entry["agreement_clear_%"] = 100 * sum(h == j for h, j in clear) / len(clear) if clear else None
            entry["kappa"] = cohen_kappa(human, judge)
            entry["disagreements"] = [
                {"index": i, "human": labels[str(i)]["correct"], "judge": a.correct, "judge_feedback": a.feedback[:300]}
                for i, a in ok if a.correct != labels[str(i)]["correct"]
            ]
        report.append(entry)
        line = f"{spec:<45} graded {len(ok):>3}  errors {len(errors):>2}  ${usage['usd']:.4f}  {elapsed:>5.0f}s"
        if "agreement_%" in entry:
            line += f"  agree {entry['agreement_%']:.1f}% (clear cases {entry['agreement_clear_%']:.1f}%)  kappa {entry['kappa']:.2f}"
        print(line)
        if errors:
            print(f"    first error: {errors[0][1]}")

    out = RESULTS_DIR / f"rejudge_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
