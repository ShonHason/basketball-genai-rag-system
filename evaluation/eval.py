import sys
import math
import threading
import time
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from litellm import completion
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests
import pro_implementation.answer as rag
from pro_implementation.answer import answer_question
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)
# Judge providers rate-limit bursts (HTTP 429); retry patiently so long runs don't drop questions
wait = wait_exponential(multiplier=2, min=5, max=90)
stop = stop_after_attempt(10)
# OpenRouter's shared Gemini capacity returns "rate-limited upstream" under bursts,
# so cap concurrent judge calls regardless of how many test workers run
JUDGE_CONCURRENCY = 3
_judge_slots = threading.BoundedSemaphore(JUDGE_CONCURRENCY)

# The judge should be stronger than (and ideally different from) the generator model,
# otherwise it tends to grade its own mistakes as correct.
# Chosen by calibration against manual grades (evaluation/calibration_labels.json, rejudge.py):
# 95% agreement, kappa 0.86, cheapest of the judges tested
JUDGE_MODEL = "openrouter/google/gemini-2.5-flash-lite"
JUDGE_REASONING_EFFORT = "low"


def judge_kwargs() -> dict:
    # Explicit per-call timeout: a hung OpenRouter request otherwise blocks a worker indefinitely
    kwargs = {"timeout": 60}
    if "gpt-5" in JUDGE_MODEL:
        kwargs["reasoning_effort"] = JUDGE_REASONING_EFFORT
    return kwargs

KNOWLEDGE_BASE = Path(__file__).parent.parent / "data"
# A keyword found in more than this share of KB files ("C", "7", "USA", "leader")
# matches almost any chunk, so it says nothing about retrieval quality.
GENERIC_KEYWORD_SHARE = 0.25


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    # Keyword-based (cheap, deterministic, but noisy)
    mrr: float = Field(description="Mean Reciprocal Rank - average across informative keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")
    informative_keywords: bool = Field(description="False if every keyword was generic, so keyword metrics are unreliable")

    # Judge-based (relevance labelled per chunk by an LLM that knows the reference answer)
    context_recall: float | None = Field(default=None, description="0-1 share of reference facts present in the context")
    context_sufficient: bool | None = Field(default=None, description="Reference answer fully derivable from the context")
    precision_at_k: float | None = Field(default=None, description="Share of retrieved chunks judged relevant")
    judged_mrr: float | None = Field(default=None, description="Reciprocal rank of the first relevant chunk")
    judged_ndcg: float | None = Field(default=None, description="nDCG over judged relevant chunks")


class ContextJudgement(BaseModel):
    """LLM judgement of the retrieved context against the reference answer."""

    reasoning: str = Field(description="Brief reasoning: which facts of the reference answer appear in which chunks")
    relevant_chunk_ids: list[int] = Field(description="IDs of chunks that contain at least one fact needed for the reference answer")
    context_recall: float = Field(description="Fraction (0.0-1.0) of the facts in the reference answer that are stated in the chunks")
    sufficient: bool = Field(description="True only if the full reference answer can be derived from the chunks alone")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    correct: bool = Field(
        description="True if the answer's key facts agree with the reference answer and it contains no contradicting facts. For unanswerable questions, True only if the answer says the information is not available."
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )
    unsupported_claims: list[str] = Field(
        description="Factual claims in the generated answer that are NOT stated in the retrieved context (hallucinations). Empty list if none."
    )
    faithful: bool = Field(description="True if every factual claim in the generated answer is supported by the retrieved context")


class TestResult(BaseModel):
    """Everything recorded for one test, so runs can be saved and compared."""

    index: int
    question: str
    category: str
    reference_answer: str
    generated_answer: str = ""
    sources: list[str] = []
    contexts: list[str] = []  # retrieved chunk texts, so a new judge can re-grade without rerunning the RAG
    retrieval: RetrievalEval | None = None
    answer: AnswerEval | None = None
    latency_s: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Keyword metrics
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _kb_texts() -> list[str]:
    return [p.read_text(encoding="utf-8").lower() for p in KNOWLEDGE_BASE.rglob("*.md")]


def is_generic_keyword(keyword: str) -> bool:
    texts = _kb_texts()
    hits = sum(keyword.lower() in t for t in texts)
    return hits > GENERIC_KEYWORD_SHARE * len(texts)


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def ndcg_from_relevances(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal, k)
    return calculate_dcg(relevances, k) / idcg if idcg > 0 else 0.0


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()
    relevances = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]
    return ndcg_from_relevances(relevances, k)


def keyword_metrics(test: TestQuestion, retrieved_docs: list, k: int = 10) -> RetrievalEval:
    keywords = [kw for kw in test.keywords if not is_generic_keyword(kw)]
    informative = bool(keywords)
    if not informative:
        keywords = test.keywords

    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in keywords]
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in keywords]
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(keywords)

    return RetrievalEval(
        mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        ndcg=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=(keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0,
        informative_keywords=informative,
    )


# ---------------------------------------------------------------------------
# LLM judges
# ---------------------------------------------------------------------------

# A judgment is well under 1,000 tokens. The cap stops a model that falls into a repetition
# loop (seen with Gemini Flash-Lite: a 330,000-character "feedback" field) within seconds.
JUDGE_MAX_TOKENS = 2000


def _judge_call(messages: list, schema: type[BaseModel]) -> BaseModel:
    """Call the judge; if the output is malformed (e.g. a truncated repetition loop), retry once
    with some randomness, since at temperature 0 the same loop would repeat on every retry."""
    for temperature in (0, 0.7):
        with _judge_slots:
            response = completion(
                model=JUDGE_MODEL, messages=messages, response_format=schema,
                temperature=temperature, max_tokens=JUDGE_MAX_TOKENS, **judge_kwargs(),
            )
        try:
            return schema.model_validate_json(response.choices[0].message.content)
        except ValidationError:
            if temperature:
                raise


def format_chunks(retrieved_docs: list) -> str:
    return "\n\n".join(
        f"### Chunk ID {i} | Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, start=1)
    )


@retry(wait=wait, stop=stop)
def judge_context(test: TestQuestion, retrieved_docs: list) -> ContextJudgement:
    messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator of retrieval systems. You decide which retrieved chunks contain the facts needed to produce a reference answer. Be strict: a chunk that only mentions the same player or team without the needed fact is NOT relevant.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Reference Answer:
{test.reference_answer}

Retrieved Chunks:
{format_chunks(retrieved_docs)}

List the IDs of the relevant chunks, estimate the fraction of reference facts present in the chunks, and say whether the chunks are sufficient to produce the reference answer.""",
        },
    ]
    return _judge_call(messages, ContextJudgement)


def judged_metrics(judgement: ContextJudgement, retrieved_docs: list) -> dict:
    relevant = set(judgement.relevant_chunk_ids)
    relevances = [1 if i in relevant else 0 for i in range(1, len(retrieved_docs) + 1)]
    first = next((rank for rank, rel in enumerate(relevances, start=1) if rel), None)
    return {
        "context_recall": max(0.0, min(1.0, judgement.context_recall)),
        "context_sufficient": judgement.sufficient,
        "precision_at_k": sum(relevances) / len(relevances) if relevances else 0.0,
        "judged_mrr": 1.0 / first if first else 0.0,
        "judged_ndcg": ndcg_from_relevances(relevances, len(relevances)),
    }


@retry(wait=wait, stop=stop)
def judge_answer(test: TestQuestion, generated_answer: str, retrieved_docs: list) -> AnswerEval:
    unanswerable_note = (
        "\nThis question is UNANSWERABLE from the league records. The ideal answer clearly says the information is not available; any invented fact makes it wrong.\n"
        if not test.answerable
        else ""
    )
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers from a RAG system. Evaluate the generated answer by comparing it to the reference answer, and check every factual claim against the retrieved context. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}
{unanswerable_note}
Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Retrieved Context (what the system saw):
{format_chunks(retrieved_docs)}

Please evaluate the generated answer:
1. Correct: Do its key facts agree with the reference answer, with no contradicting facts?
2. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
3. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
4. Relevance: How well does it directly answer the specific question asked, giving no additional information?
5. Faithfulness: List any factual claims NOT supported by the retrieved context.

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]
    return _judge_call(judge_messages, AnswerEval)


# ---------------------------------------------------------------------------
# Running tests
# ---------------------------------------------------------------------------

def grade(test: TestQuestion, result: TestResult, retrieved_docs: list, with_judge: bool = True) -> TestResult:
    """Fill in retrieval and answer metrics for an already generated answer."""
    if test.answerable and test.keywords:
        result.retrieval = keyword_metrics(test, retrieved_docs)
        if with_judge:
            judgement = judge_context(test, retrieved_docs)
            result.retrieval = result.retrieval.model_copy(update=judged_metrics(judgement, retrieved_docs))
    if with_judge:
        result.answer = judge_answer(test, result.generated_answer, retrieved_docs)
    return result


def regrade_failed(result: TestResult, test: TestQuestion) -> TestResult:
    """Retry only the judging of a test whose answer was generated but grading failed (e.g. judge rate limit)."""
    if not (result.error and result.generated_answer and result.contexts):
        return result
    docs = [SimpleNamespace(page_content=c, metadata={"source": s}) for c, s in zip(result.contexts, result.sources)]
    try:
        return grade(test, result.model_copy(update={"error": None}), docs)
    except Exception:
        return result


def run_test(
    index: int,
    test: TestQuestion,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    with_judge: bool = True,
) -> TestResult:
    """Run the RAG pipeline ONCE and score both retrieval and answer from that single run."""
    result = TestResult(
        index=index, question=test.question, category=test.category, reference_answer=test.reference_answer
    )
    try:
        rag.reset_reranker_wait()
        start = time.perf_counter()
        generated_answer, retrieved_docs = answer_question(
            test.question, [], use_rewrite=use_rewrite, use_rerank=use_rerank
        )
        # Exclude time spent queuing for the shared local reranker behind other test workers
        result.latency_s = time.perf_counter() - start - rag.reranker_wait_seconds()
        result.generated_answer = generated_answer
        result.sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
        result.contexts = [doc.page_content for doc in retrieved_docs]

        grade(test, result, retrieved_docs, with_judge)
    except Exception as e:  # keep the run going; failures are reported per test
        result.error = f"{type(e).__name__}: {e}"
    return result


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """Keyword-only retrieval evaluation (used by the Gradio dashboard)."""
    from pro_implementation.answer import fetch_context

    return keyword_metrics(test, fetch_context(test.question, []), k)


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """Evaluate answer quality using LLM-as-a-judge."""
    generated_answer, retrieved_docs = answer_question(test.question)
    return judge_answer(test, generated_answer, retrieved_docs), generated_answer, retrieved_docs


def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""
    tests = [t for t in load_tests() if t.answerable and t.keywords]
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """Evaluate all answers to tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test."""
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    result = run_test(test_number, test)
    if result.error:
        print(f"\nERROR: {result.error}")
        return

    print(f"\nSources: {result.sources}")
    if result.retrieval:
        r = result.retrieval
        print(f"\n{'=' * 80}\nRetrieval Evaluation\n{'=' * 80}")
        print(f"Keyword MRR: {r.mrr:.4f}  nDCG: {r.ndcg:.4f}  Found: {r.keywords_found}/{r.total_keywords}"
              f"{'' if r.informative_keywords else '  (generic keywords only - unreliable)'}")
        print(f"Context recall: {r.context_recall:.2f}  Sufficient: {r.context_sufficient}  "
              f"Precision@k: {r.precision_at_k:.2f}  Judged MRR: {r.judged_mrr:.2f}")

    a = result.answer
    print(f"\n{'=' * 80}\nAnswer Evaluation\n{'=' * 80}")
    print(f"\nGenerated Answer:\n{result.generated_answer}")
    print(f"\nFeedback:\n{a.feedback}")
    print(f"\nCorrect: {a.correct}  Faithful: {a.faithful}")
    if a.unsupported_claims:
        print(f"Unsupported claims: {a.unsupported_claims}")
    print(f"Accuracy: {a.accuracy:.2f}/5  Completeness: {a.completeness:.2f}/5  Relevance: {a.relevance:.2f}/5")
    print(f"Latency: {result.latency_s:.1f}s\n{'=' * 80}\n")


def main():
    """CLI to evaluate a specific test by row number."""
    if len(sys.argv) != 2:
        print("Usage: uv run python -m evaluation.eval <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
