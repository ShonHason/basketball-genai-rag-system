import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "rag_eval_nla.jsonl")


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_tests(path: str = TEST_FILE) -> list[TestQuestion]:
    """Load test questions from JSONL file."""
    tests = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(TestQuestion(**json.loads(line)))
    return tests
