"""
Offline sanity checks for the test set (no API calls). Run before trusting any metric:

    python -m evaluation.audit_dataset
"""
import json
import re
from collections import Counter
from pathlib import Path

from evaluation.test import TEST_FILE

KNOWLEDGE_BASE = Path(__file__).parent.parent / "data"
GENERIC_KEYWORD_SHARE = 0.25


def main():
    files = {p.relative_to(KNOWLEDGE_BASE): p.read_text(encoding="utf-8").lower() for p in KNOWLEDGE_BASE.rglob("*.md")}
    rows = [json.loads(line) for line in open(TEST_FILE, encoding="utf-8") if line.strip()]

    print(f"{len(rows)} tests, {len(files)} KB files\n")
    print("Tests per category:")
    for cat, n in Counter(r["category"] for r in rows).most_common():
        print(f"  {cat:<16}{n}")

    missing, generic, all_generic = [], [], []
    for i, r in enumerate(rows):
        flags = []
        for kw in r["keywords"]:
            hits = sum(kw.lower() in text for text in files.values())
            if hits == 0:
                missing.append((i, kw))
            elif hits > GENERIC_KEYWORD_SHARE * len(files):
                generic.append((i, kw, hits))
                flags.append(kw)
        if r["keywords"] and len(flags) == len(r["keywords"]):
            all_generic.append(i)

    print(f"\nKeywords not found anywhere in the KB ({len(missing)}) -> broken test or typo:")
    for i, kw in missing:
        print(f"  #{i}: {kw!r}")

    print(f"\nGeneric keywords ({len(generic)}) -> match almost any chunk, inflate keyword metrics:")
    for i, kw, hits in generic:
        print(f"  #{i}: {kw!r} appears in {hits}/{len(files)} files")

    print(f"\nTests whose keywords are ALL generic ({len(all_generic)}) -> keyword metrics meaningless:")
    print(f"  {all_generic}")

    superlative = re.compile(r"\b(oldest|youngest|highest|lowest|most|least|leads?|best|top)\b", re.I)
    ranking = [i for i, r in enumerate(rows) if superlative.search(r["question"])]
    print(f"\nSuperlative questions ({len(ranking)}) -> check the KB for ties before trusting the reference answer:")
    print(f"  {ranking}")


if __name__ == "__main__":
    main()
