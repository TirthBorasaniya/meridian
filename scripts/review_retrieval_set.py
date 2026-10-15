"""Interactive manual review of 30 random items of the retrieval set R (SPEC S6.1).

For each sampled item the reviewer sees the question, the source chunk, and the top 10 chunks
retrieved by the full pipeline. The reviewer marks the item valid or invalid, may add other
retrieved chunks as relevant spans, and writes a note. Progress is saved after every item, so
the review can be interrupted and resumed.

Usage::

    uv run python scripts/review_retrieval_set.py --reviewer "name"
"""

import argparse
import json
import random
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from meridian.common.config import load_config
from meridian.eval.build_retrieval_set import DEFAULT_CONFIG_PATH, RetrievalSetConfig
from meridian.eval.datasets import read_jsonl, write_jsonl
from meridian.eval.spans import LabeledSpan, span_coverage
from meridian.retrieval.factory import build_retriever

WRAP_WIDTH = 100


def sample_indices(record_count: int, sample_size: int, seed: int) -> list[int]:
    """Return the seeded random item indices to review."""
    return sorted(random.Random(seed).sample(range(record_count), min(sample_size, record_count)))


def ask(prompt: str, allowed: set[str] | None = None) -> str:
    """Prompt until the answer is allowed."""
    while True:
        answer = input(prompt).strip()
        if allowed is None or answer.lower() in allowed:
            return answer


def show(title: str, text: str) -> None:
    """Print a wrapped block."""
    print(f"\n--- {title}")
    print(textwrap.fill(" ".join(text.split()), WRAP_WIDTH))


def review_item(record: dict, retriever, top_k: int, reviewer: str) -> None:
    """Review one item in place."""
    span_list = [LabeledSpan.from_dict(span_dict) for span_dict in record["spans"]]
    print("\n" + "=" * WRAP_WIDTH)
    print(f"{record['qid']}  question: {record['question']}")
    result = retriever.search(record["question"], mode="hybrid_rerank", k=top_k)
    for rank, chunk in enumerate(result.chunk_list, start=1):
        coverage = max((span_coverage(chunk.payload, span) for span in span_list), default=0.0)
        marker = "LABELED" if coverage >= 0.5 else ""
        show(
            f"rank {rank} {chunk.payload['arxiv_id']}v{chunk.payload['version']} "
            f"[{chunk.payload.get('section_title', '')}] {marker}",
            chunk.text[:900],
        )
    valid = ask("\nIs the question valid and answered by its labeled span? [y/n]: ", {"y", "n"})
    extra = ask("Ranks of other relevant chunks (comma-separated, blank for none): ")
    for rank_text in [part.strip() for part in extra.split(",") if part.strip()]:
        if not rank_text.isdigit() or not 1 <= int(rank_text) <= len(result.chunk_list):
            print(f"ignored rank {rank_text!r}")
            continue
        payload = result.chunk_list[int(rank_text) - 1].payload
        record["spans"].append(
            {
                "arxiv_id": payload["arxiv_id"],
                "version": int(payload["version"]),
                "start": int(payload["char_start"]),
                "end": int(payload["char_end"]),
                "source": "review",
            }
        )
    notes = ask("Reviewer notes: ")
    record["review"] = {
        "reviewed": True,
        "valid": valid.lower() == "y",
        "notes": notes,
        "reviewer": reviewer,
        "reviewed_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> None:
    """Run the review loop."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(RetrievalSetConfig, args.config)
    record_list = read_jsonl(config.output_path)
    retriever = build_retriever()
    index_list = sample_indices(len(record_list), config.review_sample_size, config.review_seed)
    for index in index_list:
        if record_list[index]["review"].get("reviewed"):
            continue
        review_item(record_list[index], retriever, config.review_top_k, args.reviewer)
        write_jsonl(config.output_path, record_list)

    meta_path = Path(config.meta_path)
    meta_dict = json.loads(meta_path.read_text())
    reviewed_list = [record_list[index] for index in index_list]
    meta_dict["review"] = {
        "sample_size": len(index_list),
        "sample_qids": [record["qid"] for record in reviewed_list],
        "reviewed": sum(1 for record in reviewed_list if record["review"].get("reviewed")),
        "invalid": sum(1 for record in reviewed_list if record["review"].get("valid") is False),
        "spans_added": sum(
            1
            for record in reviewed_list
            for span_dict in record["spans"]
            if span_dict.get("source") == "review"
        ),
    }
    meta_path.write_text(json.dumps(meta_dict, indent=2) + "\n")
    print(json.dumps(meta_dict["review"], indent=2))


if __name__ == "__main__":
    main()
