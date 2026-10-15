"""Draft reference answers for answer-set items that lack one (SPEC S6.1).

For each item the full retrieval pipeline returns the top passages from the live index, and the
`question_writer` role (never the generator) writes a reference answer from them. The papers of
the passages it relies on are recorded in `reference_arxiv_ids`, which also feeds the training
exclusion list. Drafts are marked `reference_reviewed: false` until a person reviews them.

Usage::

    uv run python scripts/build_answer_references.py
"""

import argparse
import json

from meridian.common.llm_client import build_llm_client
from meridian.eval.datasets import ANSWER_SET_PATH, read_jsonl, write_jsonl
from meridian.retrieval.factory import build_retriever

SYSTEM_PROMPT = (
    "You write reference answers for evaluating a research assistant. Using only the numbered "
    "passages, answer the question in two to five sentences. If the passages do not answer it, "
    'reply with an empty answer. Reply with a JSON object {"answer": "...", "used_passages": '
    "[numbers of the passages you relied on]}."
)
TOP_K = 6


def draft_reference(llm, retriever, question: str) -> tuple[str, list[str], list[dict]]:
    """
    Return a drafted reference, the arXiv ids it used, and the spans of the passages it used.

    Spans (`chunk_id`, `arxiv_id`, `version`, `start`, `end`) locate the passages in the parsed
    papers without storing chunk text in the committed answer set.
    """
    result = retriever.search(question, mode="hybrid_rerank", k=TOP_K)
    passages = "\n\n".join(
        f"[{index}] {chunk.payload.get('title', '')}\n{chunk.text}"
        for index, chunk in enumerate(result.chunk_list, start=1)
    )
    reply = llm.chat_json(
        "question_writer",
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
        ],
    )
    used_chunk_list = [
        result.chunk_list[int(number) - 1]
        for number in reply.get("used_passages", [])
        if str(number).isdigit() and 1 <= int(number) <= len(result.chunk_list)
    ]
    span_list = [
        {
            "chunk_id": chunk.chunk_id,
            "arxiv_id": chunk.payload["arxiv_id"],
            "version": int(chunk.payload["version"]),
            "start": int(chunk.payload["char_start"]),
            "end": int(chunk.payload["char_end"]),
        }
        for chunk in used_chunk_list
    ]
    arxiv_id_list = sorted({chunk.payload["arxiv_id"] for chunk in used_chunk_list})
    return str(reply.get("answer", "")).strip(), arxiv_id_list, span_list


def main() -> None:
    """Fill missing references in place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=str(ANSWER_SET_PATH))
    args = parser.parse_args()
    record_list = read_jsonl(args.path)
    llm = build_llm_client()
    retriever = build_retriever()
    writer_model = llm.models_config.role("question_writer").model
    filled = empty = 0
    for record in record_list:
        if record.get("reference"):
            continue
        answer, arxiv_id_list, span_list = draft_reference(llm, retriever, record["question"])
        if not answer:
            empty += 1
            continue
        record["reference"] = answer
        record["reference_arxiv_ids"] = arxiv_id_list
        record["reference_spans"] = span_list
        record["reference_source"] = f"llm_draft:{writer_model}"
        record["reference_reviewed"] = False
        filled += 1
        write_jsonl(args.path, record_list)
    print(json.dumps({"filled": filled, "unanswerable_from_corpus": empty}))


if __name__ == "__main__":
    main()
