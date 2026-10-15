"""Loaders for the committed evaluation datasets under eval/: the retrieval set R, the answer
set A, and the out-of-scope set O, plus the paper-level exclusion list used by training.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from meridian.eval.spans import LabeledSpan

# ============= Constants =============

RETRIEVAL_SET_PATH = Path("eval/retrieval_set.jsonl")
ANSWER_SET_PATH = Path("eval/answer_set.jsonl")
OUT_OF_SCOPE_SET_PATH = Path("eval/out_of_scope.jsonl")
SPLIT_FAST = "fast"
SPLIT_FULL = "full"


# ============= JSONL =============


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON Lines file."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, record_list: list[dict]) -> None:
    """Write a JSON Lines file with one sorted-key object per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in record_list:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# ============= Retrieval set =============


@dataclass
class RetrievalItem:
    """One labeled retrieval query."""

    qid: str
    question: str
    span_list: list[LabeledSpan]
    o_in_fast: bool
    o_valid: bool = True
    record: dict = field(default_factory=dict)


def load_retrieval_set(
    path: str | Path = RETRIEVAL_SET_PATH, split: str = SPLIT_FULL
) -> list[RetrievalItem]:
    """
    Load R, keeping items not marked invalid by the manual review.

    Parameters
    ----------
    path : str or Path
        The JSONL file.
    split : str
        `fast` for R-fast (40 items) or `full` for R-full (all items).

    Returns
    -------
    item_list : list of RetrievalItem
        Items of the split whose review did not mark them invalid.
    """
    item_list: list[RetrievalItem] = []
    for record in read_jsonl(path):
        review_dict = record.get("review", {})
        item = RetrievalItem(
            qid=record["qid"],
            question=record["question"],
            span_list=[LabeledSpan.from_dict(span_dict) for span_dict in record["spans"]],
            o_in_fast=bool(record.get("in_fast", False)),
            o_valid=review_dict.get("valid") is not False,
            record=record,
        )
        if not item.o_valid:
            continue
        if split == SPLIT_FAST and not item.o_in_fast:
            continue
        item_list.append(item)
    return item_list


def count_invalid(path: str | Path = RETRIEVAL_SET_PATH) -> int:
    """Return how many items the manual review marked invalid."""
    return sum(1 for record in read_jsonl(path) if record.get("review", {}).get("valid") is False)


# ============= Answer and out-of-scope sets =============


def load_answer_set(path: str | Path = ANSWER_SET_PATH, split: str = SPLIT_FULL) -> list[dict]:
    """Load A, or A-fast when `split` is `fast`."""
    record_list = read_jsonl(path)
    if split == SPLIT_FAST:
        return [record for record in record_list if record.get("in_fast")]
    return record_list


def load_out_of_scope_set(path: str | Path = OUT_OF_SCOPE_SET_PATH) -> list[dict]:
    """Load O."""
    return read_jsonl(path)


# ============= Exclusion list =============


def evaluation_paper_ids(
    retrieval_path: str | Path = RETRIEVAL_SET_PATH,
    answer_path: str | Path = ANSWER_SET_PATH,
    out_of_scope_path: str | Path = OUT_OF_SCOPE_SET_PATH,
) -> set[str]:
    """
    Return the arXiv ids of every paper referenced by R, A, or O.

    R contributes the paper of every labeled span, including reviewer-added spans and items
    marked invalid; A contributes `reference_arxiv_ids`; O contributes any `arxiv_ids` it lists.
    Missing files raise, because an exclusion list built from absent sets would leak.
    """
    paper_set: set[str] = set()
    for record in read_jsonl(retrieval_path):
        paper_set.update(span_dict["arxiv_id"] for span_dict in record["spans"])
    for record in read_jsonl(answer_path):
        paper_set.update(record.get("reference_arxiv_ids", []))
    for record in read_jsonl(out_of_scope_path):
        paper_set.update(record.get("arxiv_ids", []))
    return paper_set
