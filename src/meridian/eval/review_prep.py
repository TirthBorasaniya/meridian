"""Private review sheets for the two manual reviews: the 30 sampled retrieval items and every
LLM-drafted reference answer. Passages are resolved from the parsed papers under `data/parsed`,
so the sheets hold paper text and must stay out of version control (they are written under the
gitignored `.planning/review/`).
"""

import json
from pathlib import Path

from meridian.eval.datasets import read_jsonl

# ============= Constants =============

REVIEW_DIR = Path(".planning/review")
PARSED_DIR = Path("data/parsed")
RETRIEVAL_FILE = "retrieval_review.md"
REFERENCE_FILE = "reference_review.md"
SEPARATOR = "\n---\n\n"


class MissingSourceError(Exception):
    """A dataset or parsed paper needed for a review sheet is absent."""


# ============= Passages =============


def passage_text(parsed_dir: Path, arxiv_id: str, version: int, start: int, end: int) -> str:
    """Return `parsed_text[start:end]` of one paper version."""
    path = parsed_dir / f"{arxiv_id}v{version}.md"
    if not path.exists():
        raise MissingSourceError(f"parsed paper {path} is missing; run the backfill first")
    return path.read_text(encoding="utf-8")[start:end]


def span_text(parsed_dir: Path, span: dict) -> str:
    """Return the passage of a span record with `arxiv_id`, `version`, `start`, and `end`."""
    return passage_text(
        parsed_dir, span["arxiv_id"], int(span["version"]), int(span["start"]), int(span["end"])
    )


def quote(text: str) -> str:
    """Return text as a Markdown block quote."""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


# ============= Retrieval sheet =============


def sample_indices(record_count: int, sample_size: int, seed: int) -> list[int]:
    """Return the same seeded sample as `scripts/review_retrieval_set.py`."""
    import random

    return sorted(random.Random(seed).sample(range(record_count), min(sample_size, record_count)))


def retrieval_sheet(
    record_list: list[dict], index_list: list[int], parsed_dir: Path = PARSED_DIR
) -> str:
    """
    Render the retrieval review sheet.

    Parameters
    ----------
    record_list : list of dict
        Every item of R.
    index_list : list of int
        Positions of the sampled items.
    parsed_dir : Path
        Directory of parsed papers.

    Returns
    -------
    sheet : str
        Markdown with, per item, the id, question, full source passage, and the two lines to fill.
    """
    part_list = [
        f"# Retrieval set review\n\n{len(index_list)} items. For each, decide whether the question "
        "is valid and is answered by the source passage below it, then fill in the two lines.\n"
    ]
    for number, index in enumerate(index_list, start=1):
        record = record_list[index]
        span = record["spans"][0]
        text = span_text(parsed_dir, span)
        part_list.append(
            f"## {number}. {record['qid']}\n\n"
            f"**Question:** {record['question']}\n\n"
            f"**Source:** {span['arxiv_id']}v{span['version']}, section "
            f'"{record.get("section_title", "")}", characters {span["start"]} to {span["end"]}\n\n'
            f"**Source passage:**\n\n{quote(text)}\n\n"
            "VALID: yes/no\n\nNOTES:\n"
        )
    return SEPARATOR.join(part_list)


# ============= Reference sheet =============


def reference_sheet(record_list: list[dict], parsed_dir: Path = PARSED_DIR) -> str:
    """
    Render the reference review sheet for every drafted reference.

    Items without a reference (the drafter found no support) are listed at the end, so they are
    not silently skipped.
    """
    drafted_list = [record for record in record_list if record.get("reference")]
    undrafted_list = [record for record in record_list if not record.get("reference")]
    part_list = [
        f"# Reference answer review\n\n{len(drafted_list)} drafted references. Check each against "
        "the passages it relies on, then fill in the two lines.\n"
    ]
    for number, record in enumerate(drafted_list, start=1):
        passage_block_list: list[str] = []
        for span in record.get("reference_spans", []):
            text = passage_text(
                parsed_dir,
                span["arxiv_id"],
                int(span["version"]),
                int(span["start"]),
                int(span["end"]),
            )
            passage_block_list.append(
                f"{span['arxiv_id']}v{span['version']}, characters {span['start']} to "
                f"{span['end']}:\n\n{quote(text)}"
            )
        passages = "\n\n".join(passage_block_list) or "(no passages were recorded for this item)"
        part_list.append(
            f"## {number}. {record['aid']}\n\n"
            f"**Question:** {record['question']}\n\n"
            f"**Reference answer:** {record['reference']}\n\n"
            f"**Passages it relies on:**\n\n{passages}\n\n"
            "CORRECT: yes/no\n\nFIX:\n"
        )
    if undrafted_list:
        part_list.append(
            "## Items without a drafted reference\n\n"
            + "\n".join(f"- {record['aid']}: {record['question']}" for record in undrafted_list)
            + "\n"
        )
    return SEPARATOR.join(part_list)


# ============= Entry point =============


def write_sheets(
    retrieval_path: Path,
    answer_path: Path,
    review_seed: int,
    sample_size: int,
    review_dir: Path = REVIEW_DIR,
    parsed_dir: Path = PARSED_DIR,
) -> dict:
    """Write both sheets and return the item counts."""
    for path in (retrieval_path, answer_path):
        if not path.exists():
            raise MissingSourceError(f"{path} does not exist yet")
    retrieval_list = read_jsonl(retrieval_path)
    answer_list = read_jsonl(answer_path)
    if not any(record.get("reference") for record in answer_list):
        raise MissingSourceError(
            "no drafted references yet; run scripts/build_answer_references.py"
        )
    index_list = sample_indices(len(retrieval_list), sample_size, review_seed)
    review_dir.mkdir(parents=True, exist_ok=True)
    retrieval_text = retrieval_sheet(retrieval_list, index_list, parsed_dir)
    reference_text = reference_sheet(answer_list, parsed_dir)
    (review_dir / RETRIEVAL_FILE).write_text(retrieval_text, encoding="utf-8")
    (review_dir / REFERENCE_FILE).write_text(reference_text, encoding="utf-8")
    return {
        "retrieval_items": len(index_list),
        "reference_items": sum(1 for record in answer_list if record.get("reference")),
        "retrieval_path": str(review_dir / RETRIEVAL_FILE),
        "reference_path": str(review_dir / REFERENCE_FILE),
    }


def main() -> None:
    """Write the sheets from the committed datasets."""
    from meridian.common.config import load_config
    from meridian.eval.build_retrieval_set import DEFAULT_CONFIG_PATH, RetrievalSetConfig
    from meridian.eval.datasets import ANSWER_SET_PATH, RETRIEVAL_SET_PATH

    config = load_config(RetrievalSetConfig, DEFAULT_CONFIG_PATH)
    print(
        json.dumps(
            write_sheets(
                RETRIEVAL_SET_PATH, ANSWER_SET_PATH, config.review_seed, config.review_sample_size
            ),
            indent=2,
        )
    )
