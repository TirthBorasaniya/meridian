"""Unit tests for the private review sheets: passages, counts, and the missing-data errors."""

import subprocess

import pytest

from meridian.eval.datasets import write_jsonl
from meridian.eval.review_prep import (
    MissingSourceError,
    passage_text,
    quote,
    sample_indices,
    write_sheets,
)

pytestmark = pytest.mark.unit

PAPER_TEXT = "Intro text. " + "Self-consistency votes over sampled paths. " * 3 + "End of paper."


@pytest.fixture
def parsed_dir(tmp_path):
    directory = tmp_path / "parsed"
    directory.mkdir()
    (directory / "2203.11171v4.md").write_text(PAPER_TEXT)
    return directory


def _retrieval_records(count: int) -> list[dict]:
    return [
        {
            "qid": f"r{index:03d}",
            "question": f"question {index}?",
            "section_title": "2 Method",
            "spans": [{"arxiv_id": "2203.11171", "version": 4, "start": 12, "end": 54}],
        }
        for index in range(count)
    ]


def _answers() -> list[dict]:
    span = {"chunk_id": "c1", "arxiv_id": "2203.11171", "version": 4, "start": 12, "end": 54}
    return [
        {
            "aid": "a001",
            "question": "What votes?",
            "reference": "Paths vote.",
            "reference_spans": [span],
        },
        {"aid": "a002", "question": "Unanswerable?", "reference": "", "reference_spans": []},
        {"aid": "a003", "question": "Second?", "reference": "Another.", "reference_spans": [span]},
    ]


def test_passage_text_and_quote(parsed_dir):
    assert passage_text(parsed_dir, "2203.11171", 4, 12, 54) == PAPER_TEXT[12:54]
    assert quote("a\n\nb") == "> a\n>\n> b"
    with pytest.raises(MissingSourceError, match="parsed paper"):
        passage_text(parsed_dir, "9999.00000", 1, 0, 5)


def test_sheets_have_the_requested_fields_and_counts(tmp_path, parsed_dir):
    retrieval_path = tmp_path / "r.jsonl"
    answer_path = tmp_path / "a.jsonl"
    write_jsonl(retrieval_path, _retrieval_records(40))
    write_jsonl(answer_path, _answers())
    review_dir = tmp_path / "review"
    counts = write_sheets(retrieval_path, answer_path, 29, 30, review_dir, parsed_dir)
    assert (counts["retrieval_items"], counts["reference_items"]) == (30, 2)

    retrieval_text = (review_dir / "retrieval_review.md").read_text()
    assert retrieval_text.count("VALID: yes/no") == 30
    assert retrieval_text.count("NOTES:") == 30
    assert PAPER_TEXT[12:54] in retrieval_text
    assert "**Question:**" in retrieval_text and "r0" in retrieval_text

    reference_text = (review_dir / "reference_review.md").read_text()
    assert reference_text.count("CORRECT: yes/no") == 2
    assert reference_text.count("FIX:") == 2
    assert "Paths vote." in reference_text and PAPER_TEXT[12:54] in reference_text
    # the undrafted item is listed, not silently dropped
    assert "Items without a drafted reference" in reference_text and "a002" in reference_text


def test_sample_matches_the_interactive_review_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "review_script", "scripts/review_retrieval_set.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert sample_indices(150, 30, 29) == module.sample_indices(150, 30, 29)


def test_missing_sources_fail_clearly(tmp_path, parsed_dir):
    with pytest.raises(MissingSourceError, match="does not exist yet"):
        write_sheets(tmp_path / "none.jsonl", tmp_path / "none2.jsonl", 1, 30, tmp_path, parsed_dir)
    retrieval_path = tmp_path / "r.jsonl"
    answer_path = tmp_path / "a.jsonl"
    write_jsonl(retrieval_path, _retrieval_records(5))
    write_jsonl(answer_path, [{"aid": "a1", "question": "q", "reference": ""}])
    with pytest.raises(MissingSourceError, match="no drafted references"):
        write_sheets(retrieval_path, answer_path, 1, 30, tmp_path / "out", parsed_dir)


def test_review_directory_is_never_tracked():
    ignored = subprocess.run(
        ["git", "check-ignore", ".planning/review/retrieval_review.md"], capture_output=True
    )
    assert ignored.returncode == 0
