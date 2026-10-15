"""Unit tests for span coverage and the construction of the retrieval set R."""

import json

import pytest

from meridian.common.config import get_models_config, load_config
from meridian.common.llm_client import LLMClient, MemoryResponseCache
from meridian.eval.build_retrieval_set import (
    DEFAULT_CONFIG_PATH,
    RetrievalSetConfig,
    build_retrieval_set,
    stratified_order,
    token_overlap,
)
from meridian.eval.datasets import (
    evaluation_paper_ids,
    load_retrieval_set,
    write_jsonl,
)
from meridian.eval.spans import LabeledSpan, covered_span_indices, span_coverage
from tests.fakes import ScriptedTransport

pytestmark = pytest.mark.unit


def _payload(arxiv_id: str, section_index: int, chunk_index: int, start: int = 0) -> dict:
    return {
        "chunk_id": f"{arxiv_id}-{section_index}-{chunk_index}",
        "arxiv_id": arxiv_id,
        "version": 1,
        "section_index": section_index,
        "section_title": f"{section_index} Section",
        "char_start": start,
        "char_end": start + 1000,
        "token_count": 300,
        "parser_version": 1,
        "text": f"zeta{section_index} omega{chunk_index} lambda results measured carefully",
    }


def test_span_coverage_rule():
    span = LabeledSpan("p", 1, 100, 200)
    assert (
        span_coverage({"arxiv_id": "p", "version": 1, "char_start": 150, "char_end": 400}, span)
        == 0.5
    )
    assert (
        span_coverage({"arxiv_id": "p", "version": 1, "char_start": 0, "char_end": 149}, span) < 0.5
    )
    assert (
        span_coverage({"arxiv_id": "p", "version": 2, "char_start": 0, "char_end": 999}, span) == 0
    )
    assert (
        span_coverage({"arxiv_id": "q", "version": 1, "char_start": 0, "char_end": 999}, span) == 0
    )
    payload = {"arxiv_id": "p", "version": 1, "char_start": 90, "char_end": 260}
    assert covered_span_indices(payload, [span, LabeledSpan("p", 1, 250, 600)]) == {0}


def test_token_overlap_ignores_stopwords():
    chunk = "Self-consistency samples diverse reasoning paths and takes a majority vote."
    assert token_overlap("What is the role of the majority vote?", chunk) == pytest.approx(2 / 3)
    assert token_overlap("Which aggregation picks the most frequent answer?", chunk) == 0.0
    assert token_overlap("what is it?", chunk) == 1.0


def test_stratified_order_visits_every_paper_before_repeating():
    payload_list = [
        _payload(paper, section, chunk)
        for paper in ("a", "b", "c")
        for section in range(3)
        for chunk in range(2)
    ]
    ordered_list = stratified_order(payload_list, seed=1)
    assert len(ordered_list) == len(payload_list)
    assert {payload["arxiv_id"] for payload in ordered_list[:3]} == {"a", "b", "c"}
    first_paper = ordered_list[0]["arxiv_id"]
    first_paper_sections = [
        payload["section_index"] for payload in ordered_list if payload["arxiv_id"] == first_paper
    ][:3]
    assert sorted(first_paper_sections) == [0, 1, 2]
    assert stratified_order(payload_list, seed=1) == ordered_list


def _config(**update) -> RetrievalSetConfig:
    return load_config(RetrievalSetConfig, DEFAULT_CONFIG_PATH).model_copy(update=update)


def test_build_drops_high_overlap_and_retries():
    def responder(model, messages, params):
        system = messages[0]["content"]
        # first attempt copies the passage, the retry paraphrases it
        if "previous question reused" in system:
            return json.dumps({"question": "Which quantity was evaluated with care?"})
        passage = messages[1]["content"].split("\n", 1)[1]
        return json.dumps({"question": passage})

    llm = LLMClient(get_models_config(), ScriptedTransport(responder), MemoryResponseCache())
    payload_list = [_payload(paper, section, 0) for paper in "abcdef" for section in range(2)]
    record_list, stats = build_retrieval_set(payload_list, llm, _config(n_items=6, n_fast=2))
    assert len(record_list) == 6
    assert stats["dropped_overlap"] == 6
    assert stats["distinct_papers"] == 6
    assert sum(record["in_fast"] for record in record_list) == 2
    assert all(record["token_overlap"] <= 0.5 for record in record_list)
    first = record_list[0]
    assert first["spans"][0]["end"] - first["spans"][0]["start"] == 1000
    assert first["review"]["valid"] is None


def test_build_fails_loudly_when_too_few_pass():
    llm = LLMClient(
        get_models_config(),
        ScriptedTransport(lambda *_: json.dumps({"question": "zeta0 omega0 lambda results"})),
        MemoryResponseCache(),
    )
    with pytest.raises(RuntimeError, match="passed the overlap filter"):
        build_retrieval_set([_payload("a", 0, 0)], llm, _config(n_items=2, n_fast=1))


def test_loader_drops_invalid_items_and_splits(tmp_path):
    record_list = [
        {
            "qid": f"r{index}",
            "question": "q",
            "spans": [{"arxiv_id": f"p{index}", "version": 1, "start": 0, "end": 10}],
            "in_fast": index < 2,
            "review": {"valid": False if index == 3 else None},
        }
        for index in range(5)
    ]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, record_list)
    assert len(load_retrieval_set(path, "full")) == 4
    assert len(load_retrieval_set(path, "fast")) == 2
    answer_path = tmp_path / "a.jsonl"
    write_jsonl(answer_path, [{"question": "x", "reference_arxiv_ids": ["z1"]}])
    oos_path = tmp_path / "o.jsonl"
    write_jsonl(oos_path, [{"question": "y"}])
    paper_set = evaluation_paper_ids(path, answer_path, oos_path)
    # invalid items still contribute their papers to the exclusion list
    assert paper_set == {"p0", "p1", "p2", "p3", "p4", "z1"}
