"""Unit tests for the span-based ranking metrics and the shared evaluation core."""

import math

import pytest

from meridian.common.config import get_index_config
from meridian.eval.datasets import RetrievalItem
from meridian.eval.metrics import mean_metrics, mrr_at_k, ndcg_at_k, query_metrics, recall_at_k
from meridian.eval.retrieval_eval import evaluate_search, scaled_chunker
from meridian.eval.spans import LabeledSpan
from meridian.retrieval.hybrid import STAGE_LIST, RetrievalResult, RetrievedChunk

pytestmark = pytest.mark.unit


def _payload(start: int, end: int, arxiv_id: str = "p") -> dict:
    return {"arxiv_id": arxiv_id, "version": 1, "char_start": start, "char_end": end}


def test_recall_counts_each_span_once():
    covered_list = [{0}, {0}, set(), {1}]
    assert recall_at_k(covered_list, 2, 2) == 0.5
    assert recall_at_k(covered_list, 2, 4) == 1.0
    assert recall_at_k([], 1, 5) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k([set(), set(), {0}], 10) == pytest.approx(1 / 3)
    assert mrr_at_k([set(), set(), {0}], 2) == 0.0


def test_ndcg_hand_computed():
    # spans 0 and 1; rank 1 misses, rank 2 hits span 0, rank 3 repeats it, rank 4 hits span 1
    covered_list = [set(), {0}, {0}, {1}]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(covered_list, 2, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([{0}], 1, 10) == pytest.approx(1.0)


def test_query_metrics_with_spans():
    span_list = [LabeledSpan("p", 1, 100, 200)]
    payload_list = [_payload(0, 120), _payload(0, 50, "q"), _payload(110, 300)] + [
        _payload(900, 950)
    ] * 7
    metrics = query_metrics(payload_list, span_list)
    # rank 1 covers 20% (not relevant), rank 3 covers 90%
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == pytest.approx(1 / 3)
    assert metrics["ndcg_at_10"] == pytest.approx(1 / math.log2(4))
    assert mean_metrics([metrics, {**metrics, "recall_at_5": 0.0}])["recall_at_5"] == 0.5


def test_evaluate_search_aggregates_and_skips_warmup_latency():
    item_list = [
        RetrievalItem(
            qid="r1", question="q1", span_list=[LabeledSpan("p", 1, 0, 100)], o_in_fast=True
        ),
        RetrievalItem(
            qid="r2", question="q2", span_list=[LabeledSpan("p", 1, 500, 600)], o_in_fast=False
        ),
    ]
    call_list: list[str] = []

    def search(query: str) -> RetrievalResult:
        call_list.append(query)
        timing = dict.fromkeys(STAGE_LIST, 0.01)
        return RetrievalResult(
            query=query,
            mode="hybrid",
            chunk_list=[RetrievedChunk("c", _payload(0, 100), 1.0)],
            timing_dict=timing,
        )

    summary = evaluate_search(search, item_list, warmup_queries=1, o_keep_per_query=True)
    assert call_list == ["q1", "q1", "q2"]
    assert summary["n_queries"] == 2
    assert summary["recall_at_5"] == 0.5
    assert summary["latency"]["total"]["count"] == 2
    assert [row["qid"] for row in summary["per_query"]] == ["r1", "r2"]


def test_scaled_chunker_keeps_overlap_proportional():
    chunker = scaled_chunker(get_index_config().chunker, 200, 0.15)
    assert chunker.chunk_tokens == 200
    assert chunker.overlap_tokens == 30
