"""Unit tests for the hybrid retriever modes, the Query API request, and the legacy RRF path."""

import pytest
from qdrant_client import QdrantClient, models

from meridian.common.config import get_index_config
from meridian.ingest import index as index_module
from meridian.retrieval.hybrid import (
    MODE_LIST,
    STAGE_LIST,
    HybridRetriever,
    hybrid_request,
    paper_filter_of,
)
from meridian.retrieval.legacy import LegacyRetriever, reciprocal_rank_fusion
from tests.fakes import FakeDenseEncoder, FakeReranker, FakeSparseEncoder

pytestmark = pytest.mark.unit

DOCUMENT_LIST = [
    ("a", "chain of thought prompting improves arithmetic reasoning"),
    ("a", "self consistency samples many reasoning paths and votes"),
    ("b", "calibration measures whether confidence matches accuracy"),
    ("b", "expected calibration error bins predictions by confidence"),
    ("c", "benchmarks such as gsm8k test grade school math"),
    ("c", "retrieval augmented generation grounds answers in documents"),
]


@pytest.fixture
def populated_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    index_module.create_index_collection(client, "meridian_v1", 32, {})
    payload_list = [
        {
            "chunk_id": f"00000000-0000-0000-0000-{index:012d}",
            "arxiv_id": arxiv_id,
            "text": text,
        }
        for index, (arxiv_id, text) in enumerate(DOCUMENT_LIST)
    ]
    text_list = [payload["text"] for payload in payload_list]
    point_list = index_module.build_points(
        payload_list,
        FakeDenseEncoder().encode_documents(text_list),
        FakeSparseEncoder().encode_documents(text_list),
    )
    index_module.upsert_points(client, "meridian_v1", point_list, batch_size=10)
    index_module.switch_alias(client, "meridian_live", "meridian_v1")
    return client


def _retriever(client) -> HybridRetriever:
    return HybridRetriever(
        client,
        "meridian_live",
        FakeDenseEncoder(),
        FakeSparseEncoder(),
        FakeReranker(),
        get_index_config().retrieval,
    )


def test_hybrid_request_has_two_prefetches_and_rrf():
    request = hybrid_request(
        [0.1, 0.2],
        models.SparseVector(indices=[1], values=[1.0]),
        prefetch_limit=50,
        fusion_limit=30,
        rrf_k=60,
        query_filter=None,
    )
    using_list = [prefetch.using for prefetch in request["prefetch"]]
    assert using_list == ["dense", "sparse"]
    assert all(prefetch.limit == 50 for prefetch in request["prefetch"])
    assert request["query"].rrf.k == 60
    assert request["limit"] == 30


def test_every_mode_returns_ranked_chunks_and_timings(populated_client):
    retriever = _retriever(populated_client)
    for mode in MODE_LIST:
        result = retriever.search("expected calibration error confidence", mode=mode, k=3)
        assert 0 < len(result.chunk_list) <= 3
        assert set(result.timing_dict) == set(STAGE_LIST)
        assert result.chunk_list[0].payload["arxiv_id"] == "b"
    reranked = retriever.search("expected calibration error", mode="hybrid_rerank", k=2)
    assert all(chunk.rerank_score is not None for chunk in reranked.chunk_list)
    score_list = [chunk.rerank_score for chunk in reranked.chunk_list]
    assert score_list == sorted(score_list, reverse=True)


def test_paper_filter_restricts_results(populated_client):
    retriever = _retriever(populated_client)
    result = retriever.search("reasoning", mode="hybrid", k=6, paper_filter=["c"])
    assert {chunk.payload["arxiv_id"] for chunk in result.chunk_list} == {"c"}
    assert paper_filter_of(None) is None


def test_unknown_mode_is_rejected(populated_client):
    with pytest.raises(ValueError):
        _retriever(populated_client).search("x", mode="bogus")


def test_rrf_formula_and_order():
    fused_list = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60, top_k=10)
    assert fused_list[0][0] == "b"
    assert fused_list[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert [item_id for item_id, _score in fused_list] == ["b", "a", "c"]
    assert len(reciprocal_rank_fusion([[str(n) for n in range(30)]], k=60, top_k=5)) == 5


def test_python_rrf_order_matches_qdrant_rrf(populated_client):
    query = "reasoning paths chain of thought"
    dense_vector = FakeDenseEncoder().encode_query(query)
    sparse_vector = FakeSparseEncoder().encode_query(query)
    dense_ids = [
        str(point.id)
        for point in populated_client.query_points(
            "meridian_v1", query=dense_vector, using="dense", limit=6
        ).points
    ]
    sparse_ids = [
        str(point.id)
        for point in populated_client.query_points(
            "meridian_v1", query=sparse_vector, using="sparse", limit=6
        ).points
    ]
    python_order = [
        item_id for item_id, _ in reciprocal_rank_fusion([dense_ids, sparse_ids], 60, 3)
    ]
    qdrant_order = [
        str(point.id)
        for point in populated_client.query_points(
            "meridian_v1",
            **hybrid_request(dense_vector, sparse_vector, 6, 3, 60, None),
        ).points
    ]
    assert python_order == qdrant_order


def test_legacy_retriever_runs_end_to_end(populated_client):
    legacy = LegacyRetriever(populated_client, "meridian_v1", FakeDenseEncoder(), FakeReranker())
    result = legacy.search("grade school math benchmarks", k=2)
    assert result.mode == "legacy"
    assert result.chunk_list[0].payload["arxiv_id"] == "c"
