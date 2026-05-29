"""Unit tests for the retrieval layer: RRF fusion, BM25 sparse retrieval, and
cross-encoder reranking and filtering."""

from meridian.retrieval.fusion import reciprocal_rank_fusion
from meridian.retrieval.reranker import Reranker
from meridian.retrieval.sparse import SparseRetriever


def _doc(chunk_id: str, text: str = "") -> dict:
    """Build a minimal candidate record."""
    return {"chunk_id": chunk_id, "text": text, "payload": {}}


def test_rrf_dedups_by_chunk_id():
    dense_list = [_doc("a"), _doc("b")]
    sparse_list = [_doc("b"), _doc("c")]
    fused_list = reciprocal_rank_fusion([dense_list, sparse_list], k=60, top_k=10)
    id_list = sorted(doc["chunk_id"] for doc in fused_list)
    assert id_list == ["a", "b", "c"]


def test_rrf_ranks_shared_doc_first():
    # "b" appears in both lists and should accumulate the highest fused score.
    dense_list = [_doc("a"), _doc("b")]
    sparse_list = [_doc("b"), _doc("c")]
    fused_list = reciprocal_rank_fusion([dense_list, sparse_list], k=60, top_k=10)
    assert fused_list[0]["chunk_id"] == "b"
    assert "rrf_score" in fused_list[0]


def test_rrf_score_formula():
    fused_list = reciprocal_rank_fusion([[_doc("a")]], k=60, top_k=10)
    # A single list with the document at rank 1 yields 1 / (60 + 1).
    assert abs(fused_list[0]["rrf_score"] - 1.0 / 61.0) < 1e-9


def test_rrf_top_k_truncates():
    dense_list = [_doc(str(index)) for index in range(30)]
    fused_list = reciprocal_rank_fusion([dense_list], top_k=5)
    assert len(fused_list) == 5


def test_sparse_retriever_ranks_relevant_chunk():
    payload_list = [
        {"chunk_id": "p0", "text": "chain of thought prompting improves reasoning"},
        {"chunk_id": "p1", "text": "the weather today is sunny and warm"},
        {"chunk_id": "p2", "text": "reasoning benchmarks evaluate arithmetic ability"},
    ]
    retriever = SparseRetriever(payload_list)
    result_list = retriever.retrieve("reasoning", top_k=2)
    assert len(result_list) == 2
    assert result_list[0]["chunk_id"] in {"p0", "p2"}
    assert result_list[0]["source"] == "sparse"


def test_sparse_retriever_empty_corpus():
    retriever = SparseRetriever([])
    assert retriever.retrieve("anything") == []


class _FakeCrossEncoder:
    """Stand-in cross-encoder returning fixed scores in input order."""

    def __init__(self, score_list: list[float]) -> None:
        self._score_list = score_list

    def predict(self, pair_list):
        return self._score_list


def _make_reranker(score_list: list[float]) -> Reranker:
    """Build a Reranker bypassing model loading, with a fake scorer."""
    reranker = object.__new__(Reranker)
    reranker._model = _FakeCrossEncoder(score_list)
    return reranker


def test_reranker_filters_below_threshold():
    doc_list = [_doc("a", "x"), _doc("b", "y"), _doc("c", "z")]
    reranker = _make_reranker([0.9, 0.3, 0.7])
    survivor_list = reranker.rerank("q", doc_list, top_k=5, threshold=0.5)
    assert [doc["chunk_id"] for doc in survivor_list] == ["a", "c"]
    assert all(doc["rerank_score"] >= 0.5 for doc in survivor_list)


def test_reranker_all_filtered_returns_empty():
    reranker = _make_reranker([0.1])
    assert reranker.rerank("q", [_doc("a", "x")], threshold=0.5) == []


def test_reranker_top_k_truncates():
    doc_list = [_doc("a"), _doc("b"), _doc("c")]
    reranker = _make_reranker([0.9, 0.8, 0.7])
    survivor_list = reranker.rerank("q", doc_list, top_k=2, threshold=0.5)
    assert len(survivor_list) == 2
