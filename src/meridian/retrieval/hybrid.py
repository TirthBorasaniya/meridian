"""Hybrid retrieval through Qdrant's Query API: dense and sparse prefetches fused with RRF, then
cross-encoder reranking of the fused top candidates.

Modes `dense`, `sparse`, `hybrid`, and `hybrid_rerank` share one code path so that E1 compares
configurations on identical inputs, and every search reports latency per stage.
"""

import time
from dataclasses import dataclass, field
from typing import Protocol

from qdrant_client import QdrantClient, models

from meridian.common.config import RetrievalConfig
from meridian.ingest.index import DENSE_VECTOR, SPARSE_VECTOR

# ============= Constants =============

MODE_DENSE = "dense"
MODE_SPARSE = "sparse"
MODE_HYBRID = "hybrid"
MODE_HYBRID_RERANK = "hybrid_rerank"
MODE_LIST = (MODE_DENSE, MODE_SPARSE, MODE_HYBRID, MODE_HYBRID_RERANK)
STAGE_LIST = ("embed_dense", "embed_sparse", "qdrant", "rerank", "total")


# ============= Protocols =============


class QueryDenseEncoder(Protocol):
    """Anything that embeds one query into a dense vector."""

    def encode_query(self, query: str) -> list[float]: ...


class QuerySparseEncoder(Protocol):
    """Anything that embeds one query into a Qdrant sparse vector."""

    def encode_query(self, query: str) -> models.SparseVector: ...


class PassageScorer(Protocol):
    """Anything that scores passages against a query."""

    def score(self, query: str, text_list: list[str]) -> list[float]: ...


# ============= Results =============


@dataclass
class RetrievedChunk:
    """One retrieved chunk with its first-stage and rerank scores."""

    chunk_id: str
    payload: dict
    score: float
    rerank_score: float | None = None

    @property
    def text(self) -> str:
        """Chunk text."""
        return self.payload.get("text", "")


@dataclass
class RetrievalResult:
    """Ranked chunks and per-stage latency in seconds."""

    query: str
    mode: str
    chunk_list: list[RetrievedChunk]
    timing_dict: dict[str, float] = field(default_factory=dict)


# ============= Request construction =============


def paper_filter_of(arxiv_id_list: list[str] | None) -> models.Filter | None:
    """Return a filter restricting results to the given papers, or None."""
    if not arxiv_id_list:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="arxiv_id", match=models.MatchAny(any=arxiv_id_list))]
    )


def hybrid_request(
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    prefetch_limit: int,
    fusion_limit: int,
    rrf_k: int,
    query_filter: models.Filter | None,
) -> dict:
    """
    Return `query_points` keyword arguments for two prefetches fused with RRF.

    Parameters
    ----------
    dense_vector : list of float
        Query embedding for the `dense` vector.
    sparse_vector : SparseVector
        Query terms for the `sparse` vector.
    prefetch_limit : int
        Candidates per prefetch.
    fusion_limit : int
        Fused candidates returned.
    rrf_k : int
        RRF constant.
    query_filter : Filter or None
        Payload filter applied inside both prefetches.

    Returns
    -------
    request : dict
        Keyword arguments for `QdrantClient.query_points`, without the collection name.
    """
    return {
        "prefetch": [
            models.Prefetch(
                query=dense_vector, using=DENSE_VECTOR, limit=prefetch_limit, filter=query_filter
            ),
            models.Prefetch(
                query=sparse_vector, using=SPARSE_VECTOR, limit=prefetch_limit, filter=query_filter
            ),
        ],
        "query": models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
        "limit": fusion_limit,
        "with_payload": True,
    }


# ============= Retriever =============


class HybridRetriever:
    """Search one collection or alias in any of the four retrieval modes."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense_encoder: QueryDenseEncoder,
        sparse_encoder: QuerySparseEncoder,
        reranker: PassageScorer | None,
        retrieval_config: RetrievalConfig,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.reranker = reranker
        self.config = retrieval_config

    def search(
        self,
        query: str,
        mode: str = MODE_HYBRID_RERANK,
        k: int | None = None,
        paper_filter: list[str] | None = None,
        prefetch_limit: int | None = None,
        fusion_limit: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve the top `k` chunks for a query.

        Parameters
        ----------
        query : str
            Standalone search query.
        mode : str
            One of `dense`, `sparse`, `hybrid`, `hybrid_rerank`.
        k : int or None
            Number of chunks returned; defaults to `final_k`.
        paper_filter : list of str or None
            Restrict to these arXiv ids.
        prefetch_limit : int or None
            Override of the per-prefetch candidate count, used by sensitivity runs.
        fusion_limit : int or None
            Override of the fused candidate count (the rerank depth in `hybrid_rerank`).

        Returns
        -------
        result : RetrievalResult
            Ranked chunks and stage latencies.
        """
        if mode not in MODE_LIST:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        top_k = k or self.config.final_k
        prefetch = prefetch_limit or self.config.prefetch_limit
        fused = fusion_limit or self.config.rerank_depth
        query_filter = paper_filter_of(paper_filter)
        timing_dict = dict.fromkeys(STAGE_LIST, 0.0)
        total_start = time.perf_counter()

        dense_vector: list[float] | None = None
        sparse_vector: models.SparseVector | None = None
        if mode != MODE_SPARSE:
            stage_start = time.perf_counter()
            dense_vector = self.dense_encoder.encode_query(query)
            timing_dict["embed_dense"] = time.perf_counter() - stage_start
        if mode != MODE_DENSE:
            stage_start = time.perf_counter()
            sparse_vector = self.sparse_encoder.encode_query(query)
            timing_dict["embed_sparse"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        if mode == MODE_DENSE:
            response = self.client.query_points(
                self.collection,
                query=dense_vector,
                using=DENSE_VECTOR,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        elif mode == MODE_SPARSE:
            response = self.client.query_points(
                self.collection,
                query=sparse_vector,
                using=SPARSE_VECTOR,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        else:
            limit = top_k if mode == MODE_HYBRID else max(fused, top_k)
            response = self.client.query_points(
                self.collection,
                **hybrid_request(
                    dense_vector,  # type: ignore[arg-type]
                    sparse_vector,  # type: ignore[arg-type]
                    prefetch,
                    limit,
                    self.config.rrf_k,
                    query_filter,
                ),
            )
        timing_dict["qdrant"] = time.perf_counter() - stage_start

        chunk_list = [
            RetrievedChunk(chunk_id=str(point.id), payload=point.payload or {}, score=point.score)
            for point in response.points
        ]
        if mode == MODE_HYBRID_RERANK:
            stage_start = time.perf_counter()
            chunk_list = self.rerank(query, chunk_list[:fused], top_k)
            timing_dict["rerank"] = time.perf_counter() - stage_start
        timing_dict["total"] = time.perf_counter() - total_start
        return RetrievalResult(
            query=query, mode=mode, chunk_list=chunk_list, timing_dict=timing_dict
        )

    def rerank(
        self, query: str, chunk_list: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Score candidates with the cross-encoder and keep the top `top_k`."""
        if self.reranker is None:
            raise ValueError("hybrid_rerank requires a reranker")
        score_list = self.reranker.score(query, [chunk.text for chunk in chunk_list])
        for chunk, score in zip(chunk_list, score_list, strict=True):
            chunk.rerank_score = score
        ranked_list = sorted(chunk_list, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
        return ranked_list[:top_k]
