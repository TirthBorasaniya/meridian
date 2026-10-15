"""The previous retrieval path, kept only as the E1 baseline: dense top-20 from Qdrant, in-process
`rank_bm25` top-20 over a full scroll of the collection, Python RRF with k=60, and cross-encoder
reranking of the fused top-20.
"""

import re
import time

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from meridian.ingest.index import DENSE_VECTOR, iter_payloads
from meridian.retrieval.hybrid import (
    PassageScorer,
    QueryDenseEncoder,
    RetrievalResult,
    RetrievedChunk,
)

# ============= Constants =============

LEGACY_DENSE_TOP_K = 20
LEGACY_SPARSE_TOP_K = 20
LEGACY_FUSED_TOP_K = 20
LEGACY_RRF_K = 60
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization used by the previous BM25 index."""
    return TOKEN_PATTERN.findall(text.lower())


def reciprocal_rank_fusion(
    ranked_list_of_lists: list[list[str]], k: int, top_k: int
) -> list[tuple[str, float]]:
    """
    Fuse ranked id lists with Reciprocal Rank Fusion, `sum 1 / (k + rank)` with 1-based ranks.

    Parameters
    ----------
    ranked_list_of_lists : list of list of str
        Each inner list holds ids, best first.
    k : int
        RRF constant.
    top_k : int
        Number of fused ids returned.

    Returns
    -------
    fused_list : list of tuple of (str, float)
        (id, fused score) ordered by descending score; ties keep first-seen order.
    """
    score_dict: dict[str, float] = {}
    for ranked_list in ranked_list_of_lists:
        for position, item_id in enumerate(ranked_list):
            score_dict[item_id] = score_dict.get(item_id, 0.0) + 1.0 / (k + position + 1)
    ordered_list = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
    return ordered_list[:top_k]


class LegacyRetriever:
    """Dense plus in-process BM25 retrieval with Python RRF, as in the previous system."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense_encoder: QueryDenseEncoder,
        reranker: PassageScorer,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dense_encoder = dense_encoder
        self.reranker = reranker
        self._payload_dict: dict[str, dict] = {}
        id_list: list[str] = []
        token_list_of_lists: list[list[str]] = []
        for record in iter_payloads(client, collection):
            chunk_id = str(record.id)
            self._payload_dict[chunk_id] = record.payload or {}
            id_list.append(chunk_id)
            token_list_of_lists.append(tokenize(self._payload_dict[chunk_id].get("text", "")))
        self._id_list = id_list
        self._bm25 = BM25Okapi(token_list_of_lists) if token_list_of_lists else None

    def bm25_top(self, query: str, top_k: int) -> list[str]:
        """Return the ids of the top BM25 matches."""
        if self._bm25 is None:
            return []
        score_list = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(score_list)), key=lambda index: score_list[index], reverse=True)
        return [self._id_list[index] for index in order[:top_k]]

    def search(self, query: str, k: int) -> RetrievalResult:
        """Run the legacy pipeline and return the reranked top `k`."""
        timing_dict = {"embed_dense": 0.0, "embed_sparse": 0.0, "qdrant": 0.0, "rerank": 0.0}
        total_start = time.perf_counter()
        stage_start = time.perf_counter()
        dense_vector = self.dense_encoder.encode_query(query)
        timing_dict["embed_dense"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        response = self.client.query_points(
            self.collection, query=dense_vector, using=DENSE_VECTOR, limit=LEGACY_DENSE_TOP_K
        )
        timing_dict["qdrant"] = time.perf_counter() - stage_start
        dense_id_list = [str(point.id) for point in response.points]

        # in-process BM25 scoring is the sparse stage of the legacy path
        stage_start = time.perf_counter()
        sparse_id_list = self.bm25_top(query, LEGACY_SPARSE_TOP_K)
        timing_dict["embed_sparse"] = time.perf_counter() - stage_start

        fused_list = reciprocal_rank_fusion(
            [dense_id_list, sparse_id_list], LEGACY_RRF_K, LEGACY_FUSED_TOP_K
        )
        candidate_list = [
            RetrievedChunk(chunk_id=chunk_id, payload=self._payload_dict[chunk_id], score=score)
            for chunk_id, score in fused_list
        ]
        stage_start = time.perf_counter()
        score_list = self.reranker.score(query, [chunk.text for chunk in candidate_list])
        for chunk, score in zip(candidate_list, score_list, strict=True):
            chunk.rerank_score = score
        ranked_list = sorted(
            candidate_list, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True
        )
        timing_dict["rerank"] = time.perf_counter() - stage_start
        timing_dict["total"] = time.perf_counter() - total_start
        return RetrievalResult(
            query=query, mode="legacy", chunk_list=ranked_list[:k], timing_dict=timing_dict
        )
