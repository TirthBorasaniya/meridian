"""Sparse retrieval with BM25Okapi.

BM25 requires the full tokenized corpus in memory, so the retriever loads every
chunk payload from Qdrant once and builds the index. The index is cached for
the process lifetime; call :func:`reset_sparse_retriever` after re-indexing.
"""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from meridian.config import get_settings
from meridian.ingestion.indexer import fetch_all_chunk_payloads

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


class SparseRetriever:
    """BM25Okapi retriever over a fixed set of chunk payloads."""

    def __init__(self, payload_list: list[dict]) -> None:
        self._payload_list = payload_list
        self._is_empty = len(payload_list) == 0
        if not self._is_empty:
            tokenized_corpus_list = [
                _tokenize(payload.get("text", "")) for payload in payload_list
            ]
            self._bm25 = BM25Okapi(tokenized_corpus_list)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve the top-k chunks for a query by BM25 score.

        Parameters
        ----------
        query : str
            The raw user query.
        top_k : int or None, optional
            Number of candidates to return. Defaults to ``sparse_top_k``.

        Returns
        -------
        list of dict
            Candidate records with keys ``chunk_id``, ``text``, ``payload``,
            ``score``, and ``source`` (always ``"sparse"``), ordered by
            descending BM25 score. Empty if the corpus is empty.
        """
        if self._is_empty:
            return []

        settings = get_settings()
        limit = top_k if top_k is not None else settings.sparse_top_k

        score_list = self._bm25.get_scores(_tokenize(query))
        ranked_index_list = sorted(
            range(len(score_list)), key=lambda index: score_list[index], reverse=True
        )[:limit]

        result_list: list[dict] = []
        for index in ranked_index_list:
            payload_dict = self._payload_list[index]
            result_list.append(
                {
                    "chunk_id": payload_dict.get("chunk_id", ""),
                    "text": payload_dict.get("text", ""),
                    "payload": payload_dict,
                    "score": float(score_list[index]),
                    "source": "sparse",
                }
            )
        return result_list


@lru_cache(maxsize=1)
def get_sparse_retriever() -> SparseRetriever:
    """Return the process-wide cached sparse retriever."""
    return SparseRetriever(fetch_all_chunk_payloads())


def reset_sparse_retriever() -> None:
    """Clear the cached retriever so the next call rebuilds from Qdrant."""
    get_sparse_retriever.cache_clear()
