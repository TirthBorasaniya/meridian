"""Cross-encoder reranking with BAAI/bge-reranker-base.

The cross-encoder scores each post-fusion candidate against the original query
and doubles as the document relevance grader: candidates scoring below the
threshold (0.5) are filtered out. The surviving top-k candidates are passed to
the generator. When every candidate is filtered, the graph routes to the CRAG
web-search fallback.
"""

from functools import lru_cache

from meridian.config import get_settings


class Reranker:
    """Cross-encoder reranker and relevance filter."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so importing this module does not load torch and
        # sentence-transformers until a reranker is actually constructed.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        doc_list: list[dict],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """Score, filter, and re-rank candidates against the query.

        Parameters
        ----------
        query : str
            The original user query (not the BGE-prefixed form).
        doc_list : list of dict
            Post-fusion candidates, each carrying a ``text`` field.
        top_k : int or None, optional
            Maximum number of survivors to return. Defaults to ``rerank_top_k``.
        threshold : float or None, optional
            Minimum score to keep a candidate. Defaults to
            ``rerank_score_threshold`` (0.5).

        Returns
        -------
        list of dict
            Candidates scoring at or above the threshold, ordered by descending
            ``rerank_score`` and truncated to ``top_k``. Empty if all
            candidates are filtered.
        """
        if not doc_list:
            return []

        settings = get_settings()
        limit = top_k if top_k is not None else settings.rerank_top_k
        score_threshold = (
            threshold if threshold is not None else settings.rerank_score_threshold
        )

        pair_list = [(query, doc["text"]) for doc in doc_list]
        score_list = self._model.predict(pair_list)

        scored_list: list[dict] = []
        # strict: the cross-encoder returns exactly one score per input pair.
        # A length mismatch means the model misbehaved, and silently dropping
        # the tail would score documents against the wrong scores.
        for doc, score in zip(doc_list, score_list, strict=True):
            scored_doc = dict(doc)
            scored_doc["rerank_score"] = float(score)
            scored_list.append(scored_doc)

        scored_list.sort(key=lambda doc: doc["rerank_score"], reverse=True)
        survivor_list = [
            doc for doc in scored_list if doc["rerank_score"] >= score_threshold
        ]
        return survivor_list[:limit]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Return the process-wide cached cross-encoder reranker."""
    settings = get_settings()
    return Reranker(settings.reranker_model)
