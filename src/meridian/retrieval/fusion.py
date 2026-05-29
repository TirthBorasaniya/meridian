"""Reciprocal Rank Fusion of independent ranked candidate lists.

Merges dense and sparse retrieval results into a single ranking. Each candidate
accumulates ``1 / (k + rank)`` from every list it appears in, where ``rank`` is
its 1-based position in that list and ``k`` is the standard RRF constant (60).
Candidates are deduplicated by ``chunk_id``.
"""

from meridian.config import get_settings


def reciprocal_rank_fusion(
    ranked_list_of_lists: list[list[dict]],
    k: int | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Fuse multiple ranked candidate lists with Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_list_of_lists : list of list of dict
        Each inner list is a ranking of candidate records, each carrying a
        ``chunk_id``. Lists are assumed already sorted best-first.
    k : int or None, optional
        The RRF constant. Defaults to ``rrf_k`` (60).
    top_k : int or None, optional
        Number of fused candidates to return. Defaults to ``dense_top_k``.

    Returns
    -------
    list of dict
        Deduplicated candidates ordered by descending fused score. Each record
        is the first-seen instance for its ``chunk_id`` with an added
        ``rrf_score`` key.
    """
    settings = get_settings()
    rrf_k = k if k is not None else settings.rrf_k
    limit = top_k if top_k is not None else settings.dense_top_k

    score_by_id: dict[str, float] = {}
    doc_by_id: dict[str, dict] = {}

    for ranked_list in ranked_list_of_lists:
        for position, doc in enumerate(ranked_list):
            chunk_id = doc["chunk_id"]
            rank = position + 1
            score_by_id[chunk_id] = score_by_id.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            doc_by_id.setdefault(chunk_id, doc)

    ranked_id_list = sorted(score_by_id, key=lambda cid: score_by_id[cid], reverse=True)[:limit]

    fused_list: list[dict] = []
    for chunk_id in ranked_id_list:
        fused_doc = dict(doc_by_id[chunk_id])
        fused_doc["rrf_score"] = score_by_id[chunk_id]
        fused_list.append(fused_doc)
    return fused_list
