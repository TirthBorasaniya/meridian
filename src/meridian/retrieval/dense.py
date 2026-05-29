"""Dense retrieval over Qdrant.

Embeds the query with the BGE asymmetric prefix and returns the top-k nearest
chunks by cosine similarity.
"""

from meridian.config import get_settings
from meridian.ingestion.embedder import get_embedder
from meridian.ingestion.indexer import get_qdrant_client


def dense_retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Retrieve the top-k chunks for a query by dense similarity.

    Parameters
    ----------
    query : str
        The raw user query. The BGE query prefix is applied internally.
    top_k : int or None, optional
        Number of candidates to return. Defaults to ``dense_top_k``.

    Returns
    -------
    list of dict
        Candidate records with keys ``chunk_id``, ``text``, ``payload``,
        ``score``, and ``source`` (always ``"dense"``), ordered by descending
        similarity.
    """
    settings = get_settings()
    limit = top_k if top_k is not None else settings.dense_top_k

    query_vector = get_embedder().embed_query(query)
    response = get_qdrant_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    result_list: list[dict] = []
    for point in response.points:
        payload_dict = point.payload or {}
        result_list.append(
            {
                "chunk_id": payload_dict.get("chunk_id", str(point.id)),
                "text": payload_dict.get("text", ""),
                "payload": payload_dict,
                "score": float(point.score),
                "source": "dense",
            }
        )
    return result_list
