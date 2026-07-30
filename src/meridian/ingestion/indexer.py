"""Qdrant indexing with deterministic chunk identifiers.

Embeds chunk text and upserts points into the configured Qdrant collection.
Qdrant point identifiers must be unsigned integers or UUIDs, so the
human-readable ``{arxiv_id}_{chunk_index}`` chunk ID is mapped to a stable
UUID5 and the original chunk ID is preserved in the payload. Using a
deterministic UUID makes re-ingestion overwrite the same point rather than
creating duplicates.
"""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from meridian.config import get_settings
from meridian.ingestion.embedder import get_embedder

# Fixed namespace so that a given chunk_id always maps to the same point ID.
CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "meridian.arxiv.chunk")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return the process-wide cached Qdrant client.

    Uses a server URL when ``QDRANT_URL`` is set, otherwise local disk
    persistence at ``QDRANT_PATH``. In server mode ``QDRANT_API_KEY`` is
    forwarded when set; it is omitted when empty so that an unauthenticated
    local Docker Compose Qdrant continues to work unchanged. The key is
    irrelevant in local disk mode, where there is no server to authenticate to.
    """
    settings = get_settings()
    if settings.qdrant_url:
        if settings.qdrant_api_key:
            return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_path)


def point_id_for(chunk_id: str) -> str:
    """Map a human-readable chunk ID to a deterministic UUID string."""
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))


def ensure_collection() -> None:
    """Create the configured collection if it does not already exist.

    The collection uses cosine distance, matching the L2-normalised BGE
    embeddings produced by the embedder.
    """
    client = get_qdrant_client()
    settings = get_settings()
    existing_name_set = {
        collection.name for collection in client.get_collections().collections
    }
    if settings.qdrant_collection not in existing_name_set:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )


def index_chunks(chunk_list: list[dict], metadata_dict: dict) -> int:
    """Embed and upsert a paper's chunks into Qdrant.

    Parameters
    ----------
    chunk_list : list of dict
        Chunk records from :func:`meridian.ingestion.chunker.chunk_text`.
    metadata_dict : dict
        Paper-level metadata to attach to every chunk payload. Expected keys:
        ``title``, ``authors``, ``categories``, ``year``.

    Returns
    -------
    int
        The number of points upserted.
    """
    if not chunk_list:
        return 0

    settings = get_settings()
    client = get_qdrant_client()
    embedder = get_embedder()

    vector_list = embedder.embed_documents([chunk["text"] for chunk in chunk_list])
    point_list: list[PointStruct] = []
    # strict: the embedder returns one vector per chunk. A mismatch would pair
    # chunks with the wrong vectors and silently corrupt the index, which is
    # far worse to discover later than a loud failure here.
    for chunk, vector in zip(chunk_list, vector_list, strict=True):
        payload_dict = {
            "chunk_id": chunk["chunk_id"],
            "arxiv_id": chunk["arxiv_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "title": metadata_dict.get("title", ""),
            "authors": metadata_dict.get("authors", ""),
            "categories": metadata_dict.get("categories", ""),
            "year": metadata_dict.get("year", 0),
        }
        point_list.append(
            PointStruct(
                id=point_id_for(chunk["chunk_id"]),
                vector=vector,
                payload=payload_dict,
            )
        )

    client.upsert(collection_name=settings.qdrant_collection, points=point_list)
    return len(point_list)


def fetch_all_chunk_payloads() -> list[dict]:
    """Scroll the entire collection and return every chunk payload.

    Used by the sparse (BM25) retriever, which must hold the full tokenized
    corpus in memory.

    Returns
    -------
    list of dict
        All chunk payloads in the collection.
    """
    settings = get_settings()
    client = get_qdrant_client()
    payload_list: list[dict] = []
    next_offset = None
    while True:
        point_list, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        # ``payload`` is Optional on the client's record type. Points are
        # always upserted with a payload, but skipping any that lack one keeps
        # None out of the list, which downstream BM25 indexing would treat as
        # a dict and fail on.
        payload_list.extend(point.payload for point in point_list if point.payload is not None)
        if next_offset is None:
            break
    return payload_list


def count_points() -> int:
    """Return the number of points currently in the collection."""
    settings = get_settings()
    client = get_qdrant_client()
    return client.count(collection_name=settings.qdrant_collection).count
