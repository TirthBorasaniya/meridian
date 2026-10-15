"""Versioned Qdrant collections behind the `meridian_live` alias.

Each index version is a collection `meridian_v{N}` with named `dense` and `sparse` vectors,
payload indexes on the filter fields, and collection metadata recording the build fingerprint.
The live version is reached only through the alias, which moves in one atomic
`update_collection_aliases` call.
"""

import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models

from meridian.common.config import IndexConfig, get_index_config
from meridian.common.settings import get_settings

logger = logging.getLogger(__name__)

# ============= Constants =============

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
PAYLOAD_INDEX_LIST: list[tuple[str, models.PayloadSchemaType]] = [
    ("arxiv_id", models.PayloadSchemaType.KEYWORD),
    ("published", models.PayloadSchemaType.DATETIME),
    ("primary_category", models.PayloadSchemaType.KEYWORD),
    ("ingest_batch_id", models.PayloadSchemaType.KEYWORD),
]
FACET_LIMIT = 1_000_000


# ============= Client =============


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return the process-wide client: server mode when QDRANT_URL is set, else local mode."""
    settings = get_settings()
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key.get_secret_value() or None
        return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=120)
    return QdrantClient(path=settings.qdrant_path)


# ============= Names and versions =============


def collection_name(prefix: str, version: int) -> str:
    """Return the collection name of an index version, for example `meridian_v3`."""
    return f"{prefix}{version}"


def parse_version(name: str, prefix: str) -> int | None:
    """Return the version number in a collection name, or None for other collections."""
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name)
    return int(match.group(1)) if match else None


def list_versions(client: QdrantClient, prefix: str) -> list[int]:
    """Return the existing index versions in ascending order."""
    version_list = [
        parse_version(collection.name, prefix)
        for collection in client.get_collections().collections
    ]
    return sorted(version for version in version_list if version is not None)


def next_version(client: QdrantClient, prefix: str) -> int:
    """Return the version number for a new collection."""
    version_list = list_versions(client, prefix)
    return (version_list[-1] + 1) if version_list else 1


def alias_target(client: QdrantClient, alias: str) -> str | None:
    """Return the collection an alias points at, or None when the alias does not exist."""
    for alias_description in client.get_aliases().aliases:
        if alias_description.alias_name == alias:
            return alias_description.collection_name
    return None


def live_version(client: QdrantClient, index_config: IndexConfig | None = None) -> int | None:
    """Return the version number behind the live alias."""
    config = index_config or get_index_config()
    target = alias_target(client, config.alias)
    return parse_version(target, config.collection_prefix) if target else None


# ============= Creation =============


def build_metadata(index_config: IndexConfig, index_version: int, dense_dim: int) -> dict:
    """Return the collection metadata recorded at creation."""
    return {
        "index_version": index_version,
        "fingerprint": index_config.build_fingerprint(),
        "index_config_sha256": index_config.sha256,
        "dense_model": index_config.dense.model,
        "dense_dim": dense_dim,
        "sparse_model": index_config.sparse.model,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def create_index_collection(
    client: QdrantClient,
    name: str,
    dense_dim: int,
    metadata: dict,
) -> None:
    """
    Create a versioned collection with named dense and sparse vectors and payload indexes.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client.
    name : str
        Collection name, for example `meridian_v2`.
    dense_dim : int
        Dense embedding dimension.
    metadata : dict
        Collection metadata, see `build_metadata`.
    """
    if client.collection_exists(name):
        raise ValueError(f"collection {name} already exists")
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
        metadata=metadata,
    )
    for field_name, field_schema in PAYLOAD_INDEX_LIST:
        client.create_payload_index(name, field_name=field_name, field_schema=field_schema)
    logger.info("collection created", extra={"collection": name, "dense_dim": dense_dim})


def collection_metadata(client: QdrantClient, name: str) -> dict:
    """Return the metadata recorded on a collection."""
    return dict(client.get_collection(name).config.metadata or {})


# ============= Writes =============


def build_points(
    payload_list: list[dict],
    dense_matrix: np.ndarray,
    sparse_list: list[models.SparseVector],
) -> list[models.PointStruct]:
    """
    Pair payloads with their vectors; the point id is the chunk id.

    Parameters
    ----------
    payload_list : list of dict
        Chunk payloads, each with a `chunk_id`.
    dense_matrix : np.ndarray
        Dense vectors, shape (len(payload_list), dense_dim).
    sparse_list : list of SparseVector
        Sparse vectors, one per payload.

    Returns
    -------
    point_list : list of PointStruct
        Points ready for upsert.
    """
    if not (len(payload_list) == len(dense_matrix) == len(sparse_list)):
        raise ValueError("payloads, dense vectors, and sparse vectors differ in length")
    return [
        models.PointStruct(
            id=payload["chunk_id"],
            vector={DENSE_VECTOR: dense_vector.tolist(), SPARSE_VECTOR: sparse_vector},
            payload=payload,
        )
        for payload, dense_vector, sparse_vector in zip(
            payload_list, dense_matrix, sparse_list, strict=True
        )
    ]


def upsert_points(
    client: QdrantClient, name: str, point_list: list[models.PointStruct], batch_size: int
) -> int:
    """Upsert points in batches and return how many were written."""
    for batch_start in range(0, len(point_list), batch_size):
        client.upsert(
            collection_name=name,
            points=point_list[batch_start : batch_start + batch_size],
            wait=True,
        )
    return len(point_list)


def field_filter(field_name: str, value) -> models.Filter:
    """Return a filter matching one payload value."""
    return models.Filter(
        must=[models.FieldCondition(key=field_name, match=models.MatchValue(value=value))]
    )


def count_points(client: QdrantClient, name: str, query_filter: models.Filter | None = None) -> int:
    """Return the exact number of points, optionally under a filter."""
    return int(client.count(collection_name=name, count_filter=query_filter, exact=True).count)


def delete_by_filter(client: QdrantClient, name: str, query_filter: models.Filter) -> int:
    """Delete every point matching a filter and return how many were deleted."""
    matched_count = count_points(client, name, query_filter)
    if matched_count:
        client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )
    return matched_count


def delete_batch(client: QdrantClient, name: str, ingest_batch_id: str) -> int:
    """Delete all points written by one ingest batch."""
    return delete_by_filter(client, name, field_filter("ingest_batch_id", ingest_batch_id))


def delete_paper(client: QdrantClient, name: str, arxiv_id: str) -> int:
    """Delete all points of every version of one paper."""
    return delete_by_filter(client, name, field_filter("arxiv_id", arxiv_id))


def count_papers(client: QdrantClient, name: str) -> int:
    """Return the number of distinct papers through an exact facet on `arxiv_id`."""
    response = client.facet(collection_name=name, key="arxiv_id", limit=FACET_LIMIT, exact=True)
    return len(response.hits)


def iter_payloads(
    client: QdrantClient, name: str, with_vectors: bool = False, page_size: int = 512
) -> Iterable[models.Record]:
    """Yield every point of a collection with payload, optionally with vectors."""
    offset = None
    while True:
        record_list, offset = client.scroll(
            collection_name=name,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        yield from record_list
        if offset is None:
            return


# ============= Alias =============


def alias_operations(alias: str, new_collection: str, current_target: str | None) -> list:
    """Return the operations that move an alias to a new collection in one request."""
    operation_list: list = []
    if current_target is not None:
        operation_list.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    operation_list.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=new_collection, alias_name=alias)
        )
    )
    return operation_list


def switch_alias(client: QdrantClient, alias: str, new_collection: str) -> str | None:
    """
    Atomically point an alias at a collection.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client.
    alias : str
        Alias name, normally `meridian_live`.
    new_collection : str
        Collection the alias should resolve to.

    Returns
    -------
    previous : str or None
        The collection the alias pointed at before the switch.
    """
    if not client.collection_exists(new_collection):
        raise ValueError(f"cannot switch {alias}: {new_collection} does not exist")
    previous = alias_target(client, alias)
    if previous == new_collection:
        return previous
    client.update_collection_aliases(
        change_aliases_operations=alias_operations(alias, new_collection, previous)
    )
    logger.info(
        "alias switched", extra={"alias": alias, "collection": new_collection, "previous": previous}
    )
    return previous


def prune_versions(client: QdrantClient, index_config: IndexConfig) -> list[str]:
    """
    Delete versions older than the live one beyond `keep_previous_versions`.

    Returns
    -------
    deleted_list : list of str
        Names of the deleted collections.
    """
    current = live_version(client, index_config)
    if current is None:
        return []
    older_list = [
        version
        for version in list_versions(client, index_config.collection_prefix)
        if version < current
    ]
    keep_count = index_config.keep_previous_versions
    doomed_list = older_list[: max(0, len(older_list) - keep_count)]
    deleted_list: list[str] = []
    for version in doomed_list:
        name = collection_name(index_config.collection_prefix, version)
        client.delete_collection(name)
        deleted_list.append(name)
    if deleted_list:
        logger.info("old versions deleted", extra={"collections": deleted_list})
    return deleted_list


# ============= Status =============


def index_status(client: QdrantClient, index_config: IndexConfig) -> dict:
    """Return the alias target and per-version point counts."""
    version_list = list_versions(client, index_config.collection_prefix)
    return {
        "alias": index_config.alias,
        "live": alias_target(client, index_config.alias),
        "collections": {
            collection_name(index_config.collection_prefix, version): count_points(
                client, collection_name(index_config.collection_prefix, version)
            )
            for version in version_list
        },
    }


def cli_status(args) -> int:
    """Print the index status as JSON."""
    import json

    print(json.dumps(index_status(get_qdrant_client(), get_index_config()), indent=2))
    return 0
