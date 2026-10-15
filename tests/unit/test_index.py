"""Unit tests for versioned collections, batch deletion, and the atomic alias switch, run against
Qdrant local mode in memory."""

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from meridian.common.config import get_index_config
from meridian.ingest import index as index_module
from tests.fakes import FakeDenseEncoder, FakeSparseEncoder

pytestmark = pytest.mark.unit


def _payload(chunk_number: int, arxiv_id: str, batch_id: str) -> dict:
    return {
        "chunk_id": f"00000000-0000-0000-0000-{chunk_number:012d}",
        "arxiv_id": arxiv_id,
        "ingest_batch_id": batch_id,
        "text": f"chunk {chunk_number} about chain of thought",
    }


def _populate(client: QdrantClient, name: str, payload_list: list[dict]) -> None:
    text_list = [payload["text"] for payload in payload_list]
    point_list = index_module.build_points(
        payload_list,
        FakeDenseEncoder().encode_documents(text_list),
        FakeSparseEncoder().encode_documents(text_list),
    )
    index_module.upsert_points(client, name, point_list, batch_size=2)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_collection_has_named_vectors_idf_and_metadata(client):
    index_config = get_index_config()
    metadata = index_module.build_metadata(index_config, 1, 32)
    index_module.create_index_collection(client, "meridian_v1", 32, metadata)
    info = client.get_collection("meridian_v1")
    assert set(info.config.params.vectors) == {"dense"}
    assert info.config.params.vectors["dense"].distance == models.Distance.COSINE
    sparse_params = info.config.params.sparse_vectors["sparse"]
    assert sparse_params.modifier == models.Modifier.IDF
    assert index_module.collection_metadata(client, "meridian_v1")["index_version"] == 1
    with pytest.raises(ValueError, match="already exists"):
        index_module.create_index_collection(client, "meridian_v1", 32, metadata)


def test_payload_indexes_cover_the_spec_fields():
    field_set = {field_name for field_name, _schema in index_module.PAYLOAD_INDEX_LIST}
    assert field_set == {"arxiv_id", "published", "primary_category", "ingest_batch_id"}


def test_build_points_rejects_length_mismatch():
    with pytest.raises(ValueError):
        index_module.build_points([{"chunk_id": "x"}], np.zeros((2, 4)), [])


def test_delete_batch_removes_only_that_batch(client):
    index_module.create_index_collection(client, "meridian_v1", 32, {})
    _populate(
        client,
        "meridian_v1",
        [_payload(1, "a", "b1"), _payload(2, "a", "b1"), _payload(3, "b", "b2")],
    )
    assert index_module.count_points(client, "meridian_v1") == 3
    assert index_module.count_papers(client, "meridian_v1") == 2
    assert index_module.delete_batch(client, "meridian_v1", "b1") == 2
    assert index_module.count_points(client, "meridian_v1") == 1
    assert index_module.delete_batch(client, "meridian_v1", "missing") == 0


def test_alias_switch_is_one_request_and_returns_previous(client):
    for version in (1, 2):
        index_module.create_index_collection(client, f"meridian_v{version}", 32, {})
    assert index_module.switch_alias(client, "meridian_live", "meridian_v1") is None
    assert index_module.alias_target(client, "meridian_live") == "meridian_v1"
    operation_list = index_module.alias_operations("meridian_live", "meridian_v2", "meridian_v1")
    assert isinstance(operation_list[0], models.DeleteAliasOperation)
    assert isinstance(operation_list[1], models.CreateAliasOperation)
    assert index_module.switch_alias(client, "meridian_live", "meridian_v2") == "meridian_v1"
    assert index_module.live_version(client) == 2
    with pytest.raises(ValueError, match="does not exist"):
        index_module.switch_alias(client, "meridian_live", "meridian_v9")


def test_versions_and_pruning_keep_one_previous(client):
    index_config = get_index_config()
    for version in (1, 2, 3):
        index_module.create_index_collection(client, f"meridian_v{version}", 32, {})
    client.create_collection(
        "unrelated", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    assert index_module.list_versions(client, "meridian_v") == [1, 2, 3]
    assert index_module.next_version(client, "meridian_v") == 4
    index_module.switch_alias(client, index_config.alias, "meridian_v3")
    assert index_module.prune_versions(client, index_config) == ["meridian_v1"]
    assert index_module.list_versions(client, "meridian_v") == [2, 3]
    assert client.collection_exists("unrelated")


def test_parse_version():
    assert index_module.parse_version("meridian_v12", "meridian_v") == 12
    assert index_module.parse_version("meridian_v12_tmp", "meridian_v") is None
    assert index_module.parse_version("other", "meridian_v") is None
