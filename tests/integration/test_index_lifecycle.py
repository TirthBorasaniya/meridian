"""Integration test of the index lifecycle against the Qdrant server of the compose stack.

Requires `QDRANT_URL` (for example `http://127.0.0.1:6333`). Collections and the alias use an
`itest_` prefix, so a developer's real `meridian_v{N}` collections are never touched.
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient

from meridian.common.config import get_index_config
from tests.lifecycle import run_lifecycle_scenario

pytestmark = pytest.mark.integration


@pytest.fixture
def server_client():
    url = os.environ.get("QDRANT_URL")
    if not url:
        pytest.fail("QDRANT_URL must point at the compose Qdrant for integration tests")
    client = QdrantClient(url=url, timeout=60)
    yield client
    client.close()


def test_lifecycle_against_qdrant_server(server_client, tmp_path):
    run_token = uuid.uuid4().hex[:8]
    index_config = get_index_config().model_copy(
        update={
            "collection_prefix": f"itest_{run_token}_meridian_v",
            "alias": f"itest_{run_token}_meridian_live",
        }
    )
    try:
        run_lifecycle_scenario(server_client, tmp_path, index_config)
        info = server_client.get_collection(f"itest_{run_token}_meridian_v1")
        # payload indexes exist in server mode, unlike local mode
        assert {"arxiv_id", "published", "primary_category", "ingest_batch_id"} <= set(
            info.payload_schema
        )
    finally:
        for alias_description in server_client.get_aliases().aliases:
            if alias_description.alias_name == index_config.alias:
                server_client.delete_alias(index_config.alias)
        for collection in server_client.get_collections().collections:
            if collection.name.startswith(f"itest_{run_token}_"):
                server_client.delete_collection(collection.name)
