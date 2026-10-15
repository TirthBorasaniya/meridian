"""Unit tests for the rebuild gate, rollback, and the lifecycle scenario on in-memory Qdrant."""

import pytest
from qdrant_client import QdrantClient

from meridian.common.config import get_gate_config, get_index_config
from meridian.ingest import index as index_module
from meridian.ingest.gate import decide_promotion
from meridian.ingest.rebuild import config_for_collection, rollback
from tests.lifecycle import run_lifecycle_scenario

pytestmark = pytest.mark.unit

METRICS = {"recall_at_5": 0.80, "ndcg_at_10": 0.70}


def test_promotion_rule_follows_spec():
    gate_config = get_gate_config()
    assert decide_promotion(METRICS, None, None, None, gate_config) == (True, [])
    assert decide_promotion(METRICS, METRICS, 0.90, 0.91, gate_config)[0]
    worse_recall = {**METRICS, "recall_at_5": 0.77}
    o_pass, reason_list = decide_promotion(worse_recall, METRICS, 0.9, 0.9, gate_config)
    assert not o_pass and "recall_at_5" in reason_list[0]
    worse_ndcg = {**METRICS, "ndcg_at_10": 0.67}
    assert not decide_promotion(worse_ndcg, METRICS, 0.9, 0.9, gate_config)[0]
    assert not decide_promotion(METRICS, METRICS, 0.87, 0.90, gate_config)[0]
    o_pass, reason_list = decide_promotion(METRICS, METRICS, None, 0.9, gate_config)
    assert not o_pass and "not measured" in reason_list[0]


def test_rollback_requires_a_kept_version():
    client = QdrantClient(":memory:")
    index_config = get_index_config()
    index_module.create_index_collection(client, "meridian_v1", 32, {})
    index_module.switch_alias(client, index_config.alias, "meridian_v1")
    with pytest.raises(RuntimeError, match="nothing to roll back"):
        rollback(client, index_config)


def test_live_collection_is_queried_with_its_own_dense_model():
    index_config = get_index_config()
    assert config_for_collection(index_config, {}) is index_config
    other = config_for_collection(index_config, {"dense_model": "org/adapted-model"})
    assert other.dense.model == "org/adapted-model"
    assert other.dense.query_prefix == index_config.dense.query_prefix


def test_lifecycle_scenario_in_memory(tmp_path):
    run_lifecycle_scenario(QdrantClient(":memory:"), tmp_path, get_index_config())
