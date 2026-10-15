"""Unit tests for the daily scheduler registration and the E4 helpers."""

import json

import pytest
from qdrant_client import QdrantClient

from meridian.common.config import get_scheduler_config
from meridian.eval.ingest_eval import clone_collection, read_run_log, summarize_runs
from meridian.ingest import index as index_module
from meridian.ingest.scheduler import build_scheduler
from tests.fakes import FakeDenseEncoder, FakeSparseEncoder

pytestmark = pytest.mark.unit


def test_scheduler_registers_one_coalesced_daily_job():
    scheduler = build_scheduler(get_scheduler_config(), o_blocking=False)
    job_list = scheduler.get_jobs()
    assert len(job_list) == 1
    job = job_list[0]
    assert job.id == "daily_ingest"
    assert job.max_instances == 1
    assert job.coalesce is True
    hour, minute = get_scheduler_config().daily_time.split(":")
    field_dict = {field.name: str(field) for field in job.trigger.fields}
    assert field_dict["hour"] == str(int(hour))
    assert field_dict["minute"] == str(int(minute))


def test_clone_collection_copies_points_with_vectors():
    client = QdrantClient(":memory:")
    index_module.create_index_collection(client, "meridian_v1", 32, {"index_version": 1})
    text_list = ["alpha beta", "gamma delta", "epsilon"]
    payload_list = [
        {"chunk_id": f"00000000-0000-0000-0000-{index:012d}", "arxiv_id": "a", "text": text}
        for index, text in enumerate(text_list)
    ]
    point_list = index_module.build_points(
        payload_list,
        FakeDenseEncoder().encode_documents(text_list),
        FakeSparseEncoder().encode_documents(text_list),
    )
    index_module.upsert_points(client, "meridian_v1", point_list, 10)
    assert clone_collection(client, "meridian_v1", "meridian_e4_sim", page_size=2) == 3
    assert index_module.count_points(client, "meridian_e4_sim") == 3
    metadata = index_module.collection_metadata(client, "meridian_e4_sim")
    assert metadata["clone_of"] == "meridian_v1"
    hit = client.query_points(
        "meridian_e4_sim", query=FakeDenseEncoder().encode_query("gamma delta"), using="dense"
    ).points[0]
    assert hit.payload["text"] == "gamma delta"


def test_run_log_summary(tmp_path):
    assert read_run_log(tmp_path / "missing.jsonl") == []
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(run)
            for run in [
                {"decision": "accepted", "snapshot": "s1"},
                {"decision": "rejected", "snapshot": None},
                {"decision": "accepted", "snapshot": "s2"},
            ]
        )
    )
    summary = summarize_runs(read_run_log(path))
    assert summary == {"runs": 3, "by_decision": {"accepted": 2, "rejected": 1}, "published": 2}
