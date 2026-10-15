"""Unit tests for incremental ingestion: dedup, the fast gate with batch rollback, version
replacement, the deadline guard, and batch records, on an in-memory collection."""

import json
import types
from pathlib import Path

import pytest

from meridian.eval.datasets import write_jsonl
from meridian.ingest import index as index_module
from meridian.ingest import pipeline as pipeline_module
from meridian.ingest.gate import regressed
from meridian.ingest.manifest import (
    DECISION_ACCEPTED,
    DECISION_REJECTED,
    STATUS_INDEXED,
    STATUS_REJECTED,
    STATUS_SUPERSEDED,
)
from meridian.ingest.pipeline import DeadlineExceededError, run_increment
from tests.helpers import FakeArxiv, paper_meta, wired_context

pytestmark = pytest.mark.unit

TOPIC_DICT = {
    "2410.00001v1": "chain of thought prompting elicits stepwise arithmetic reasoning in models",
    "2410.00001v2": "chain of thought prompting revised edition with new arithmetic experiments",
    "2410.00002v1": "expected calibration error measures confidence versus accuracy of predictions",
    "2410.00003v1": "retrieval augmented generation grounds answers in retrieved documents",
    "2410.00004v1": "sparse attention kernels accelerate long context transformers",
}


def _body(versioned_id: str) -> str:
    return (TOPIC_DICT[versioned_id] + ". ") * 12


@pytest.fixture
def setup(tmp_path):
    arxiv = FakeArxiv({key: _body(key) for key in TOPIC_DICT})
    retrieval_set_path = tmp_path / "retrieval_set.jsonl"
    context = wired_context(tmp_path, arxiv, retrieval_set_path)
    # seed paper A without a gate, then label R-fast against its chunks
    arxiv.search_list = [paper_meta("2410.00001")]
    record = run_increment(context, since="2024-11-01", until="2024-11-04", o_skip_gate=True)
    assert record.decision == DECISION_ACCEPTED
    payload_list = [
        point.payload
        for point in index_module.iter_payloads(context.client, "meridian_v1")
        if point.payload["arxiv_id"] == "2410.00001"
    ]
    write_jsonl(
        retrieval_set_path,
        [
            {
                "qid": f"r{index}",
                "question": "chain of thought arithmetic reasoning",
                "spans": [
                    {
                        "arxiv_id": payload["arxiv_id"],
                        "version": payload["version"],
                        "start": payload["char_start"],
                        "end": payload["char_end"],
                    }
                ],
                "in_fast": True,
                "review": {"valid": None},
            }
            for index, payload in enumerate(payload_list[:1])
        ],
    )
    return context, arxiv


def _paper_points(context, arxiv_id: str, version: int | None = None) -> int:
    return sum(
        1
        for point in index_module.iter_payloads(context.client, "meridian_v1")
        if point.payload["arxiv_id"] == arxiv_id
        and (version is None or point.payload["version"] == version)
    )


def test_gate_rule_tolerates_exactly_max_regression():
    assert not regressed(0.73, 0.75, 0.02)
    assert regressed(0.7299, 0.75, 0.02)
    assert not regressed(0.80, 0.75, 0.02)


def test_passing_batch_is_kept_and_creates_the_baseline(setup):
    context, arxiv = setup
    arxiv.search_list = [paper_meta("2410.00001"), paper_meta("2410.00002")]
    record = run_increment(context, since="2024-11-04", until="2024-11-05")
    assert record.decision == DECISION_ACCEPTED
    assert record.papers_skipped == 1
    assert record.papers_indexed == 1
    assert record.eval["o_baseline_created"] is True
    assert _paper_points(context, "2410.00002") == record.chunks_added > 0
    assert json.loads((context.batch_dir / f"{record.batch_id}.json").read_text())["decision"] == (
        "accepted"
    )
    assert context.manifest.last_success_watermark() == record.watermark == record.until


def test_failing_gate_deletes_the_batch_and_alerts(setup):
    context, arxiv = setup
    previous_watermark = context.manifest.last_success_watermark()
    # an unreachable baseline forces the gate to fail
    Path(context.baseline_path).write_text(json.dumps({"metric": "recall_at_5", "value": 1.5}))
    arxiv.search_list = [paper_meta("2410.00003")]
    record = run_increment(context, since="2024-11-04", until="2024-11-06")
    assert record.decision == DECISION_REJECTED
    assert record.chunks_deleted == record.chunks_added > 0
    assert _paper_points(context, "2410.00003") == 0
    assert _paper_points(context, "2410.00001") > 0
    assert context.manifest.get_paper("2410.00003", 1).status == STATUS_REJECTED
    assert context.manifest.last_success_watermark() == previous_watermark
    alert = json.loads(context.alert_path.read_text().splitlines()[-1])
    assert alert["batch_id"] == record.batch_id
    assert "FAIL" in alert["message"]


def test_newer_version_replaces_older_points_after_acceptance(setup):
    context, arxiv = setup
    arxiv.latest_list = [paper_meta("2410.00001", version=2)]
    arxiv.search_list = []
    record = run_increment(context, since="2024-11-05", until="2024-11-05")
    assert record.decision == DECISION_ACCEPTED
    assert record.versions_replaced == 1
    assert _paper_points(context, "2410.00001", version=1) == 0
    assert _paper_points(context, "2410.00001", version=2) > 0
    assert context.manifest.get_paper("2410.00001", 1).status == STATUS_SUPERSEDED
    assert context.manifest.get_paper("2410.00001", 2).status == STATUS_INDEXED


def test_evaluation_papers_are_not_reversioned(setup):
    context, arxiv = setup
    context.excluded_version_ids = {"2410.00001"}
    arxiv.latest_list = [paper_meta("2410.00001", version=2)]
    arxiv.search_list = []
    record = run_increment(context, since="2024-11-05", until="2024-11-05")
    assert record.versions_replaced == 0
    assert _paper_points(context, "2410.00001", version=1) > 0


def test_cap_keeps_the_watermark_at_the_last_processed_paper(setup):
    context, arxiv = setup
    arxiv.search_list = [
        paper_meta("2410.00002", published="2024-11-05T01:00:00Z"),
        paper_meta("2410.00003", published="2024-11-05T02:00:00Z"),
        paper_meta("2410.00004", published="2024-11-05T03:00:00Z"),
    ]
    record = run_increment(context, since="2024-11-05", until="2024-11-05", max_papers=2)
    assert record.papers_indexed == 2
    assert not record.o_window_exhausted
    assert record.watermark == "2024-11-05T02:00:00Z"


def test_deadline_rolls_back_the_partial_batch(setup, monkeypatch):
    context, arxiv = setup
    tick = {"now": 0.0}

    def perf_counter():
        tick["now"] += 20.0
        return tick["now"]

    monkeypatch.setattr(pipeline_module, "time", types.SimpleNamespace(perf_counter=perf_counter))
    arxiv.search_list = [paper_meta("2410.00002"), paper_meta("2410.00003")]
    with pytest.raises(DeadlineExceededError):
        run_increment(context, since="2024-11-05", until="2024-11-05", deadline_minutes=0.5)
    assert _paper_points(context, "2410.00002") == 0
    aborted = context.manifest.batches("increment")[-1]
    assert aborted["decision"] == "aborted"


def test_dry_run_writes_nothing(setup):
    context, arxiv = setup
    arxiv.search_list = [paper_meta("2410.00004")]
    before = arxiv.download_count
    record = run_increment(context, since="2024-11-05", until="2024-11-05", o_dry_run=True)
    assert record.decision == "dry_run"
    assert record.paper_id_list == ["2410.00004v1"]
    assert arxiv.download_count == before
    assert context.manifest.get_paper("2410.00004", 1) is None
