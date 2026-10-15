"""Unit tests for manifest deduplication, version replacement, and batch records."""

import pytest

from meridian.ingest.manifest import (
    DECISION_ACCEPTED,
    DECISION_REJECTED,
    MAX_FAILED_ATTEMPTS,
    STATUS_INDEXED,
    STATUS_PARSED,
    STATUS_SUPERSEDED,
    STATUS_WITHDRAWN,
    Manifest,
)
from meridian.ingest.records import PaperMeta

pytestmark = pytest.mark.unit


def _meta(arxiv_id: str = "2401.00001", version: int = 1) -> PaperMeta:
    return PaperMeta(
        arxiv_id=arxiv_id,
        version=version,
        title="T",
        authors=["A"],
        published="2024-01-01T00:00:00Z",
        updated="2024-01-01T00:00:00Z",
        primary_category="cs.CL",
        categories=["cs.CL"],
    )


@pytest.fixture
def manifest(tmp_path) -> Manifest:
    return Manifest(tmp_path / "manifest.sqlite")


def test_same_version_already_indexed_is_skipped(manifest):
    meta = _meta()
    assert not manifest.should_skip(meta)
    manifest.upsert_meta(meta, STATUS_PARSED)
    assert not manifest.should_skip(meta)
    manifest.update_paper(meta.arxiv_id, meta.version, status=STATUS_INDEXED)
    assert manifest.should_skip(meta)


def test_withdrawn_and_exhausted_failures_are_skipped(manifest):
    withdrawn = _meta("2401.00002")
    manifest.upsert_meta(withdrawn, STATUS_WITHDRAWN)
    assert manifest.should_skip(withdrawn)
    failing = _meta("2401.00003")
    manifest.upsert_meta(failing, STATUS_PARSED)
    for _attempt in range(MAX_FAILED_ATTEMPTS - 1):
        manifest.record_failure(failing.arxiv_id, failing.version, "boom")
    assert not manifest.should_skip(failing)
    manifest.record_failure(failing.arxiv_id, failing.version, "boom")
    assert manifest.should_skip(failing)


def test_newer_version_supersedes_older(manifest):
    manifest.upsert_meta(_meta(version=1), STATUS_INDEXED)
    manifest.upsert_meta(_meta(version=2), STATUS_PARSED)
    assert manifest.supersede_older("2401.00001", 2) == [1]
    assert manifest.get_paper("2401.00001", 1).status == STATUS_SUPERSEDED
    # an older version arriving after a newer one is skipped
    assert manifest.should_skip(_meta(version=1))
    corpus_list = manifest.corpus_papers()
    assert [(row.meta.arxiv_id, row.meta.version) for row in corpus_list] == [("2401.00001", 2)]


def test_update_rejects_unknown_columns(manifest):
    manifest.upsert_meta(_meta(), STATUS_PARSED)
    with pytest.raises(ValueError):
        manifest.update_paper("2401.00001", 1, title="no")


def test_batch_records_and_last_success_watermark(manifest):
    assert manifest.last_success_watermark() is None
    manifest.start_batch("b1", "increment", "2024-01-01", "", "meridian_live", 1)
    manifest.finish_batch("b1", DECISION_ACCEPTED, "2024-01-02T10:00:00Z", {"chunks_added": 5})
    manifest.start_batch("b2", "increment", "2024-01-02", "", "meridian_live", 1)
    manifest.finish_batch("b2", DECISION_REJECTED, "2024-01-03T10:00:00Z", {"chunks_added": 3})
    assert manifest.last_success_watermark() == "2024-01-02T10:00:00Z"
    assert manifest.get_batch("b2")["record"] == {"chunks_added": 3}
    assert [batch["batch_id"] for batch in manifest.batches("increment")] == ["b1", "b2"]
