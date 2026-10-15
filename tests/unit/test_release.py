"""Unit tests for releases: names, manifest fields, upload order with the pointer last, restore into
a fresh copy with checksum verification, and the run log."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from meridian.common.config import (
    ReleaseConfig,
    get_corpus_config,
    get_gate_config,
    get_index_config,
    load_config,
)
from meridian.ingest import index as index_module
from meridian.release import publish as publish_module
from meridian.release import restore as restore_module
from meridian.release.runlog import build_run_record, gate_report_markdown

pytestmark = pytest.mark.unit

MOMENT = datetime(2026, 10, 15, 6, 30, tzinfo=UTC)


def test_release_names_and_files():
    config = load_config(ReleaseConfig, "configs/release.yaml")
    name = publish_module.release_name(3, MOMENT)
    assert name == "meridian_v3_20261015"
    assert publish_module.release_files(config, name) == {
        "snapshot": "snapshots/meridian_v3_20261015.snapshot",
        "manifest": "snapshots/meridian_v3_20261015.manifest.json",
        "ingest_manifest": "snapshots/meridian_v3_20261015.manifest.sqlite",
        "baseline": "snapshots/meridian_v3_20261015.baseline.json",
    }


def test_release_config_refuses_a_public_repo(tmp_path):
    raw = (Path("configs/release.yaml").read_text()).replace("o_private: true", "o_private: false")
    path = tmp_path / "release.yaml"
    path.write_text(raw)
    with pytest.raises(ValueError, match="private"):
        load_config(ReleaseConfig, path)


def test_restore_names_keep_the_version():
    restored = index_module.restore_collection_name("meridian_v3_r202610140630", MOMENT)
    assert restored == "meridian_v3_r202610150630"
    assert index_module.parse_version(restored, "meridian_v") == 3
    assert index_module.base_collection_name(restored) == "meridian_v3"


def test_manifest_has_the_spec_fields():
    manifest = publish_module.build_release_manifest(
        "meridian_v3_20261015", "meridian_v3", 3, 210, 11000, {"value": 0.81}, {}, {}, MOMENT
    )
    for key in ("index_version", "papers", "chunks", "recall_at_5", "git_sha"):
        assert key in manifest
    assert manifest["recall_at_5"] == 0.81


class FakeHub:
    """Records uploads to a local directory in order."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.upload_list: list[str] = []

    def __call__(self, token=None):
        return self

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        assert private is True

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        target = self.root / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path_or_fileobj).read_bytes())
        self.upload_list.append(path_in_repo)


@pytest.fixture
def release_env(tmp_path, monkeypatch):
    client = QdrantClient(":memory:")
    index_config = get_index_config()
    index_module.create_index_collection(client, "meridian_v3", 32, {})
    index_module.switch_alias(client, index_config.alias, "meridian_v3")
    monkeypatch.setattr(index_module, "get_qdrant_client", lambda: client)
    monkeypatch.setenv("HF_INDEX_REPO", "user/meridian-index")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    from meridian.common.settings import get_settings

    get_settings.cache_clear()
    manifest_path = tmp_path / "manifest.sqlite"
    from meridian.ingest.manifest import Manifest

    Manifest(manifest_path).close()
    monkeypatch.setattr(get_corpus_config().paths, "manifest", str(manifest_path))
    monkeypatch.setattr(get_gate_config(), "baseline_path", str(tmp_path / "baseline.json"))
    hub = FakeHub(tmp_path / "repo")
    monkeypatch.setattr("huggingface_hub.HfApi", hub)

    def fake_download(client, collection, dest_path, timeout_s):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"snapshot bytes of " + collection.encode())
        return dest_path

    monkeypatch.setattr(publish_module, "create_and_download", fake_download)
    yield client, hub, tmp_path
    get_settings.cache_clear()


def test_publish_uploads_the_pointer_last(release_env):
    client, hub, tmp_path = release_env
    manifest = publish_module.publish(tmp_path / "work")
    assert hub.upload_list[-1] == "latest.json"
    assert len(hub.upload_list) == 5
    assert manifest["collection"] == "meridian_v3"
    pointer = json.loads((hub.root / "latest.json").read_text())
    assert pointer["name"] == manifest["name"]


def test_restore_verifies_hashes_and_moves_the_alias(release_env, monkeypatch):
    client, hub, tmp_path = release_env
    publish_module.publish(tmp_path / "work")
    restored_list: list[str] = []

    def fake_hub_download(repo_id, filename, repo_type, token, local_dir, force_download=False):
        source = hub.root / filename
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return str(target)

    def fake_upload(collection, snapshot_path, timeout_s):
        restored_list.append(collection)
        index_module.create_index_collection(client, collection, 32, {})

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hub_download)
    monkeypatch.setattr(restore_module, "upload_and_restore", fake_upload)
    monkeypatch.setattr(
        restore_module.get_serving_config(), "index_info_path", str(tmp_path / "info.json")
    )
    info = restore_module.restore_latest(tmp_path / "download", o_install_manifest=False)
    assert info is not None
    assert index_module.base_collection_name(restored_list[0]) == "meridian_v3"
    assert index_module.alias_target(client, get_index_config().alias) == restored_list[0]
    assert (
        restore_module.restore_latest(tmp_path / "download", current_snapshot=info["snapshot"])
        is None
    )

    snapshot_file = (
        hub.root / json.loads((hub.root / "latest.json").read_text())["files"]["snapshot"]
    )
    snapshot_file.write_bytes(b"tampered")
    with pytest.raises(restore_module.ChecksumError):
        restore_module.restore_latest(tmp_path / "download2", o_install_manifest=False)


def test_run_record_and_issue_body():
    ingest = {
        "batch_id": "inc-1",
        "decision": "rejected",
        "papers_fetched": 50,
        "papers_skipped": 10,
        "papers_failed": 1,
        "papers_indexed": 39,
        "chunks_added": 1200,
        "eval": {"value": 0.70, "baseline": 0.75, "max_regression": 0.02},
        "failure_dict": {"2410.00001v1": "parse error"},
    }
    record = build_run_record(ingest, None, 2, "42", "2026-10-15T06:30:00Z", "2026-10-15T07:00:00Z")
    assert record["decision"] == "rejected"
    assert record["duration_s"] == 1800
    assert record["snapshot"] is None
    assert record["recall_at_5"] == 0.70
    body = gate_report_markdown(record, ingest)
    assert "No snapshot was published" in body and "2410.00001v1" in body
    aborted = build_run_record(None, None, 3, "43", "2026-10-15T06:30:00Z", "2026-10-15T06:31:00Z")
    assert aborted["decision"] == "aborted"
    published = build_run_record(
        {**ingest, "decision": "accepted"},
        {"name": "meridian_v3_20261015", "index_version": 3},
        0,
        "44",
        "2026-10-15T06:30:00Z",
        "2026-10-15T06:40:00Z",
    )
    assert published["snapshot"] == "meridian_v3_20261015"
