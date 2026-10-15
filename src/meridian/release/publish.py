"""Publish the live index to the private dataset repo: the Qdrant snapshot, the ingestion
manifest, the gate baseline, and a release manifest, then move `latest.json` (SPEC S3.9).

`latest.json` is uploaded last, so a reader never sees a pointer to a partial release.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from meridian.common.config import (
    ReleaseConfig,
    get_corpus_config,
    get_gate_config,
    get_index_config,
    get_release_config,
)
from meridian.common.results import git_sha
from meridian.common.settings import get_settings
from meridian.ingest import index as index_module
from meridian.release.snapshot import copy_sqlite, create_and_download, sha256_of

logger = logging.getLogger(__name__)


def release_name(index_version: int, moment: datetime) -> str:
    """Return `meridian_v{N}_{YYYYMMDD}`."""
    return f"meridian_v{index_version}_{moment.strftime('%Y%m%d')}"


def release_files(config: ReleaseConfig, name: str) -> dict[str, str]:
    """Return the repo paths of one release's files."""
    base = f"{config.snapshot_prefix}/{name}"
    return {
        "snapshot": f"{base}.snapshot",
        "manifest": f"{base}.manifest.json",
        "ingest_manifest": f"{base}.manifest.sqlite",
        "baseline": f"{base}.baseline.json",
    }


def build_release_manifest(
    name: str,
    collection: str,
    index_version: int,
    papers: int,
    chunks: int,
    baseline: dict | None,
    file_dict: dict[str, str],
    sha_dict: dict[str, str],
    moment: datetime,
) -> dict:
    """Return the release manifest with the SPEC S3.9 fields plus file hashes."""
    return {
        "name": name,
        "index_version": index_version,
        "collection": collection,
        "papers": papers,
        "chunks": chunks,
        "recall_at_5": baseline.get("value") if baseline else None,
        "git_sha": git_sha(),
        "created_utc": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": file_dict,
        "sha256": sha_dict,
    }


def require_repo() -> tuple[str, str]:
    """Return (repo id, token) for the dataset repo or raise."""
    settings = get_settings()
    token = settings.hf_token.get_secret_value()
    if not settings.hf_index_repo or not token:
        raise RuntimeError("HF_INDEX_REPO and HF_TOKEN must be set to publish or restore")
    return settings.hf_index_repo, token


def publish(work_dir: Path | None = None) -> dict:
    """
    Snapshot the live collection and publish the release.

    Returns
    -------
    manifest : dict
        The release manifest that `latest.json` now points at.
    """
    from huggingface_hub import HfApi

    config = get_release_config()
    index_config = get_index_config()
    repo_id, token = require_repo()
    client = index_module.get_qdrant_client()
    collection = index_module.alias_target(client, index_config.alias)
    if collection is None:
        raise RuntimeError("no live collection to publish")
    index_version = index_module.parse_version(collection, index_config.collection_prefix) or 0
    moment = datetime.now(UTC)
    name = release_name(index_version, moment)
    local_dir = Path(work_dir or config.work_dir) / name
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = create_and_download(
        client, collection, local_dir / f"{name}.snapshot", config.upload_timeout_seconds
    )
    manifest_copy = copy_sqlite(
        Path(get_corpus_config().paths.manifest), local_dir / f"{name}.manifest.sqlite"
    )
    baseline_path = Path(get_gate_config().baseline_path)
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    baseline_copy = local_dir / f"{name}.baseline.json"
    baseline_copy.write_text(json.dumps(baseline or {}, indent=2) + "\n")

    file_dict = release_files(config, name)
    local_dict = {
        "snapshot": snapshot_path,
        "ingest_manifest": manifest_copy,
        "baseline": baseline_copy,
    }
    manifest = build_release_manifest(
        name,
        index_module.base_collection_name(collection),
        index_version,
        index_module.count_papers(client, collection),
        index_module.count_points(client, collection),
        baseline,
        file_dict,
        {key: sha256_of(path) for key, path in local_dict.items()},
        moment,
    )
    manifest_path = local_dir / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    local_dict["manifest"] = manifest_path

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=config.o_private, exist_ok=True)
    for key, path in local_dict.items():
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=file_dict[key],
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"release {name}: {key}",
        )
    pointer_path = local_dir / config.latest_pointer
    pointer_path.write_text(json.dumps(manifest, indent=2) + "\n")
    api.upload_file(
        path_or_fileobj=str(pointer_path),
        path_in_repo=config.latest_pointer,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"latest -> {name}",
    )
    logger.info("release published", extra={"release": name, "repo": repo_id})
    return manifest


def cli_publish(args) -> int:
    """`meridian release publish`."""
    manifest = publish(Path(args.work_dir))
    output = Path(args.work_dir) / "publish.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0
