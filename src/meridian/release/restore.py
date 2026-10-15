"""Restore the accepted release listed in `latest.json`: download its snapshot, restore it into
Qdrant under the release's collection name, move the alias, and install the ingestion manifest,
the gate baseline, and `data/index_info.json` for the API footer.
"""

import json
import logging
import shutil
from pathlib import Path

from meridian.common.config import (
    get_corpus_config,
    get_gate_config,
    get_index_config,
    get_release_config,
    get_serving_config,
)
from meridian.ingest import index as index_module
from meridian.release.publish import require_repo
from meridian.release.snapshot import sha256_of, upload_and_restore

logger = logging.getLogger(__name__)


class ChecksumError(Exception):
    """A downloaded release file does not match its recorded SHA-256."""


def fetch_latest(repo_id: str, token: str, work_dir: Path) -> dict:
    """Download and parse `latest.json`."""
    from huggingface_hub import hf_hub_download

    config = get_release_config()
    path = hf_hub_download(
        repo_id,
        config.latest_pointer,
        repo_type="dataset",
        token=token,
        local_dir=str(work_dir),
        force_download=True,
    )
    return json.loads(Path(path).read_text())


def download_release(repo_id: str, token: str, manifest: dict, work_dir: Path) -> dict[str, Path]:
    """Download every file of a release and verify the recorded hashes."""
    from huggingface_hub import hf_hub_download

    path_dict: dict[str, Path] = {}
    for key, repo_path in manifest["files"].items():
        local = Path(
            hf_hub_download(
                repo_id, repo_path, repo_type="dataset", token=token, local_dir=str(work_dir)
            )
        )
        expected = manifest.get("sha256", {}).get(key)
        if expected and sha256_of(local) != expected:
            raise ChecksumError(f"{repo_path} does not match its recorded SHA-256")
        path_dict[key] = local
    return path_dict


def install_release(
    manifest: dict, path_dict: dict[str, Path], o_install_manifest: bool = True
) -> dict:
    """
    Restore the snapshot, move the alias, and install the side files.

    Returns
    -------
    info : dict
        What was restored, also written to the serving profile's `index_info_path`.
    """
    from datetime import UTC, datetime

    index_config = get_index_config()
    release_config = get_release_config()
    client = index_module.get_qdrant_client()
    # a fresh copy is restored and the alias then moves, so serving never sees a half restore
    collection = index_module.restore_collection_name(manifest["collection"], datetime.now(UTC))
    upload_and_restore(collection, path_dict["snapshot"], release_config.upload_timeout_seconds)
    previous = index_module.switch_alias(client, index_config.alias, collection)
    if (
        previous is not None
        and previous != collection
        and index_module.base_collection_name(previous) == manifest["collection"]
    ):
        client.delete_collection(previous)
    if o_install_manifest and "ingest_manifest" in path_dict:
        manifest_path = Path(get_corpus_config().paths.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path_dict["ingest_manifest"], manifest_path)
    baseline = json.loads(path_dict["baseline"].read_text()) if "baseline" in path_dict else {}
    if baseline:
        baseline_path = Path(get_gate_config().baseline_path)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    info = {
        "snapshot": manifest["name"],
        "collection": collection,
        "index_version": manifest["index_version"],
        "papers": manifest["papers"],
        "chunks": manifest["chunks"],
        "last_ingest": manifest["created_utc"],
        "previous_alias_target": previous,
    }
    info_path = Path(get_serving_config().index_info_path)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return info


def restore_latest(
    work_dir: Path, o_install_manifest: bool = True, current_snapshot: str | None = None
) -> dict | None:
    """
    Restore the latest release unless it is already live.

    Parameters
    ----------
    work_dir : Path
        Download directory.
    o_install_manifest : bool
        Also install the ingestion manifest (CI does; the demo Space does not need it).
    current_snapshot : str or None
        Name of the release already restored; when it equals the latest, nothing is done.

    Returns
    -------
    info : dict or None
        Restored release information, or None when already current.
    """
    repo_id, token = require_repo()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_latest(repo_id, token, work_dir)
    if current_snapshot is not None and manifest["name"] == current_snapshot:
        return None
    path_dict = download_release(repo_id, token, manifest, work_dir)
    info = install_release(manifest, path_dict, o_install_manifest)
    logger.info("release restored", extra={"release": manifest["name"]})
    return info


def cli_restore(args) -> int:
    """`meridian release restore`."""
    info = restore_latest(Path(args.work_dir))
    print(json.dumps(info, indent=2))
    return 0
