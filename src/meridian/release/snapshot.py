"""Qdrant collection snapshots over the server's REST API: create and download, and upload and
restore. Local mode has no snapshots, so releases always use a Qdrant server.
"""

import hashlib
import logging
import sqlite3
from pathlib import Path

import httpx

from meridian.common.settings import get_settings

logger = logging.getLogger(__name__)

DOWNLOAD_CHUNK_BYTES = 1 << 20


def qdrant_base_url() -> str:
    """Return the Qdrant server URL; releases need server mode."""
    url = get_settings().qdrant_url
    if not url:
        raise RuntimeError("snapshots need a Qdrant server; set QDRANT_URL")
    return url.rstrip("/")


def qdrant_headers() -> dict[str, str]:
    """Return the API key header when one is configured."""
    api_key = get_settings().qdrant_api_key.get_secret_value()
    return {"api-key": api_key} if api_key else {}


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def create_and_download(client, collection: str, dest_path: Path, timeout_s: float) -> Path:
    """
    Snapshot a collection on the server and download it.

    Parameters
    ----------
    client : QdrantClient
        Server-mode client.
    collection : str
        Concrete collection name (not the alias).
    dest_path : Path
        Local file to write.
    timeout_s : float
        Download timeout.

    Returns
    -------
    path : Path
        The downloaded snapshot. The server-side snapshot is deleted afterwards.
    """
    description = client.create_snapshot(collection_name=collection, wait=True)
    if description is None:
        raise RuntimeError(f"Qdrant did not return a snapshot for {collection}")
    url = f"{qdrant_base_url()}/collections/{collection}/snapshots/{description.name}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, headers=qdrant_headers(), timeout=timeout_s) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as handle:
            for block in response.iter_bytes(DOWNLOAD_CHUNK_BYTES):
                handle.write(block)
    client.delete_snapshot(collection_name=collection, snapshot_name=description.name, wait=True)
    logger.info("snapshot downloaded", extra={"collection": collection, "path": str(dest_path)})
    return dest_path


def upload_and_restore(collection: str, snapshot_path: Path, timeout_s: float) -> None:
    """Upload a snapshot file into a collection with snapshot priority, replacing its data."""
    url = f"{qdrant_base_url()}/collections/{collection}/snapshots/upload"
    with open(snapshot_path, "rb") as handle:
        response = httpx.post(
            url,
            params={"priority": "snapshot", "wait": "true"},
            files={"snapshot": (snapshot_path.name, handle, "application/octet-stream")},
            headers=qdrant_headers(),
            timeout=timeout_s,
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"snapshot restore failed: HTTP {response.status_code} {response.text[:300]}"
        )
    logger.info("snapshot restored", extra={"collection": collection})


def copy_sqlite(source: Path, dest: Path) -> Path:
    """Copy a SQLite database consistently with the backup API."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    with (
        sqlite3.connect(str(source)) as source_connection,
        sqlite3.connect(str(dest)) as dest_connection,
    ):
        source_connection.backup(dest_connection)
    return dest
