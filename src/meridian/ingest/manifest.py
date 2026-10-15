"""The ingestion manifest `data/manifest.sqlite`: one row per paper version and per batch.

The manifest is the source of truth for deduplication (a paper version already present is
skipped), for PDF integrity (SHA-256), for parse statistics, and for batch records with gate
decisions.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from meridian.ingest.records import PaperMeta

# ============= Constants =============

STATUS_FETCHED = "fetched"
STATUS_PARSED = "parsed"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUS_WITHDRAWN = "withdrawn"
STATUS_SUPERSEDED = "superseded"
STATUS_REJECTED = "rejected"

# a paper version in one of these states is never fetched again
TERMINAL_STATUS_SET = {STATUS_INDEXED, STATUS_WITHDRAWN, STATUS_SUPERSEDED}
# papers in these states are part of the corpus a rebuild indexes
CORPUS_STATUS_SET = {STATUS_PARSED, STATUS_INDEXED}
MAX_FAILED_ATTEMPTS = 3

DECISION_PENDING = "pending"
DECISION_ACCEPTED = "accepted"
DECISION_REJECTED = "rejected"
DECISION_ABORTED = "aborted"

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    published TEXT NOT NULL,
    updated TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    comment TEXT NOT NULL DEFAULT '',
    pdf_url TEXT NOT NULL DEFAULT '',
    pdf_path TEXT NOT NULL DEFAULT '',
    pdf_sha256 TEXT NOT NULL DEFAULT '',
    parse_method TEXT NOT NULL DEFAULT '',
    o_parse_fallback INTEGER NOT NULL DEFAULT 0,
    parser_version INTEGER NOT NULL DEFAULT 0,
    page_count INTEGER NOT NULL DEFAULT 0,
    section_count INTEGER NOT NULL DEFAULT 0,
    char_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    ingest_batch_id TEXT NOT NULL DEFAULT '',
    index_version INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, version)
);
CREATE INDEX IF NOT EXISTS papers_status ON papers (status);
CREATE INDEX IF NOT EXISTS papers_batch ON papers (ingest_batch_id);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL DEFAULT '',
    since TEXT NOT NULL DEFAULT '',
    until TEXT NOT NULL DEFAULT '',
    watermark TEXT NOT NULL DEFAULT '',
    collection TEXT NOT NULL DEFAULT '',
    index_version INTEGER NOT NULL DEFAULT 0,
    decision TEXT NOT NULL,
    record_json TEXT NOT NULL DEFAULT '{}'
);
"""


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with a Z suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============= Rows =============


@dataclass
class PaperRow:
    """One manifest row."""

    meta: PaperMeta
    pdf_path: str
    pdf_sha256: str
    parse_method: str
    o_parse_fallback: bool
    parser_version: int
    page_count: int
    section_count: int
    char_count: int
    chunk_count: int
    status: str
    attempts: int
    ingest_batch_id: str
    index_version: int
    error: str


def _row_to_paper(row: sqlite3.Row) -> PaperRow:
    meta = PaperMeta(
        arxiv_id=row["arxiv_id"],
        version=row["version"],
        title=row["title"],
        authors=json.loads(row["authors_json"]),
        published=row["published"],
        updated=row["updated"],
        primary_category=row["primary_category"],
        categories=json.loads(row["categories_json"]),
        abstract=row["abstract"],
        comment=row["comment"],
        pdf_url=row["pdf_url"],
        o_withdrawn=row["status"] == STATUS_WITHDRAWN,
    )
    return PaperRow(
        meta=meta,
        pdf_path=row["pdf_path"],
        pdf_sha256=row["pdf_sha256"],
        parse_method=row["parse_method"],
        o_parse_fallback=bool(row["o_parse_fallback"]),
        parser_version=row["parser_version"],
        page_count=row["page_count"],
        section_count=row["section_count"],
        char_count=row["char_count"],
        chunk_count=row["chunk_count"],
        status=row["status"],
        attempts=row["attempts"],
        ingest_batch_id=row["ingest_batch_id"],
        index_version=row["index_version"],
        error=row["error"],
    )


# ============= Manifest =============


class Manifest:
    """SQLite-backed manifest of paper versions and ingest batches."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        """Close the connection."""
        self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the connection inside a transaction."""
        with self._connection:
            yield self._connection

    # ---------- papers ----------

    def get_paper(self, arxiv_id: str, version: int) -> PaperRow | None:
        """Return one paper version, or None."""
        row = self._connection.execute(
            "SELECT * FROM papers WHERE arxiv_id = ? AND version = ?", (arxiv_id, version)
        ).fetchone()
        return _row_to_paper(row) if row else None

    def versions_of(self, arxiv_id: str) -> list[PaperRow]:
        """Return every recorded version of a paper, oldest first."""
        row_list = self._connection.execute(
            "SELECT * FROM papers WHERE arxiv_id = ? ORDER BY version", (arxiv_id,)
        ).fetchall()
        return [_row_to_paper(row) for row in row_list]

    def should_skip(self, meta: PaperMeta) -> bool:
        """
        Return whether a fetched paper version needs no work.

        A version is skipped when it is already indexed, withdrawn, or superseded, when it has
        failed `MAX_FAILED_ATTEMPTS` times, or when a newer version is already recorded.
        """
        row = self.get_paper(meta.arxiv_id, meta.version)
        if row is not None:
            if row.status in TERMINAL_STATUS_SET:
                return True
            if row.status == STATUS_FAILED and row.attempts >= MAX_FAILED_ATTEMPTS:
                return True
        newer_list = [
            other
            for other in self.versions_of(meta.arxiv_id)
            if other.meta.version > meta.version and other.status != STATUS_FAILED
        ]
        return bool(newer_list)

    def upsert_meta(self, meta: PaperMeta, status: str) -> None:
        """Insert a paper version, or refresh its metadata and status."""
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO papers (arxiv_id, version, title, authors_json, published, updated,
                    primary_category, categories_json, abstract, comment, pdf_url, status,
                    created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (arxiv_id, version) DO UPDATE SET
                    title = excluded.title, authors_json = excluded.authors_json,
                    published = excluded.published, updated = excluded.updated,
                    primary_category = excluded.primary_category,
                    categories_json = excluded.categories_json, abstract = excluded.abstract,
                    comment = excluded.comment, pdf_url = excluded.pdf_url,
                    status = excluded.status, updated_at = excluded.updated_at
                """,
                (
                    meta.arxiv_id,
                    meta.version,
                    meta.title,
                    json.dumps(meta.authors),
                    meta.published,
                    meta.updated,
                    meta.primary_category,
                    json.dumps(meta.categories),
                    meta.abstract,
                    meta.comment,
                    meta.pdf_url,
                    status,
                    now,
                    now,
                ),
            )

    def update_paper(self, arxiv_id: str, version: int, **field_dict) -> None:
        """Update named columns of one paper version."""
        if not field_dict:
            return
        allowed_set = {
            "pdf_path",
            "pdf_sha256",
            "parse_method",
            "o_parse_fallback",
            "parser_version",
            "page_count",
            "section_count",
            "char_count",
            "chunk_count",
            "status",
            "attempts",
            "ingest_batch_id",
            "index_version",
            "error",
        }
        unknown_set = set(field_dict) - allowed_set
        if unknown_set:
            raise ValueError(f"unknown manifest columns: {sorted(unknown_set)}")
        assignment = ", ".join(f"{name} = ?" for name in field_dict)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE papers SET {assignment}, updated_at = ? "
                "WHERE arxiv_id = ? AND version = ?",
                (*field_dict.values(), utc_now(), arxiv_id, version),
            )

    def record_failure(self, arxiv_id: str, version: int, error: str) -> None:
        """Mark a paper version failed and count the attempt."""
        with self.transaction() as connection:
            connection.execute(
                "UPDATE papers SET status = ?, attempts = attempts + 1, error = ?, updated_at = ? "
                "WHERE arxiv_id = ? AND version = ?",
                (STATUS_FAILED, error[:2000], utc_now(), arxiv_id, version),
            )

    def supersede_older(self, arxiv_id: str, version: int) -> list[int]:
        """Mark every older version of a paper superseded and return their numbers."""
        older_list = [
            row.meta.version
            for row in self.versions_of(arxiv_id)
            if row.meta.version < version and row.status != STATUS_SUPERSEDED
        ]
        for older_version in older_list:
            self.update_paper(arxiv_id, older_version, status=STATUS_SUPERSEDED)
        return older_list

    def papers_with_status(self, status_set: set[str]) -> list[PaperRow]:
        """Return paper versions whose status is in `status_set`, ordered by id and version."""
        placeholder = ", ".join("?" for _ in status_set)
        row_list = self._connection.execute(
            f"SELECT * FROM papers WHERE status IN ({placeholder}) ORDER BY arxiv_id, version",
            tuple(sorted(status_set)),
        ).fetchall()
        return [_row_to_paper(row) for row in row_list]

    def corpus_papers(self) -> list[PaperRow]:
        """Return the papers a rebuild indexes: parsed or indexed, latest version only."""
        latest_dict: dict[str, PaperRow] = {}
        for row in self.papers_with_status(CORPUS_STATUS_SET):
            current = latest_dict.get(row.meta.arxiv_id)
            if current is None or row.meta.version > current.meta.version:
                latest_dict[row.meta.arxiv_id] = row
        return [latest_dict[arxiv_id] for arxiv_id in sorted(latest_dict)]

    def papers_in_batch(self, batch_id: str) -> list[PaperRow]:
        """Return the paper versions written by one batch."""
        row_list = self._connection.execute(
            "SELECT * FROM papers WHERE ingest_batch_id = ? ORDER BY arxiv_id", (batch_id,)
        ).fetchall()
        return [_row_to_paper(row) for row in row_list]

    def count_by_status(self) -> dict[str, int]:
        """Return the number of paper versions per status."""
        row_list = self._connection.execute(
            "SELECT status, COUNT(*) AS n FROM papers GROUP BY status"
        ).fetchall()
        return {row["status"]: row["n"] for row in row_list}

    # ---------- batches ----------

    def start_batch(
        self, batch_id: str, kind: str, since: str, until: str, collection: str, index_version: int
    ) -> None:
        """Insert a pending batch."""
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO batches (batch_id, kind, started_at, since, until, collection, "
                "index_version, decision) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    kind,
                    utc_now(),
                    since,
                    until,
                    collection,
                    index_version,
                    DECISION_PENDING,
                ),
            )

    def finish_batch(self, batch_id: str, decision: str, watermark: str, record: dict) -> None:
        """Record a batch decision, its watermark, and its full record."""
        with self.transaction() as connection:
            connection.execute(
                "UPDATE batches SET finished_at = ?, decision = ?, watermark = ?, record_json = ? "
                "WHERE batch_id = ?",
                (utc_now(), decision, watermark, json.dumps(record, sort_keys=True), batch_id),
            )

    def get_batch(self, batch_id: str) -> dict | None:
        """Return one batch as a dict with its parsed record."""
        row = self._connection.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            return None
        batch_dict = dict(row)
        batch_dict["record"] = json.loads(batch_dict.pop("record_json"))
        return batch_dict

    def batches(self, kind: str | None = None) -> list[dict]:
        """Return every batch, oldest first, optionally filtered by kind."""
        query = "SELECT batch_id FROM batches"
        parameter_list: tuple = ()
        if kind is not None:
            query += " WHERE kind = ?"
            parameter_list = (kind,)
        # rowid breaks ties between batches started within the same second
        row_list = self._connection.execute(query + " ORDER BY started_at, rowid", parameter_list)
        return [self.get_batch(row["batch_id"]) for row in row_list.fetchall()]  # type: ignore[misc]

    def last_success_watermark(self) -> str | None:
        """Return the watermark of the most recent accepted batch, or None."""
        row = self._connection.execute(
            "SELECT watermark FROM batches WHERE decision = ? AND watermark != '' "
            "ORDER BY finished_at DESC, rowid DESC LIMIT 1",
            (DECISION_ACCEPTED,),
        ).fetchone()
        return row["watermark"] if row else None
