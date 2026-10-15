"""Backfill and daily incremental ingestion with deduplication, version replacement, the fast
regression gate, and batch rollback (SPEC S3.6).

An increment fetches papers submitted since the last accepted watermark (oldest first, capped),
skips versions already in the manifest, downloads, parses, chunks, embeds, and upserts them into
the collection behind `meridian_live` under a new `ingest_batch_id`, then runs R-fast. A failing
gate deletes every point of the batch; a passing gate removes superseded versions' points and
advances the watermark. Every batch writes a record.
"""

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qdrant_client import QdrantClient, models

from meridian.common.config import (
    CorpusConfig,
    GateConfig,
    IndexConfig,
    file_sha256,
    get_corpus_config,
    get_gate_config,
    get_index_config,
)
from meridian.eval.retrieval_eval import SearchFunction
from meridian.ingest import index as index_module
from meridian.ingest.arxiv_client import (
    ArxivClient,
    ArxivRequestError,
    PdfUnavailableError,
    backfill_query,
    increment_query,
    read_seed_ids,
    to_datetime,
)
from meridian.ingest.build import PaperIndexer
from meridian.ingest.gate import GateResult, run_fast_gate, save_baseline
from meridian.ingest.manifest import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_REJECTED,
    STATUS_FETCHED,
    STATUS_INDEXED,
    STATUS_PARSED,
    STATUS_REJECTED,
    STATUS_WITHDRAWN,
    Manifest,
    utc_now,
)
from meridian.ingest.parse import ParsedPaper, ParseError, parse_and_cache
from meridian.ingest.records import PaperMeta

logger = logging.getLogger(__name__)

# ============= Constants =============

KIND_BACKFILL = "backfill"
KIND_INCREMENT = "increment"
DECISION_DRY_RUN = "dry_run"
EXIT_ACCEPTED = 0
EXIT_REJECTED = 2
EXIT_ABORTED = 3


class DeadlineExceededError(Exception):
    """The run passed its deadline; the partial batch was rolled back."""


# ============= Context =============


@dataclass
class IngestContext:
    """Everything an ingestion run reads or writes, injectable for tests and simulations."""

    corpus_config: CorpusConfig
    index_config: IndexConfig
    gate_config: GateConfig
    manifest: Manifest
    client: QdrantClient
    collection: str
    index_version: int
    arxiv: ArxivClient
    baseline_path: str
    retrieval_set_path: str
    batch_dir: Path
    alert_path: Path
    indexer: PaperIndexer | None = None
    search: SearchFunction | None = None
    excluded_version_ids: set[str] = field(default_factory=set)

    def get_indexer(self) -> PaperIndexer:
        """Return the indexer for the target collection, loading models on first use."""
        if self.indexer is None:
            self.indexer = PaperIndexer(
                self.client, self.collection, self.index_version, self.index_config
            )
        return self.indexer

    def get_search(self) -> SearchFunction:
        """Return the full-pipeline search used by the gate."""
        if self.search is None:
            from meridian.retrieval.factory import build_retriever

            retriever = build_retriever(self.index_config, self.client, self.collection)

            def search(query: str):
                return retriever.search(query, mode="hybrid_rerank", k=10)

            self.search = search
        return self.search


def default_context(collection: str | None = None) -> IngestContext:
    """Build the context for the live collection from configs and settings."""
    from meridian.eval.datasets import evaluation_paper_ids

    corpus_config = get_corpus_config()
    index_config = get_index_config()
    gate_config = get_gate_config()
    client = index_module.get_qdrant_client()
    target = collection or index_module.alias_target(client, index_config.alias)
    if target is None:
        raise RuntimeError(f"alias {index_config.alias} does not exist; build an index first")
    try:
        excluded = evaluation_paper_ids()
    except FileNotFoundError:
        excluded = set()
    return IngestContext(
        corpus_config=corpus_config,
        index_config=index_config,
        gate_config=gate_config,
        manifest=Manifest(corpus_config.paths.manifest),
        client=client,
        collection=target,
        index_version=index_module.parse_version(target, index_config.collection_prefix) or 1,
        arxiv=ArxivClient(corpus_config),
        baseline_path=gate_config.baseline_path,
        retrieval_set_path=gate_config.retrieval_set_path,
        batch_dir=Path("data/batches"),
        alert_path=Path("data/alerts.jsonl"),
        excluded_version_ids=excluded,
    )


# ============= Records =============


@dataclass
class BatchRecord:
    """The record every batch writes (SPEC S3.6)."""

    batch_id: str
    kind: str
    started_utc: str
    since: str
    until: str
    collection: str
    index_version: int
    papers_fetched: int = 0
    papers_skipped: int = 0
    papers_failed: int = 0
    papers_withdrawn: int = 0
    papers_indexed: int = 0
    versions_replaced: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    eval: dict | None = None
    decision: str = ""
    watermark: str = ""
    o_window_exhausted: bool = False
    o_gate_skipped: bool = False
    finished_utc: str = ""
    duration_s: float = 0.0
    failure_dict: dict[str, str] = field(default_factory=dict)
    paper_id_list: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize."""
        return asdict(self)


def write_batch_record(context: IngestContext, record: BatchRecord) -> Path:
    """Write the record to the manifest and to `data/batches/{batch_id}.json`."""
    from meridian.common.metrics import INGEST_BATCHES_TOTAL

    INGEST_BATCHES_TOTAL.labels(decision=record.decision).inc()
    context.manifest.finish_batch(
        record.batch_id, record.decision, record.watermark, record.to_dict()
    )
    context.batch_dir.mkdir(parents=True, exist_ok=True)
    path = context.batch_dir / f"{record.batch_id}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def emit_alert(context: IngestContext, record: BatchRecord, message: str) -> None:
    """Log an alert and append it to the alert log read by CI."""
    alert_dict = {
        "ts": utc_now(),
        "batch_id": record.batch_id,
        "decision": record.decision,
        "message": message,
        "eval": record.eval,
    }
    # "message" is reserved on log records, so the alert text is logged under another key
    logger.error(
        "ingest alert",
        extra={"batch_id": record.batch_id, "decision": record.decision, "alert": message},
    )
    context.alert_path.parent.mkdir(parents=True, exist_ok=True)
    with open(context.alert_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(alert_dict, sort_keys=True) + "\n")


def new_batch_id(kind: str) -> str:
    """Return a unique, sortable batch id."""
    return f"{kind[:3]}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


# ============= Paper acquisition =============


def acquire_paper(
    context: IngestContext,
    meta: PaperMeta,
    record: BatchRecord,
    legacy_pdf_dir: Path | None = None,
) -> ParsedPaper | None:
    """
    Record, download, and parse one paper version.

    Parameters
    ----------
    context : IngestContext
        Run context.
    meta : PaperMeta
        Paper version to acquire.
    record : BatchRecord
        Batch record whose counters are updated.
    legacy_pdf_dir : Path or None
        Directory of previously downloaded `{id}v{ver}.pdf` files reused instead of fetching.

    Returns
    -------
    paper : ParsedPaper or None
        The parse, or None when the version is withdrawn or failed.
    """
    manifest = context.manifest
    paths = context.corpus_config.paths
    manifest.upsert_meta(meta, STATUS_FETCHED)
    if meta.o_withdrawn:
        manifest.update_paper(meta.arxiv_id, meta.version, status=STATUS_WITHDRAWN)
        record.papers_withdrawn += 1
        return None
    pdf_path = Path(paths.pdf_dir) / f"{meta.versioned_id}.pdf"
    try:
        if pdf_path.exists():
            pdf_sha256 = file_sha256(pdf_path)
        elif legacy_pdf_dir is not None and (legacy_pdf_dir / pdf_path.name).exists():
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_pdf_dir / pdf_path.name, pdf_path)
            pdf_sha256 = file_sha256(pdf_path)
        else:
            pdf_sha256 = context.arxiv.download_pdf(meta, pdf_path)
        manifest.update_paper(
            meta.arxiv_id, meta.version, pdf_path=str(pdf_path), pdf_sha256=pdf_sha256
        )
        paper = parse_and_cache(
            pdf_path, paths.parsed_dir, meta.versioned_id, context.index_config.parser
        )
    except PdfUnavailableError as exc:
        manifest.update_paper(meta.arxiv_id, meta.version, status=STATUS_WITHDRAWN, error=str(exc))
        record.papers_withdrawn += 1
        return None
    except (ArxivRequestError, ParseError, OSError) as exc:
        logger.warning("paper failed", extra={"paper": meta.versioned_id, "error": str(exc)})
        manifest.record_failure(meta.arxiv_id, meta.version, str(exc))
        record.papers_failed += 1
        record.failure_dict[meta.versioned_id] = str(exc)
        return None
    stats = paper.stats()
    manifest.update_paper(
        meta.arxiv_id,
        meta.version,
        status=STATUS_PARSED,
        parse_method=stats["parse_method"],
        o_parse_fallback=int(stats["o_fallback"]),
        parser_version=stats["parser_version"],
        page_count=stats["page_count"],
        section_count=stats["section_count"],
        char_count=stats["char_count"],
        ingest_batch_id=record.batch_id,
        error="",
    )
    return paper


# ============= Backfill =============


def run_backfill(
    context: IngestContext,
    source: str = "seed-ids",
    max_papers: int | None = None,
    legacy_pdf_dir: Path | None = None,
) -> BatchRecord:
    """
    Fetch, download, and parse the initial corpus into the manifest without indexing it.

    Parameters
    ----------
    context : IngestContext
        Run context; `collection` is recorded but not written.
    source : str
        `seed-ids` reproduces the recorded 199-paper corpus by versioned id; `query` re-runs
        the relevance-sorted backfill query.
    max_papers : int or None
        Cap on papers; defaults to the configured `max_results` for the query source.
    legacy_pdf_dir : Path or None
        Reuse PDFs from an earlier download directory.

    Returns
    -------
    record : BatchRecord
        Accepted backfill record. Its watermark is the run time, so increments start from
        the backfill date rather than the backfill ceiling.
    """
    query_config = context.corpus_config.arxiv
    started = time.perf_counter()
    batch_id = new_batch_id(KIND_BACKFILL)
    record = BatchRecord(
        batch_id=batch_id,
        kind=KIND_BACKFILL,
        started_utc=utc_now(),
        since=query_config.date_floor.isoformat(),
        until=query_config.backfill.date_ceiling.isoformat(),
        collection="",
        index_version=0,
    )
    context.manifest.start_batch(batch_id, KIND_BACKFILL, record.since, record.until, "", 0)
    if source == "seed-ids":
        meta_list = context.arxiv.fetch_by_ids(read_seed_ids(query_config.backfill.seed_ids_path))
    else:
        meta_list = list(
            context.arxiv.iter_search(
                backfill_query(query_config),
                query_config.backfill.sort_by,
                query_config.backfill.sort_order,
                limit=max_papers or query_config.backfill.max_results,
            )
        )
    if max_papers is not None:
        meta_list = meta_list[:max_papers]
    for meta in meta_list:
        record.papers_fetched += 1
        if context.manifest.should_skip(meta):
            record.papers_skipped += 1
            continue
        if acquire_paper(context, meta, record, legacy_pdf_dir) is not None:
            record.paper_id_list.append(meta.versioned_id)
    record.decision = DECISION_ACCEPTED
    record.watermark = utc_now()
    record.finished_utc = utc_now()
    record.duration_s = time.perf_counter() - started
    write_batch_record(context, record)
    return record


# ============= Increment =============


def resolve_since(manifest: Manifest, since: str) -> datetime:
    """Turn `last_success` or an ISO date into a datetime."""
    if since == "last_success":
        watermark = manifest.last_success_watermark()
        if watermark is None:
            raise RuntimeError("no accepted batch yet; run a backfill or pass --since <date>")
        return to_datetime(watermark)
    return to_datetime(since)


def collect_candidates(
    context: IngestContext,
    window_start: datetime,
    window_end: datetime,
    max_papers: int,
    record: BatchRecord,
) -> list[PaperMeta]:
    """
    Return new paper versions for this run: newer versions of corpus papers first, then new
    submissions in the window, oldest first, up to `max_papers` in total.

    Evaluation papers keep the version their labels refer to, so they are not re-versioned.
    """
    manifest = context.manifest
    query_config = context.corpus_config.arxiv
    candidate_list: list[PaperMeta] = []
    if query_config.increment.o_check_versions:
        current_dict = {row.meta.arxiv_id: row.meta.version for row in manifest.corpus_papers()}
        check_id_list = [
            arxiv_id for arxiv_id in current_dict if arxiv_id not in context.excluded_version_ids
        ]
        for meta in context.arxiv.fetch_by_ids(check_id_list) if check_id_list else []:
            if meta.version > current_dict.get(meta.arxiv_id, meta.version) and not (
                manifest.should_skip(meta)
            ):
                candidate_list.append(meta)
    record.versions_replaced = len(candidate_list)
    record.o_window_exhausted = True
    for meta in context.arxiv.iter_search(
        increment_query(query_config, window_start, window_end),
        query_config.increment.sort_by,
        query_config.increment.sort_order,
    ):
        record.papers_fetched += 1
        if manifest.should_skip(meta) or any(
            meta.arxiv_id == other.arxiv_id for other in candidate_list
        ):
            record.papers_skipped += 1
            continue
        if len(candidate_list) >= max_papers:
            record.o_window_exhausted = False
            break
        candidate_list.append(meta)
    return candidate_list


def delete_superseded_points(context: IngestContext, meta: PaperMeta) -> int:
    """Delete points of other versions of a paper from the target collection."""
    query_filter = models.Filter(
        must=[models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=meta.arxiv_id))],
        must_not=[
            models.FieldCondition(key="version", match=models.MatchValue(value=meta.version))
        ],
    )
    return index_module.delete_by_filter(context.client, context.collection, query_filter)


def rollback_batch(context: IngestContext, record: BatchRecord, status: str) -> None:
    """Delete every point of a batch and mark its papers with `status`."""
    record.chunks_deleted = index_module.delete_batch(
        context.client, context.collection, record.batch_id
    )
    for row in context.manifest.papers_in_batch(record.batch_id):
        context.manifest.update_paper(
            row.meta.arxiv_id, row.meta.version, status=status, chunk_count=0
        )


def run_increment(
    context: IngestContext,
    since: str = "last_success",
    until: str | None = None,
    max_papers: int | None = None,
    deadline_minutes: float | None = None,
    o_skip_gate: bool = False,
    batch_id: str | None = None,
    o_dry_run: bool = False,
) -> BatchRecord:
    """
    Run one gated incremental batch.

    Parameters
    ----------
    context : IngestContext
        Run context with the target (live) collection.
    since : str
        `last_success` or an ISO date or datetime; the window starts `lookback_days` earlier.
    until : str or None
        Inclusive ISO upper bound; defaults to now.
    max_papers : int or None
        Cap on new paper versions; defaults to `max_papers_per_run`.
    deadline_minutes : float or None
        Abort and roll back when the run exceeds this duration.
    o_skip_gate : bool
        Accept without running R-fast (manual runs only; recorded in the batch record).
    batch_id : str or None
        Explicit batch id.
    o_dry_run : bool
        Only list candidates; nothing is downloaded or written.

    Returns
    -------
    record : BatchRecord
        The batch record, also written to the manifest and `data/batches/`.

    Raises
    ------
    DeadlineExceededError
        After rolling back a batch that passed its deadline.
    """
    started = time.perf_counter()
    query_config = context.corpus_config.arxiv
    since_moment = resolve_since(context.manifest, since)
    window_start = since_moment - timedelta(days=query_config.increment.lookback_days)
    window_end = (
        to_datetime(until).replace(hour=23, minute=59)
        if until
        else datetime.now(UTC).replace(tzinfo=None)
    )
    batch = batch_id or new_batch_id(KIND_INCREMENT)
    record = BatchRecord(
        batch_id=batch,
        kind=KIND_INCREMENT,
        started_utc=utc_now(),
        since=window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        until=window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        collection=context.collection,
        index_version=context.index_version,
    )
    candidate_list = collect_candidates(
        context,
        window_start,
        window_end,
        max_papers or query_config.increment.max_papers_per_run,
        record,
    )
    if o_dry_run:
        record.decision = DECISION_DRY_RUN
        record.paper_id_list = [meta.versioned_id for meta in candidate_list]
        record.duration_s = time.perf_counter() - started
        return record

    context.manifest.start_batch(
        batch, KIND_INCREMENT, record.since, record.until, context.collection, context.index_version
    )
    deadline = started + deadline_minutes * 60 if deadline_minutes else None
    indexed_meta_list: list[PaperMeta] = []
    for meta in candidate_list:
        if deadline is not None and time.perf_counter() > deadline:
            rollback_batch(context, record, STATUS_PARSED)
            record.decision = DECISION_ABORTED
            record.finished_utc = utc_now()
            record.duration_s = time.perf_counter() - started
            write_batch_record(context, record)
            emit_alert(context, record, f"deadline of {deadline_minutes} minutes exceeded")
            raise DeadlineExceededError(f"batch {batch} rolled back after the deadline")
        paper = acquire_paper(context, meta, record)
        if paper is None:
            continue
        chunk_count = context.get_indexer().index_paper(meta, paper, batch)
        context.manifest.update_paper(
            meta.arxiv_id,
            meta.version,
            status=STATUS_INDEXED,
            ingest_batch_id=batch,
            index_version=context.index_version,
            chunk_count=chunk_count,
        )
        record.chunks_added += chunk_count
        record.papers_indexed += 1
        record.paper_id_list.append(meta.versioned_id)
        indexed_meta_list.append(meta)

    gate_result: GateResult | None = None
    if record.chunks_added == 0:
        record.decision = DECISION_ACCEPTED
    elif o_skip_gate:
        record.o_gate_skipped = True
        record.decision = DECISION_ACCEPTED
    else:
        gate_result = run_fast_gate(
            context.get_search(),
            context.gate_config,
            context.collection,
            context.baseline_path,
            context.retrieval_set_path,
        )
        record.eval = gate_result.to_dict()
        record.decision = DECISION_ACCEPTED if gate_result.o_pass else DECISION_REJECTED

    if record.decision == DECISION_ACCEPTED:
        if gate_result is not None:
            save_baseline(context.baseline_path, gate_result, context.retrieval_set_path)
        for meta in indexed_meta_list:
            record.chunks_deleted += delete_superseded_points(context, meta)
            context.manifest.supersede_older(meta.arxiv_id, meta.version)
        # version updates come first in the candidate list and do not move the watermark
        window_candidate_list = candidate_list[record.versions_replaced :]
        if record.o_window_exhausted or not window_candidate_list:
            record.watermark = record.until
        else:
            record.watermark = max(meta.published for meta in window_candidate_list)
    else:
        rollback_batch(context, record, STATUS_REJECTED)
        emit_alert(
            context,
            record,
            gate_result.report() if gate_result is not None else "batch rejected",
        )
    record.finished_utc = utc_now()
    record.duration_s = time.perf_counter() - started
    write_batch_record(context, record)
    return record


# ============= CLI =============


def cli_backfill(args) -> int:
    """`meridian backfill`."""
    corpus_config = get_corpus_config()
    context = IngestContext(
        corpus_config=corpus_config,
        index_config=get_index_config(),
        gate_config=get_gate_config(),
        manifest=Manifest(corpus_config.paths.manifest),
        client=index_module.get_qdrant_client(),
        collection="",
        index_version=0,
        arxiv=ArxivClient(corpus_config),
        baseline_path=get_gate_config().baseline_path,
        retrieval_set_path=get_gate_config().retrieval_set_path,
        batch_dir=Path("data/batches"),
        alert_path=Path("data/alerts.jsonl"),
    )
    legacy_dir = Path(args.legacy_pdf_dir) if args.legacy_pdf_dir else None
    record = run_backfill(context, args.source, args.max_papers, legacy_dir)
    print(json.dumps(record.to_dict(), indent=2))
    return 0 if not record.failure_dict else 1


def cli_ingest(args) -> int:
    """`meridian ingest`; exit 0 accepted, 2 rejected by the gate, 3 aborted at the deadline."""
    context = default_context()
    try:
        record = run_increment(
            context,
            since=args.since,
            until=args.until,
            max_papers=args.max_papers,
            deadline_minutes=args.deadline_minutes,
            o_skip_gate=args.o_skip_gate,
            batch_id=args.batch_id,
            o_dry_run=args.o_dry_run,
        )
    except DeadlineExceededError as exc:
        print(json.dumps({"decision": DECISION_ABORTED, "error": str(exc)}))
        return EXIT_ABORTED
    print(json.dumps(record.to_dict(), indent=2))
    return EXIT_REJECTED if record.decision == DECISION_REJECTED else EXIT_ACCEPTED
