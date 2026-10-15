"""Index construction: chunk, embed, and upsert parsed papers into a versioned collection.

`PaperIndexer` is shared by full builds (a new `meridian_v{N}` from every corpus paper in the
manifest) and by incremental ingestion (new papers into the live collection).
"""

import logging
import time
import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from meridian.common.config import (
    CorpusConfig,
    IndexConfig,
    get_corpus_config,
    get_index_config,
    load_config,
)
from meridian.ingest import index as index_module
from meridian.ingest.chunk import TokenizerLike, build_payload, chunk_paper, load_tokenizer
from meridian.ingest.embed import (
    DenseEncoder,
    SparseEncoder,
    get_dense_encoder,
    get_sparse_encoder,
)
from meridian.ingest.manifest import STATUS_INDEXED, Manifest, PaperRow
from meridian.ingest.parse import ParsedPaper, ParseError, parse_and_cache
from meridian.ingest.records import PaperMeta

logger = logging.getLogger(__name__)


# ============= Indexer =============


class PaperIndexer:
    """Chunks, embeds, and upserts papers into one collection."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        index_version: int,
        index_config: IndexConfig,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        tokenizer: TokenizerLike | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.index_version = index_version
        self.index_config = index_config
        self.dense_encoder = dense_encoder or get_dense_encoder(index_config.dense)
        self.sparse_encoder = sparse_encoder or get_sparse_encoder(index_config.sparse)
        if tokenizer is None or max_tokens is None:
            tokenizer, max_tokens = load_tokenizer(index_config.dense.model)
        self.tokenizer, self.max_tokens = tokenizer, max_tokens

    def payloads_for(self, meta: PaperMeta, paper: ParsedPaper, ingest_batch_id: str) -> list[dict]:
        """Return the validated payloads of one paper's chunks."""
        chunk_list = chunk_paper(
            paper, meta, self.tokenizer, self.max_tokens, self.index_config.chunker
        )
        return [
            build_payload(chunk, meta, ingest_batch_id, self.index_version, paper.parser_version)
            for chunk in chunk_list
        ]

    def index_paper(self, meta: PaperMeta, paper: ParsedPaper, ingest_batch_id: str) -> int:
        """
        Index one paper and return the number of chunks written.

        Parameters
        ----------
        meta : PaperMeta
            Paper metadata.
        paper : ParsedPaper
            Parsed text and sections.
        ingest_batch_id : str
            Batch id written into every payload, used for rollback.

        Returns
        -------
        chunk_count : int
            Points upserted.
        """
        payload_list = self.payloads_for(meta, paper, ingest_batch_id)
        if not payload_list:
            return 0
        text_list = [payload["text"] for payload in payload_list]
        point_list = index_module.build_points(
            payload_list,
            self.dense_encoder.encode_documents(text_list),
            self.sparse_encoder.encode_documents(text_list),
        )
        return index_module.upsert_points(
            self.client, self.collection, point_list, self.index_config.upsert_batch_size
        )


# ============= Parsing helper =============


def load_or_parse(
    row: PaperRow, corpus_config: CorpusConfig, index_config: IndexConfig
) -> ParsedPaper:
    """Return the cached parse of a manifest paper, re-parsing its PDF when needed."""
    return parse_and_cache(
        row.pdf_path,
        corpus_config.paths.parsed_dir,
        row.meta.versioned_id,
        index_config.parser,
    )


# ============= Full build =============


@dataclass
class BuildReport:
    """Outcome of a full collection build."""

    collection: str
    index_version: int
    papers_indexed: int = 0
    chunks_indexed: int = 0
    seconds: float = 0.0
    chunk_count_dict: dict[str, int] = field(default_factory=dict)
    failure_dict: dict[str, str] = field(default_factory=dict)


def build_collection(
    client: QdrantClient,
    manifest: Manifest,
    index_config: IndexConfig,
    corpus_config: CorpusConfig,
    collection: str | None = None,
    build_batch_id: str | None = None,
    index_version: int | None = None,
    indexer_factory=None,
) -> BuildReport:
    """
    Build a new versioned collection from every corpus paper in the manifest.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client.
    manifest : Manifest
        Manifest supplying corpus papers.
    index_config : IndexConfig
        Index configuration of the new version.
    corpus_config : CorpusConfig
        Corpus paths.
    collection : str or None
        Explicit collection name; defaults to the next `meridian_v{N}`.
    build_batch_id : str or None
        Batch id assigned to papers that were never indexed before.
    index_version : int or None
        Version recorded in payloads; required when `collection` is not a versioned name.
    indexer_factory : callable or None
        `(client, collection, index_version, index_config) -> PaperIndexer`; tests inject
        fake models through it. Defaults to the configured models.

    Returns
    -------
    report : BuildReport
        Counts, timing, and per-paper failures. The manifest is not changed here; the caller
        records the new version once the alias points at it.
    """
    started = time.perf_counter()
    prefix = index_config.collection_prefix
    if collection is None:
        index_version = index_module.next_version(client, prefix)
        collection = index_module.collection_name(prefix, index_version)
    elif index_version is None:
        index_version = index_module.parse_version(collection, prefix)
        if index_version is None:
            raise ValueError(f"{collection} is not a versioned name; pass index_version")
    batch_id = build_batch_id or f"build-{uuid.uuid4().hex[:12]}"

    if indexer_factory is None:
        indexer = PaperIndexer(client, collection, index_version, index_config)
    else:
        indexer = indexer_factory(client, collection, index_version, index_config)
    dense_dim = indexer.dense_encoder.dimension
    index_module.create_index_collection(
        client,
        collection,
        dense_dim,
        index_module.build_metadata(index_config, index_version, dense_dim),
    )
    report = BuildReport(collection=collection, index_version=index_version)
    for row in manifest.corpus_papers():
        versioned_id = row.meta.versioned_id
        try:
            paper = load_or_parse(row, corpus_config, index_config)
        except (ParseError, FileNotFoundError, OSError) as exc:
            logger.warning(
                "parse failed during build", extra={"paper": versioned_id, "error": str(exc)}
            )
            report.failure_dict[versioned_id] = str(exc)
            continue
        # papers keep their original batch id so batch rollback stays valid after a rebuild
        chunk_count = indexer.index_paper(row.meta, paper, row.ingest_batch_id or batch_id)
        report.chunk_count_dict[versioned_id] = chunk_count
        report.papers_indexed += 1
        report.chunks_indexed += chunk_count
    report.seconds = time.perf_counter() - started
    logger.info(
        "collection built",
        extra={
            "collection": collection,
            "papers": report.papers_indexed,
            "chunks": report.chunks_indexed,
            "failures": len(report.failure_dict),
            "seconds": round(report.seconds, 1),
        },
    )
    return report


def record_build_in_manifest(manifest: Manifest, report: BuildReport, build_batch_id: str) -> None:
    """Mark the built papers as indexed in the new version once it is live."""
    for row in manifest.corpus_papers():
        chunk_count = report.chunk_count_dict.get(row.meta.versioned_id)
        if chunk_count is None:
            continue
        manifest.update_paper(
            row.meta.arxiv_id,
            row.meta.version,
            status=STATUS_INDEXED,
            index_version=report.index_version,
            chunk_count=chunk_count,
            ingest_batch_id=row.ingest_batch_id or build_batch_id,
        )


def cli_build(args) -> int:
    """`meridian index build`: build a new collection, optionally pointing the alias at it."""
    import json

    index_config = (
        load_config(IndexConfig, args.index_config) if args.index_config else get_index_config()
    )
    corpus_config = get_corpus_config()
    client = index_module.get_qdrant_client()
    manifest = Manifest(corpus_config.paths.manifest)
    build_batch_id = f"build-{uuid.uuid4().hex[:12]}"
    report = build_collection(
        client, manifest, index_config, corpus_config, args.collection, build_batch_id
    )
    if args.o_switch_alias:
        if index_module.alias_target(client, index_config.alias) is not None:
            raise SystemExit(
                "the live alias already exists; use `meridian rebuild` so the full gate decides"
            )
        index_module.switch_alias(client, index_config.alias, report.collection)
        record_build_in_manifest(manifest, report, build_batch_id)
    print(
        json.dumps(
            {
                "collection": report.collection,
                "papers": report.papers_indexed,
                "chunks": report.chunks_indexed,
                "failures": report.failure_dict,
                "seconds": round(report.seconds, 1),
                "alias_switched": bool(args.o_switch_alias),
            },
            indent=2,
        )
    )
    return 0 if not report.failure_dict else 1
