"""Builders shared by tests: synthetic PDFs, a fake arXiv client, and wired ingest contexts."""

import re
from pathlib import Path

import pymupdf
from qdrant_client import QdrantClient

from meridian.common.config import get_corpus_config, get_gate_config, get_index_config
from meridian.ingest import index as index_module
from meridian.ingest.build import PaperIndexer
from meridian.ingest.manifest import Manifest
from meridian.ingest.pipeline import IngestContext
from meridian.ingest.records import PaperMeta
from meridian.retrieval.hybrid import HybridRetriever
from tests.fakes import FakeDenseEncoder, FakeReranker, FakeSparseEncoder

WORD_PATTERN = re.compile(r"\S+")


class WhitespaceTokenizer:
    """One token per whitespace-separated word, with offsets."""

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [match.span() for match in WORD_PATTERN.finditer(text)]}


def write_pdf(path: Path, section_list: list[tuple[str, str]]) -> None:
    """Write a PDF with one heading and body per page."""
    document = pymupdf.open()
    for heading, body in section_list:
        page = document.new_page()
        page.insert_text((72, 90), heading, fontsize=18)
        y = 130
        for line_start in range(0, len(body), 90):
            page.insert_text((72, y), body[line_start : line_start + 90], fontsize=10)
            y += 14
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    document.close()


def paper_meta(
    arxiv_id: str, version: int = 1, published: str = "2024-11-04T10:00:00Z"
) -> PaperMeta:
    """Return metadata for a synthetic paper."""
    return PaperMeta(
        arxiv_id=arxiv_id,
        version=version,
        title=f"Paper {arxiv_id}",
        authors=["A. Author"],
        published=published,
        updated=published,
        primary_category="cs.CL",
        categories=["cs.CL"],
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v{version}",
    )


class FakeArxiv:
    """In-memory stand-in for ArxivClient serving synthetic papers."""

    def __init__(self, body_by_id: dict[str, str]) -> None:
        self.body_by_id = body_by_id
        self.search_list: list[PaperMeta] = []
        self.latest_list: list[PaperMeta] = []
        self.query_config = get_corpus_config().arxiv
        self.download_count = 0

    def iter_search(self, query, sort_by, sort_order, limit=None):
        yield from self.search_list[:limit] if limit else self.search_list

    def fetch_by_ids(self, id_list):
        latest = {meta.arxiv_id: meta for meta in self.latest_list}
        return [latest[arxiv_id] for arxiv_id in id_list if arxiv_id in latest]

    def download_pdf(self, meta: PaperMeta, dest_path: Path) -> str:
        self.download_count += 1
        body = self.body_by_id[meta.versioned_id]
        write_pdf(dest_path, [("1 Introduction", body), ("2 Results", body)])
        return "0" * 64


def wired_context(tmp_path: Path, arxiv: FakeArxiv, retrieval_set_path: Path) -> IngestContext:
    """Return an ingest context over an in-memory collection with fake models."""
    corpus_config = get_corpus_config()
    paths = corpus_config.paths.model_copy(
        update={
            "pdf_dir": str(tmp_path / "pdfs"),
            "parsed_dir": str(tmp_path / "parsed"),
            "manifest": str(tmp_path / "manifest.sqlite"),
        }
    )
    corpus_config = corpus_config.model_copy(update={"paths": paths})
    index_config = get_index_config()
    client = QdrantClient(":memory:")
    index_module.create_index_collection(client, "meridian_v1", 32, {})
    index_module.switch_alias(client, index_config.alias, "meridian_v1")
    indexer = PaperIndexer(
        client,
        "meridian_v1",
        1,
        index_config,
        FakeDenseEncoder(),
        FakeSparseEncoder(),
        WhitespaceTokenizer(),
        510,
    )
    retriever = HybridRetriever(
        client,
        "meridian_v1",
        FakeDenseEncoder(),
        FakeSparseEncoder(),
        FakeReranker(),
        index_config.retrieval,
    )
    return IngestContext(
        corpus_config=corpus_config,
        index_config=index_config,
        gate_config=get_gate_config(),
        manifest=Manifest(paths.manifest),
        client=client,
        collection="meridian_v1",
        index_version=1,
        arxiv=arxiv,  # type: ignore[arg-type]
        baseline_path=str(tmp_path / "baseline.json"),
        retrieval_set_path=str(retrieval_set_path),
        batch_dir=tmp_path / "batches",
        alert_path=tmp_path / "alerts.jsonl",
        indexer=indexer,
        search=lambda query: retriever.search(query, mode="hybrid_rerank", k=10),
    )


# ============= Index lifecycle fixture =============

LIFECYCLE_FIXTURE = Path(__file__).parent / "fixtures" / "lifecycle_papers.json"
BODY_REPEAT = 10


def lifecycle_papers() -> list[dict]:
    """Return the five synthetic fixture papers."""
    import json

    return json.loads(LIFECYCLE_FIXTURE.read_text())["papers"]


def fake_indexer_factory(client, collection, index_version, index_config) -> PaperIndexer:
    """Indexer factory with fake models and a whitespace tokenizer."""
    return PaperIndexer(
        client,
        collection,
        index_version,
        index_config,
        FakeDenseEncoder(),
        FakeSparseEncoder(),
        WhitespaceTokenizer(),
        510,
    )


def fake_search_factory(client):
    """Search factory with fake models, for rebuild tests."""

    def factory(collection, index_config):
        retriever = HybridRetriever(
            client,
            collection,
            FakeDenseEncoder(),
            FakeSparseEncoder(),
            FakeReranker(),
            index_config.retrieval,
        )
        return lambda query: retriever.search(query, mode="hybrid_rerank", k=10)

    return factory


def acquire_fixture_papers(context: IngestContext, paper_list: list[dict]) -> None:
    """Write fixture PDFs and record them in the manifest as a backfill batch."""
    from meridian.ingest.parse import parse_and_cache
    from meridian.ingest.pipeline import KIND_BACKFILL

    manifest = context.manifest
    manifest.start_batch("bac-fixture", KIND_BACKFILL, "", "", "", 0)
    for paper in paper_list:
        meta = paper_meta(paper["arxiv_id"])
        meta.title = paper["title"]
        pdf_path = Path(context.corpus_config.paths.pdf_dir) / f"{meta.versioned_id}.pdf"
        write_pdf(pdf_path, [(heading, body * BODY_REPEAT) for heading, body in paper["sections"]])
        parsed = parse_and_cache(
            pdf_path,
            context.corpus_config.paths.parsed_dir,
            meta.versioned_id,
            context.index_config.parser,
        )
        manifest.upsert_meta(meta, "parsed")
        manifest.update_paper(
            meta.arxiv_id,
            meta.version,
            pdf_path=str(pdf_path),
            status="parsed",
            parser_version=parsed.parser_version,
            ingest_batch_id="bac-fixture",
        )
    manifest.finish_batch("bac-fixture", "accepted", "2024-11-04T00:00:00Z", {})


def write_retrieval_set_for(client, collection: str, path: Path, query_by_paper: dict) -> None:
    """Label one R-fast item per paper: its question and the span of the paper's first chunk."""
    from meridian.eval.datasets import write_jsonl

    first_payload: dict[str, dict] = {}
    for point in index_module.iter_payloads(client, collection):
        payload = point.payload
        current = first_payload.get(payload["arxiv_id"])
        if current is None or payload["char_start"] < current["char_start"]:
            first_payload[payload["arxiv_id"]] = payload
    record_list = []
    for index, (arxiv_id, question) in enumerate(sorted(query_by_paper.items())):
        payload = first_payload[arxiv_id]
        record_list.append(
            {
                "qid": f"r{index}",
                "question": question,
                "spans": [
                    {
                        "arxiv_id": arxiv_id,
                        "version": payload["version"],
                        "start": payload["char_start"],
                        "end": payload["char_end"],
                    }
                ],
                "in_fast": True,
                "review": {"valid": None},
            }
        )
    write_jsonl(path, record_list)
