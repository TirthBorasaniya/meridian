"""Prefect ingestion flows.

``fetch_papers_flow`` retrieves arXiv metadata, persists it to PostgreSQL, and
downloads PDFs. ``parse_and_index_flow`` parses fetched PDFs with Docling,
chunks the text, embeds the chunks, and upserts them into Qdrant. Per-paper
failures are recorded against the paper row and do not abort the flow.
"""

import httpx
from prefect import flow, get_run_logger, task

from meridian.db import (
    STATUS_FAILED,
    STATUS_FETCHED,
    STATUS_INDEXED,
    STATUS_PENDING,
    Paper,
    get_session_factory,
    init_db,
)
from meridian.ingestion.arxiv_client import PaperMetadata, download_pdf, fetch_paper_metadata
from meridian.ingestion.chunker import chunk_text
from meridian.ingestion.indexer import ensure_collection, index_chunks
from meridian.ingestion.pdf_parser import parse_pdf


@task
def task_fetch_metadata(max_papers: int) -> list[PaperMetadata]:
    """Fetch up to ``max_papers`` metadata records from the arXiv API."""
    logger = get_run_logger()
    metadata_list = fetch_paper_metadata(max_papers)
    logger.info(f"Fetched metadata for {len(metadata_list)} papers")
    return metadata_list


@task
def task_persist_metadata(metadata_list: list[PaperMetadata]) -> int:
    """Upsert metadata records into PostgreSQL with pending status."""
    session_factory = get_session_factory()
    with session_factory() as session:
        for metadata in metadata_list:
            existing = session.get(Paper, metadata.arxiv_id)
            if existing is None:
                session.add(
                    Paper(
                        arxiv_id=metadata.arxiv_id,
                        title=metadata.title,
                        authors=metadata.authors,
                        abstract=metadata.abstract,
                        categories=metadata.categories,
                        published=metadata.published,
                        ingestion_status=STATUS_PENDING,
                    )
                )
        session.commit()
    return len(metadata_list)


@task(retries=2, retry_delay_seconds=5)
def task_download_and_record(metadata: PaperMetadata) -> str:
    """Download a paper PDF and record the path and fetched status.

    Raises
    ------
    httpx.HTTPError
        If the download fails after retries; the flow records the failure.
    """
    pdf_path = download_pdf(metadata)
    session_factory = get_session_factory()
    with session_factory() as session:
        paper = session.get(Paper, metadata.arxiv_id)
        if paper is not None:
            paper.pdf_path = pdf_path
            paper.ingestion_status = STATUS_FETCHED
            session.commit()
    return pdf_path


@task
def task_parse_and_index(arxiv_id: str) -> int:
    """Parse, chunk, embed, and index a single fetched paper.

    Returns
    -------
    int
        The number of chunks indexed for the paper.
    """
    session_factory = get_session_factory()
    with session_factory() as session:
        paper = session.get(Paper, arxiv_id)
        if paper is None or not paper.pdf_path:
            return 0
        pdf_path = paper.pdf_path
        metadata_dict = {
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "year": paper.published.year,
        }

    text = parse_pdf(pdf_path, arxiv_id)
    chunk_list = chunk_text(text, arxiv_id)
    indexed_count = index_chunks(chunk_list, metadata_dict)

    with session_factory() as session:
        paper = session.get(Paper, arxiv_id)
        if paper is not None:
            paper.parsed_path = f"{arxiv_id}.md"
            paper.chunk_count = indexed_count
            paper.ingestion_status = STATUS_INDEXED
            session.commit()
    return indexed_count


def _mark_failed(arxiv_id: str) -> None:
    """Record a failed ingestion status for a paper."""
    session_factory = get_session_factory()
    with session_factory() as session:
        paper = session.get(Paper, arxiv_id)
        if paper is not None:
            paper.ingestion_status = STATUS_FAILED
            session.commit()


@flow(name="fetch-papers")
def fetch_papers_flow(max_papers: int = 200) -> int:
    """Fetch metadata and download PDFs for up to ``max_papers`` papers.

    Returns
    -------
    int
        The number of papers successfully downloaded.
    """
    logger = get_run_logger()
    init_db()
    metadata_list = task_fetch_metadata(max_papers)
    task_persist_metadata(metadata_list)

    downloaded_count = 0
    for metadata in metadata_list:
        try:
            task_download_and_record(metadata)
            downloaded_count += 1
        except httpx.HTTPError as exc:
            logger.warning(f"Download failed for {metadata.arxiv_id}: {exc}")
            _mark_failed(metadata.arxiv_id)
    logger.info(f"Downloaded {downloaded_count} of {len(metadata_list)} papers")
    return downloaded_count


@flow(name="parse-and-index")
def parse_and_index_flow() -> int:
    """Parse and index every fetched paper that is not yet indexed.

    Returns
    -------
    int
        The total number of chunks indexed across all processed papers.
    """
    logger = get_run_logger()
    ensure_collection()

    session_factory = get_session_factory()
    with session_factory() as session:
        arxiv_id_list = [
            paper.arxiv_id
            for paper in session.query(Paper)
            .filter(Paper.ingestion_status == STATUS_FETCHED)
            .all()
        ]

    total_chunks = 0
    for arxiv_id in arxiv_id_list:
        try:
            total_chunks += task_parse_and_index(arxiv_id)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning(f"Parse/index failed for {arxiv_id}: {exc}")
            _mark_failed(arxiv_id)
    logger.info(f"Indexed {total_chunks} chunks across {len(arxiv_id_list)} papers")
    return total_chunks


def run_full_ingestion(max_papers: int = 200) -> None:
    """Run the fetch flow followed by the parse-and-index flow."""
    fetch_papers_flow(max_papers)
    parse_and_index_flow()
