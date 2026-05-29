"""Rate-limited client for the arXiv API.

Fetches paper metadata for the configured categories, topic terms, and date
range, and downloads source PDFs into the local corpus directory. All network
access is throttled to respect the arXiv API terms of service.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from threading import Lock

import arxiv
import httpx

from meridian.config import Settings, get_settings


@dataclass
class PaperMetadata:
    """Structured metadata for a single arXiv paper."""

    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    published: datetime
    pdf_url: str


class RateLimiter:
    """Thread-safe limiter enforcing a maximum request rate.

    Parameters
    ----------
    max_requests_per_second : float
        Upper bound on the sustained request rate across all callers.
    """

    def __init__(self, max_requests_per_second: float) -> None:
        self._min_interval_seconds = 1.0 / max_requests_per_second
        self._lock = Lock()
        self._last_request_time = 0.0

    def wait(self) -> None:
        """Block until the configured minimum interval has elapsed."""
        with self._lock:
            elapsed = time.monotonic() - self._last_request_time
            sleep_for = self._min_interval_seconds - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_request_time = time.monotonic()


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    """Return the process-wide arXiv rate limiter."""
    settings = get_settings()
    return RateLimiter(settings.arxiv_max_requests_per_second)


def build_search_query(settings: Settings) -> str:
    """Construct the arXiv API query for the configured corpus scope.

    Parameters
    ----------
    settings : Settings
        Application settings supplying categories, topic terms, and the
        publication year range.

    Returns
    -------
    str
        A query combining category, topic, and submitted-date constraints.
    """
    category_clause = " OR ".join(f"cat:{category}" for category in settings.arxiv_categories)
    topic_clause = " OR ".join(f'abs:"{term}"' for term in settings.arxiv_topic_terms)
    date_clause = (
        f"submittedDate:[{settings.arxiv_start_year}01010000 "
        f"TO {settings.arxiv_end_year}12312359]"
    )
    return f"({category_clause}) AND ({topic_clause}) AND {date_clause}"


def fetch_paper_metadata(max_results: int) -> list[PaperMetadata]:
    """Fetch paper metadata for the configured categories and date range.

    Parameters
    ----------
    max_results : int
        Maximum number of papers to return.

    Returns
    -------
    list of PaperMetadata
        Metadata records sorted by submission date descending and filtered to
        the configured publication year range.
    """
    settings = get_settings()
    delay_seconds = 1.0 / settings.arxiv_max_requests_per_second
    client = arxiv.Client(page_size=100, delay_seconds=delay_seconds, num_retries=3)
    search = arxiv.Search(
        query=build_search_query(settings),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    metadata_list: list[PaperMetadata] = []
    for result in client.results(search):
        published_year = result.published.year
        if not (settings.arxiv_start_year <= published_year <= settings.arxiv_end_year):
            continue
        metadata_list.append(
            PaperMetadata(
                arxiv_id=result.get_short_id(),
                title=result.title.strip().replace("\n", " "),
                authors="; ".join(author.name for author in result.authors),
                abstract=result.summary.strip().replace("\n", " "),
                categories=" ".join(result.categories),
                published=result.published.replace(tzinfo=None),
                pdf_url=result.pdf_url,
            )
        )
    return metadata_list


def download_pdf(metadata: PaperMetadata, *, o_overwrite: bool = False) -> str:
    """Download the PDF for a paper into the raw corpus directory.

    Parameters
    ----------
    metadata : PaperMetadata
        Metadata record carrying the PDF URL and arxiv_id.
    o_overwrite : bool, optional
        If False and the file already exists, return the existing path without
        a network request. Defaults to False.

    Returns
    -------
    str
        Local filesystem path to the downloaded PDF.
    """
    settings = get_settings()
    os.makedirs(settings.corpus_raw_dir, exist_ok=True)
    safe_id = metadata.arxiv_id.replace("/", "_")
    dest_path = os.path.join(settings.corpus_raw_dir, f"{safe_id}.pdf")

    if os.path.exists(dest_path) and not o_overwrite:
        return dest_path

    get_rate_limiter().wait()
    with httpx.Client(follow_redirects=True, timeout=60.0) as http_client:
        response = http_client.get(metadata.pdf_url)
        response.raise_for_status()
        pdf_bytes = response.content

    with open(dest_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)
    return dest_path
