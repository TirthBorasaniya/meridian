"""arXiv API and PDF access on one throttled connection.

arXiv's terms of use allow one request every 3 seconds on a single connection. Every request,
API or PDF, goes through one `ArxivThrottle` with jitter; 429 and 503 responses are retried with
exponential backoff that honors `Retry-After`. Fetches are never parallelized.
"""

import hashlib
import logging
import random
import re
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from meridian.common.config import ArxivQueryConfig, CorpusConfig, ThrottleConfig
from meridian.ingest.records import PaperMeta, split_versioned_id

logger = logging.getLogger(__name__)

# ============= Constants =============

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
RETRY_STATUS_SET = {429, 503}
WITHDRAWN_PATTERN = re.compile(r"withdraw", re.IGNORECASE)
ABS_ID_PATTERN = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v(?P<version>\d+))?$")
SORT_BY_DICT = {
    "relevance": "relevance",
    "submittedDate": "submittedDate",
    "lastUpdatedDate": "lastUpdatedDate",
}
ID_LIST_PAGE_SIZE = 100


class PdfUnavailableError(Exception):
    """arXiv serves no PDF for this version (withdrawn or removed); never retried."""


class ArxivRequestError(Exception):
    """A request failed permanently or exhausted its retries."""


# ============= Throttle =============


class ArxivThrottle:
    """Enforces a minimum spacing plus jitter between consecutive requests."""

    def __init__(
        self,
        min_interval_s: float,
        jitter_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.jitter_s = jitter_s
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._next_allowed = 0.0

    def wait(self) -> None:
        """Block until the next request may start, then reserve the following slot."""
        now = self._clock()
        if now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._clock()
        self._next_allowed = now + self.min_interval_s + self._rng.uniform(0.0, self.jitter_s)

    def defer(self, delay_s: float) -> None:
        """Push the next allowed request at least `delay_s` seconds into the future."""
        self._next_allowed = max(self._next_allowed, self._clock() + delay_s)


def retry_after_seconds(header_value: str | None, now: datetime | None = None) -> float | None:
    """
    Parse a `Retry-After` header given as seconds or as an HTTP date.

    Parameters
    ----------
    header_value : str or None
        Raw header value.
    now : datetime or None
        Reference time for HTTP dates; defaults to the current UTC time.

    Returns
    -------
    seconds : float or None
        Non-negative delay, or None when absent or unparseable.
    """
    if not header_value:
        return None
    stripped = header_value.strip()
    if stripped.isdigit():
        return float(stripped)
    try:
        target = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    reference = now or datetime.now(UTC)
    return max(0.0, (target - reference).total_seconds())


# ============= Query construction =============


def date_clause(start: datetime, end: datetime) -> str:
    """Return a `submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]` clause."""
    return f"submittedDate:[{start.strftime('%Y%m%d%H%M')} TO {end.strftime('%Y%m%d%H%M')}]"


def topic_query(query_config: ArxivQueryConfig) -> str:
    """Return the category and abstract-term part of the corpus query."""
    category_clause = " OR ".join(f"cat:{category}" for category in query_config.categories)
    term_clause = " OR ".join(f'abs:"{term}"' for term in query_config.abstract_terms)
    return f"({category_clause}) AND ({term_clause})"


def backfill_query(query_config: ArxivQueryConfig) -> str:
    """Return the recorded backfill query: topic within the floor and ceiling dates."""
    start = datetime.combine(query_config.date_floor, datetime.min.time())
    end = datetime.combine(query_config.backfill.date_ceiling, datetime.min.time()).replace(
        hour=23, minute=59
    )
    return f"{topic_query(query_config)} AND {date_clause(start, end)}"


def increment_query(query_config: ArxivQueryConfig, start: datetime, end: datetime) -> str:
    """Return the incremental query: topic, optional narrowing clause, and a date window."""
    if start.date() < query_config.date_floor:
        start = datetime.combine(query_config.date_floor, datetime.min.time())
    narrowing = query_config.increment.extra_clause
    focus = f" AND ({narrowing})" if narrowing else ""
    return f"{topic_query(query_config)}{focus} AND {date_clause(start, end)}"


def normalize_query(query: str) -> str:
    """Collapse whitespace, for comparing a built query with the recorded one."""
    return " ".join(query.split())


# ============= Atom parsing =============


def parse_entry(entry: ElementTree.Element) -> PaperMeta:
    """Convert one Atom `entry` into `PaperMeta`."""

    def text_of(path: str) -> str:
        node = entry.find(path, ATOM_NS)
        return " ".join((node.text or "").split()) if node is not None else ""

    abs_url = text_of("atom:id")
    match = ABS_ID_PATTERN.search(abs_url)
    if match is None or match.group("version") is None:
        raise ValueError(f"unexpected arXiv entry id {abs_url!r}")
    arxiv_id = match.group("id")
    version = int(match.group("version"))
    pdf_url = ""
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
    primary_node = entry.find("arxiv:primary_category", ATOM_NS)
    category_list = [
        node.get("term", "") for node in entry.findall("atom:category", ATOM_NS) if node.get("term")
    ]
    comment = text_of("arxiv:comment")
    return PaperMeta(
        arxiv_id=arxiv_id,
        version=version,
        title=text_of("atom:title"),
        authors=[
            " ".join((node.text or "").split())
            for node in entry.findall("atom:author/atom:name", ATOM_NS)
        ],
        published=text_of("atom:published"),
        updated=text_of("atom:updated"),
        primary_category=primary_node.get("term", "") if primary_node is not None else "",
        categories=category_list,
        abstract=text_of("atom:summary"),
        comment=comment,
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}v{version}",
        o_withdrawn=bool(WITHDRAWN_PATTERN.search(comment)),
    )


def parse_feed(xml_text: str) -> tuple[list[PaperMeta], int]:
    """
    Parse an Atom feed into entries and the total result count.

    Raises
    ------
    ArxivRequestError
        If the feed reports an API error entry.
    """
    root = ElementTree.fromstring(xml_text)
    total_node = root.find("opensearch:totalResults", ATOM_NS)
    total = int(total_node.text or 0) if total_node is not None else 0
    meta_list: list[PaperMeta] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        if "/api/errors" in entry_id:
            raise ArxivRequestError(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        meta_list.append(parse_entry(entry))
    return meta_list, total


# ============= Client =============


class ArxivClient:
    """Throttled, retrying client for the arXiv API and PDF host on a single connection."""

    def __init__(
        self,
        corpus_config: CorpusConfig,
        http_client: httpx.Client | None = None,
        throttle: ArxivThrottle | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.query_config = corpus_config.arxiv
        self.throttle_config: ThrottleConfig = corpus_config.throttle
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._throttle = throttle or ArxivThrottle(
            self.throttle_config.min_interval_seconds,
            self.throttle_config.jitter_seconds,
            sleep=sleep,
            rng=self._rng,
        )
        self._http = http_client or httpx.Client(
            timeout=self.throttle_config.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.throttle_config.user_agent},
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )

    def close(self) -> None:
        """Close the underlying connection."""
        self._http.close()

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        base = self.throttle_config.backoff_base_seconds * (2**attempt)
        delay = min(self.throttle_config.backoff_max_seconds, base)
        return max(delay, retry_after or 0.0) + self._rng.uniform(0.0, 1.0)

    def request(self, url: str, params: dict | None = None) -> httpx.Response:
        """
        Issue one throttled GET, retrying 429, 503, and transport errors with backoff.

        Raises
        ------
        PdfUnavailableError
            On 404 for a PDF URL.
        ArxivRequestError
            On another 4xx, or when retries are exhausted.
        """
        for attempt in range(self.throttle_config.max_retries + 1):
            self._throttle.wait()
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                if attempt >= self.throttle_config.max_retries:
                    raise ArxivRequestError(f"{url}: {exc}") from exc
                delay = self._backoff(attempt, None)
                logger.warning("arxiv transport error", extra={"url": url, "delay_s": delay})
                self._throttle.defer(delay)
                continue
            if response.status_code in RETRY_STATUS_SET:
                if attempt >= self.throttle_config.max_retries:
                    raise ArxivRequestError(f"{url}: HTTP {response.status_code} after retries")
                delay = self._backoff(
                    attempt, retry_after_seconds(response.headers.get("Retry-After"))
                )
                logger.warning(
                    "arxiv asked to slow down",
                    extra={"url": url, "status": response.status_code, "delay_s": round(delay, 1)},
                )
                self._throttle.defer(delay)
                continue
            if response.status_code == 404 and "/pdf/" in url:
                raise PdfUnavailableError(f"arXiv serves no PDF at {url}")
            if response.status_code >= 400:
                raise ArxivRequestError(f"{url}: HTTP {response.status_code}")
            return response
        raise ArxivRequestError(f"{url}: retries exhausted")

    # ---------- metadata ----------

    def search_page(
        self, query: str, start: int, max_results: int, sort_by: str, sort_order: str
    ) -> tuple[list[PaperMeta], int]:
        """Return one page of search results and the total count."""
        response = self.request(
            self.query_config.api_url,
            params={
                "search_query": query,
                "start": start,
                "max_results": max_results,
                "sortBy": SORT_BY_DICT[sort_by],
                "sortOrder": sort_order,
            },
        )
        return parse_feed(response.text)

    def iter_search(
        self, query: str, sort_by: str, sort_order: str, limit: int | None = None
    ) -> Iterator[PaperMeta]:
        """Yield search results page by page until `limit` or the end of the result set."""
        page_size = self.query_config.page_size
        start = 0
        yielded = 0
        while True:
            meta_list, total = self.search_page(query, start, page_size, sort_by, sort_order)
            for meta in meta_list:
                yield meta
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            start += len(meta_list)
            if not meta_list or start >= total:
                return

    def fetch_by_ids(self, id_list: list[str]) -> list[PaperMeta]:
        """
        Return metadata for arXiv ids, in pages of 100.

        A versioned id returns that version; a bare id returns the latest version.
        """
        meta_list: list[PaperMeta] = []
        for page_start in range(0, len(id_list), ID_LIST_PAGE_SIZE):
            page_id_list = id_list[page_start : page_start + ID_LIST_PAGE_SIZE]
            response = self.request(
                self.query_config.api_url,
                params={"id_list": ",".join(page_id_list), "max_results": len(page_id_list)},
            )
            page_meta_list, _total = parse_feed(response.text)
            meta_list.extend(page_meta_list)
        return meta_list

    # ---------- PDFs ----------

    def download_pdf(self, meta: PaperMeta, dest_path: Path) -> str:
        """
        Download a paper's PDF to `dest_path` and return its SHA-256.

        Raises
        ------
        PdfUnavailableError
            For withdrawn versions (without a request) and for 404 responses.
        """
        if meta.o_withdrawn:
            raise PdfUnavailableError(f"{meta.versioned_id} is withdrawn: {meta.comment}")
        response = self.request(meta.pdf_url or f"https://arxiv.org/pdf/{meta.versioned_id}")
        content = response.content
        if not content.startswith(b"%PDF"):
            raise ArxivRequestError(f"{meta.versioned_id}: response is not a PDF")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = dest_path.with_suffix(".part")
        temporary_path.write_bytes(content)
        temporary_path.replace(dest_path)
        return hashlib.sha256(content).hexdigest()


# ============= Helpers =============


def read_seed_ids(path: str | Path) -> list[str]:
    """Read versioned ids from a seed list, ignoring comments and blank lines."""
    id_list: list[str] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            split_versioned_id(stripped)
            id_list.append(stripped)
    return id_list


def to_datetime(value: str | date | datetime) -> datetime:
    """Parse an ISO date or datetime into a naive UTC datetime."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime.combine(value, datetime.min.time())
    else:
        text = value.strip().replace("Z", "+00:00")
        moment = datetime.fromisoformat(text) if "T" in text else datetime.fromisoformat(text)
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    return moment
