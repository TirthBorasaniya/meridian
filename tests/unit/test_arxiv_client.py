"""Unit tests for the arXiv throttle, retries, Atom parsing, and query construction."""

import random
from datetime import UTC, datetime

import httpx
import pytest

from meridian.common.config import get_corpus_config
from meridian.ingest.arxiv_client import (
    ArxivClient,
    ArxivRequestError,
    ArxivThrottle,
    PdfUnavailableError,
    backfill_query,
    increment_query,
    normalize_query,
    parse_feed,
    read_seed_ids,
    retry_after_seconds,
)
from meridian.ingest.records import PaperMeta

pytestmark = pytest.mark.unit


class FakeClock:
    """Monotonic clock advanced only by sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleep_list: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_list.append(seconds)
        self.now += seconds


def _client(handler, clock: FakeClock) -> ArxivClient:
    corpus_config = get_corpus_config()
    throttle = ArxivThrottle(
        corpus_config.throttle.min_interval_seconds,
        corpus_config.throttle.jitter_seconds,
        clock=clock,
        sleep=clock.sleep,
        rng=random.Random(0),
    )
    return ArxivClient(
        corpus_config,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        throttle=throttle,
        sleep=clock.sleep,
        rng=random.Random(0),
    )


def test_throttle_spaces_requests_by_at_least_three_seconds():
    clock = FakeClock()
    throttle = ArxivThrottle(3.0, 0.5, clock=clock, sleep=clock.sleep, rng=random.Random(1))
    start_list = []
    for _ in range(4):
        throttle.wait()
        start_list.append(clock.now)
    gap_list = [later - earlier for earlier, later in zip(start_list, start_list[1:], strict=False)]
    assert all(3.0 <= gap <= 3.5 for gap in gap_list)


def test_retry_after_parses_seconds_and_dates():
    assert retry_after_seconds("12") == 12.0
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
    assert retry_after_seconds("Wed, 14 Oct 2026 12:00:30 GMT", now) == pytest.approx(30.0)
    assert retry_after_seconds(None) is None
    assert retry_after_seconds("soon") is None


def test_429_and_503_are_retried_with_retry_after(fixture_dir):
    feed_text = (fixture_dir / "arxiv_feed.xml").read_text()
    status_list = [429, 503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = status_list.pop(0)
        if status == 200:
            return httpx.Response(200, text=feed_text)
        return httpx.Response(status, headers={"Retry-After": "40"})

    clock = FakeClock()
    client = _client(handler, clock)
    meta_list, total = client.search_page("q", 0, 10, "relevance", "descending")
    assert total == 2
    assert len(meta_list) == 2
    assert sum(clock.sleep_list) >= 80.0


def test_retries_exhaust_and_other_4xx_fail_fast():
    clock = FakeClock()
    client = _client(lambda request: httpx.Response(503), clock)
    with pytest.raises(ArxivRequestError, match="after retries"):
        client.request("https://export.arxiv.org/api/query")
    calls = []

    def bad_request(request):
        calls.append(1)
        return httpx.Response(400)

    with pytest.raises(ArxivRequestError, match="400"):
        _client(bad_request, FakeClock()).request("https://export.arxiv.org/api/query")
    assert len(calls) == 1


def test_pdf_404_is_permanent_and_withdrawn_skips_the_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(404)

    client = _client(handler, FakeClock())
    meta = PaperMeta(
        "2401.00001", 1, "t", [], "", "", "cs.CL", pdf_url="https://arxiv.org/pdf/2401.00001v1"
    )
    with pytest.raises(PdfUnavailableError):
        client.download_pdf(meta, tmp_path / "x.pdf")
    assert len(calls) == 1
    withdrawn = PaperMeta("2401.00002", 2, "t", [], "", "", "cs.CL", o_withdrawn=True)
    with pytest.raises(PdfUnavailableError):
        client.download_pdf(withdrawn, tmp_path / "y.pdf")
    assert len(calls) == 1


def test_pdf_download_writes_file_and_hash(tmp_path):
    body = b"%PDF-1.5 fake body"
    client = _client(lambda request: httpx.Response(200, content=body), FakeClock())
    meta = PaperMeta(
        "2401.00001", 1, "t", [], "", "", "cs.CL", pdf_url="https://arxiv.org/pdf/2401.00001v1"
    )
    sha = client.download_pdf(meta, tmp_path / "p.pdf")
    assert (tmp_path / "p.pdf").read_bytes() == body
    assert len(sha) == 64


def test_parse_feed_extracts_versions_and_withdrawals(fixture_dir):
    meta_list, total = parse_feed((fixture_dir / "arxiv_feed.xml").read_text())
    assert total == 2
    first, second = meta_list
    assert (first.arxiv_id, first.version) == ("2201.11903", 6)
    assert first.title == "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    assert first.authors == ["Jason Wei", "Denny Zhou"]
    assert first.primary_category == "cs.CL"
    assert first.categories == ["cs.CL", "cs.AI"]
    assert first.published == "2022-01-28T02:33:07Z"
    assert not first.o_withdrawn
    assert second.o_withdrawn


def test_builder_reproduces_the_recorded_backfill_query():
    query_config = get_corpus_config().arxiv
    assert normalize_query(backfill_query(query_config)) == normalize_query(
        query_config.recorded_backfill_query
    )


def test_increment_query_has_window_and_optional_clause():
    query_config = get_corpus_config().arxiv
    query = increment_query(
        query_config, datetime(2026, 10, 1, 6, 30), datetime(2026, 10, 2, 6, 30)
    )
    assert query.endswith("submittedDate:[202610010630 TO 202610020630]")
    narrowed_config = query_config.model_copy(
        update={
            "increment": query_config.increment.model_copy(
                update={"extra_clause": 'abs:"calibration"'}
            )
        }
    )
    narrowed = increment_query(narrowed_config, datetime(2026, 10, 1), datetime(2026, 10, 2))
    assert ' AND (abs:"calibration") AND submittedDate' in narrowed


def test_read_seed_ids_skips_comments(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# comment\n2201.11903v6\n\n2203.11171v4\n")
    assert read_seed_ids(path) == ["2201.11903v6", "2203.11171v4"]
    path.write_text("2201.11903\n")
    with pytest.raises(ValueError):
        read_seed_ids(path)
