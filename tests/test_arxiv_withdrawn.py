"""Tests for withdrawn arXiv version handling in the ingestion client.

No network call is made: the httpx client is replaced with a fake whose status
code is set per test.
"""

from datetime import datetime

import httpx
import pytest

from meridian.ingestion import arxiv_client
from meridian.ingestion.arxiv_client import (
    PaperMetadata,
    PdfUnavailableError,
    detect_withdrawn,
    download_pdf,
)


def _metadata(**overrides) -> PaperMetadata:
    """Build a PaperMetadata record with test defaults."""
    base = {
        "arxiv_id": "2309.12481v2",
        "title": "HANS, are you clever?",
        "authors": "A Author",
        "abstract": "An abstract.",
        "categories": "cs.CL",
        "published": datetime(2023, 9, 21),
        "pdf_url": "https://arxiv.org/pdf/2309.12481v2",
    }
    base.update(overrides)
    return PaperMetadata(**base)


class _FakeResponse:
    """A response with a fixed status code and body."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.content = b"%PDF-1.4 fake"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=None, response=None)


class _FakeClient:
    """A context-manager http client returning a fixed response."""

    def __init__(self, status_code: int, call_counter: list) -> None:
        self._status_code = status_code
        self._call_counter = call_counter

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url):
        self._call_counter.append(url)
        return _FakeResponse(self._status_code)


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Point the corpus raw directory at a temporary path."""
    settings = arxiv_client.get_settings()
    monkeypatch.setattr(settings, "corpus_raw_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(arxiv_client, "get_rate_limiter", lambda: _NoWait())
    return tmp_path


class _NoWait:
    """Rate limiter stand-in that does not sleep."""

    def wait(self) -> None:
        return None


@pytest.mark.parametrize(
    "comment,expected",
    [
        ("This paper contains erroneous evaluations and we would like to withdraw it", True),
        ("Withdrawn by the authors", True),
        ("", False),
        ("Accepted at ACL 2024", False),
    ],
)
def test_detect_withdrawn(comment, expected):
    """The submission comment is the withdrawal signal."""
    assert detect_withdrawn(comment) is expected


def test_withdrawn_metadata_never_requests_a_pdf(raw_dir, monkeypatch):
    """A withdrawn version must not cost a single HTTP request."""
    call_list: list = []
    monkeypatch.setattr(arxiv_client.httpx, "Client", lambda **kwargs: _FakeClient(200, call_list))
    metadata = _metadata(o_is_withdrawn=True, comment="we would like to withdraw it")

    with pytest.raises(PdfUnavailableError):
        download_pdf(metadata)
    assert call_list == []


def test_404_raises_permanent_error_not_a_retryable_one(raw_dir, monkeypatch):
    """A 404 is permanent, so it must not surface as a retryable HTTPError."""
    call_list: list = []
    monkeypatch.setattr(arxiv_client.httpx, "Client", lambda **kwargs: _FakeClient(404, call_list))

    with pytest.raises(PdfUnavailableError):
        download_pdf(_metadata())
    assert len(call_list) == 1


def test_server_error_remains_retryable(raw_dir, monkeypatch):
    """A 503 is transient and must keep raising a retryable HTTPError."""
    call_list: list = []
    monkeypatch.setattr(arxiv_client.httpx, "Client", lambda **kwargs: _FakeClient(503, call_list))

    with pytest.raises(httpx.HTTPStatusError):
        download_pdf(_metadata())


def test_successful_download_writes_the_pdf(raw_dir, monkeypatch):
    """A 200 response is written to the raw corpus directory."""
    call_list: list = []
    monkeypatch.setattr(arxiv_client.httpx, "Client", lambda **kwargs: _FakeClient(200, call_list))

    path = download_pdf(_metadata())
    assert path.endswith("2309.12481v2.pdf")
    with open(path, "rb") as handle:
        assert handle.read() == b"%PDF-1.4 fake"
