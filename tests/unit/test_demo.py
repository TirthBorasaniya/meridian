"""Unit tests for the UI, the release watcher, the image-build local-mode check, and E6."""

import importlib.util
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from meridian.eval.demo_eval import ask_once, measure_rate_limit
from meridian.release.watcher import ReleaseWatcher
from meridian.serving.app import create_app
from tests.unit.test_serving import _runtime

pytestmark = pytest.mark.unit


def test_ui_page_and_assets_are_served():
    with TestClient(create_app(_runtime(profile="demo"))) as test_client:
        page = test_client.get("/")
        assert page.status_code == 200
        for element_id in (
            "question",
            "answer",
            "sources",
            "index-version",
            "paper-count",
            "last-ingest",
        ):
            assert f'id="{element_id}"' in page.text
        script = test_client.get("/static/app.js")
        assert script.status_code == 200
        assert "/v1/ask" in script.text and "external source" in script.text
        assert "Not enough support in the corpus" in script.text
        assert test_client.get("/static/style.css").status_code == 200


def test_watcher_restores_only_newer_releases():
    runtime = _runtime()
    call_list = []

    def fake_restore(work_dir, o_install_manifest, current_snapshot):
        call_list.append(current_snapshot)
        return None if len(call_list) > 1 else {"snapshot": "meridian_v3_20261016"}

    watcher = ReleaseWatcher(runtime, 3600, Path("unused"), restore=fake_restore)
    assert watcher.check_once() == {"snapshot": "meridian_v3_20261016"}
    assert watcher.check_once() is None
    assert len(call_list) == 2


def test_local_mode_check_runs():
    spec = importlib.util.spec_from_file_location("check_local_mode", "space/check_local_mode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.check()
    assert result["named_sparse_with_idf"] and result["rrf_fusion"]


def test_ask_once_reads_the_stream_to_the_final_event():
    body = (
        'event: meta\ndata: {"session_id": "s"}\n\n'
        'event: token\ndata: {"text": "Hi"}\n\n'
        'event: final\ndata: {"status": "answered", "cached": false, "citations": [{}, {}]}\n\n'
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    with httpx.Client(transport=transport) as client:
        result = ask_once(client, "https://demo", "q")
    assert result["status"] == "answered"
    assert result["citations"] == 2
    assert result["first_token_s"] is not None


def test_rate_limit_burst_summary():
    counter = {"n": 0}

    def handler(request):
        counter["n"] += 1
        if counter["n"] > 10:
            return httpx.Response(429, headers={"Retry-After": "42"})
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summary = measure_rate_limit(client, "https://demo", 12)
    assert (summary["allowed"], summary["limited"], summary["first_429_index"]) == (10, 2, 10)
    assert summary["retry_after_s"] == 42
