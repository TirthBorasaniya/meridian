"""Endpoint tests for the FastAPI layer.

The graph and Qdrant access are mocked, so no model is loaded and no network
call is made. The TestClient is used without its context manager so the
lifespan warm-up does not run.
"""

from fastapi.testclient import TestClient

from meridian.api import routes
from meridian.api.main import app


def test_health_ok(monkeypatch):
    monkeypatch.setattr(routes, "count_points", lambda: 42)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["indexed_points"] == 42


def test_health_degraded_when_qdrant_unavailable(monkeypatch):
    def _raise() -> int:
        raise RuntimeError("collection missing")

    monkeypatch.setattr(routes, "count_points", _raise)
    client = TestClient(app)
    response = client.get("/health")
    assert response.json()["status"] == "degraded"


def test_query_returns_answer_and_documents(monkeypatch):
    fake_state = {
        "generation": "the answer",
        "source": "corpus",
        "query_type": "factual",
        "iteration_count": 1,
        "graded_docs": [
            {
                "chunk_id": "2201.00001_0",
                "rerank_score": 0.91,
                "payload": {"arxiv_id": "2201.00001", "title": "A Paper"},
            }
        ],
    }
    monkeypatch.setattr(
        routes, "run_query", lambda query, thread_id="default", session_id=None: fake_state
    )
    client = TestClient(app)
    response = client.post("/query", json={"query": "what is chain-of-thought?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "the answer"
    assert payload["num_documents"] == 1
    assert payload["documents"][0]["arxiv_id"] == "2201.00001"
    assert payload["iteration_count"] == 1


def test_eval_summary_absent(monkeypatch):
    monkeypatch.setattr(routes, "load_latest_summary", lambda: None)
    client = TestClient(app)
    response = client.get("/eval-summary")
    assert response.json()["available"] is False


def test_eval_summary_present(monkeypatch):
    monkeypatch.setattr(
        routes,
        "load_latest_summary",
        lambda: {"timestamp": "20240101T000000Z", "scores": {"faithfulness": 0.9}},
    )
    client = TestClient(app)
    response = client.get("/eval-summary")
    payload = response.json()
    assert payload["available"] is True
    assert payload["scores"]["faithfulness"] == 0.9
