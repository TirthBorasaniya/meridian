"""Unit tests for graph node functions with all external dependencies mocked.

No model is loaded and no network call is made: the retrievers, reranker, LLM
chains, and Tavily client are replaced with lightweight fakes.
"""

import types

from meridian.graph import nodes


class _FakeChain:
    """A chain whose ``invoke`` returns a fixed result."""

    def __init__(self, result) -> None:
        self._result = result

    def invoke(self, _inputs):
        return self._result


def test_route_query(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "get_router_chain",
        lambda: _FakeChain(types.SimpleNamespace(query_type="comparative")),
    )
    assert nodes.route_query({"query": "compare A and B"}) == {"query_type": "comparative"}


def test_retrieve_fuses_dense_and_sparse(monkeypatch):
    monkeypatch.setattr(nodes, "dense_retrieve", lambda query: [{"chunk_id": "a", "text": "t"}])
    fake_sparse = types.SimpleNamespace(retrieve=lambda query: [{"chunk_id": "b", "text": "u"}])
    monkeypatch.setattr(nodes, "get_sparse_retriever", lambda: fake_sparse)
    monkeypatch.setattr(
        nodes, "reciprocal_rank_fusion", lambda lists: [{"chunk_id": "a"}, {"chunk_id": "b"}]
    )
    out = nodes.retrieve({"query": "q"})
    assert [doc["chunk_id"] for doc in out["retrieved_docs"]] == ["a", "b"]


def test_grade_documents_filters(monkeypatch):
    fake_reranker = types.SimpleNamespace(rerank=lambda query, docs: [docs[0]])
    monkeypatch.setattr(nodes, "get_reranker", lambda: fake_reranker)
    state = {
        "query": "q",
        "retrieved_docs": [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}],
    }
    out = nodes.grade_documents(state)
    assert len(out["graded_docs"]) == 1


def test_generate_uses_corpus_source(monkeypatch):
    monkeypatch.setattr(nodes, "get_generation_chain", lambda: _FakeChain("the answer"))
    state = {"query": "q", "graded_docs": [{"text": "ctx", "payload": {"title": "T"}}]}
    out = nodes.generate(state)
    assert out["generation"] == "the answer"
    assert out["source"] == "corpus"


def test_generate_uses_web_source(monkeypatch):
    monkeypatch.setattr(nodes, "get_generation_chain", lambda: _FakeChain("web answer"))
    state = {"query": "q", "source": "web", "web_search_result": "web ctx"}
    out = nodes.generate(state)
    assert out["source"] == "web"


def test_check_hallucination_increments_on_yes(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "get_hallucination_chain",
        lambda: _FakeChain(types.SimpleNamespace(binary_score="yes")),
    )
    out = nodes.check_hallucination({"query": "q", "generation": "g", "graded_docs": []})
    assert out["hallucination_score"] == "yes"
    assert out["iteration_count"] == 1


def test_check_hallucination_no_increment_on_no(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "get_hallucination_chain",
        lambda: _FakeChain(types.SimpleNamespace(binary_score="no")),
    )
    out = nodes.check_hallucination({"query": "q", "generation": "g", "graded_docs": []})
    assert out == {"hallucination_score": "no"}


def test_check_answer(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "get_answer_chain",
        lambda: _FakeChain(types.SimpleNamespace(binary_score="yes")),
    )
    assert nodes.check_answer({"query": "q", "generation": "g"}) == {"answer_score": "yes"}


def test_rewrite_query_increments_iteration(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "get_rewrite_chain",
        lambda: _FakeChain(types.SimpleNamespace(rewritten_query="better q")),
    )
    out = nodes.rewrite_query({"query": "q"})
    assert out["rewritten_query"] == "better q"
    assert out["iteration_count"] == 1


def test_web_search(monkeypatch):
    fake_client = types.SimpleNamespace(
        search=lambda **kwargs: {"results": [{"content": "r1"}, {"content": "r2"}]}
    )
    monkeypatch.setattr(nodes, "get_tavily_client", lambda: fake_client)
    out = nodes.web_search({"query": "q"})
    assert "r1" in out["web_search_result"]
    assert out["source"] == "web"
