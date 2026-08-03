"""Tests that a reused thread_id starts each query from a clean counter.

The graph is compiled against an in-memory checkpointer and every node's
external dependency is replaced with a fake, so no model is loaded and no
network call is made. The first query is driven down the generation-retry loop
until it saturates ``max_iterations``; the second query on the same thread must
then begin at zero rather than inheriting the saturated count.
"""

import sqlite3
import types

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from meridian.graph import graph as graph_module
from meridian.graph import nodes
from meridian.graph.state import add_or_reset


class _FakeChain:
    """A chain whose ``invoke`` returns whatever the holder currently holds."""

    def __init__(self, holder, key) -> None:
        self._holder = holder
        self._key = key

    def invoke(self, _inputs):
        return self._holder[self._key]


@pytest.fixture
def wired_graph(monkeypatch):
    """Compile the real graph with fake nodes and an in-memory checkpointer."""
    holder = {
        "hallucination": types.SimpleNamespace(binary_score="yes"),
        "answer": types.SimpleNamespace(binary_score="yes"),
    }

    monkeypatch.setattr(
        nodes,
        "get_router_chain",
        lambda: _FakeChain({"r": types.SimpleNamespace(query_type="factual")}, "r"),
    )
    monkeypatch.setattr(nodes, "dense_retrieve", lambda query: [{"chunk_id": "a", "text": "t"}])
    monkeypatch.setattr(
        nodes,
        "get_sparse_retriever",
        lambda: types.SimpleNamespace(retrieve=lambda query: [{"chunk_id": "a", "text": "t"}]),
    )
    monkeypatch.setattr(
        nodes, "reciprocal_rank_fusion", lambda lists: [{"chunk_id": "a", "text": "t"}]
    )
    # A non-empty graded list keeps routing on the corpus path, so the Tavily
    # fallback is never reached.
    monkeypatch.setattr(
        nodes,
        "get_reranker",
        lambda: types.SimpleNamespace(
            rerank=lambda query, docs: [{"text": "ctx", "payload": {"title": "T"}}]
        ),
    )
    monkeypatch.setattr(nodes, "get_generation_chain", lambda: _FakeChain({"g": "an answer"}, "g"))
    monkeypatch.setattr(
        nodes, "get_hallucination_chain", lambda: _FakeChain(holder, "hallucination")
    )
    monkeypatch.setattr(nodes, "get_answer_chain", lambda: _FakeChain(holder, "answer"))
    monkeypatch.setattr(
        nodes,
        "get_rewrite_chain",
        lambda: _FakeChain({"w": types.SimpleNamespace(rewritten_query="rewritten")}, "w"),
    )

    # Session memory is a separate store; keep it out of the graph assertions.
    monkeypatch.setattr(graph_module, "get_session_context", lambda session_id: "")
    monkeypatch.setattr(graph_module, "save_turn", lambda *args, **kwargs: None)

    checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda: checkpointer)
    graph_module.get_graph.cache_clear()
    yield holder
    graph_module.get_graph.cache_clear()


def test_second_query_on_same_thread_resets_iteration_count(wired_graph):
    """A reused thread_id must not inherit the previous query's counter."""
    holder = wired_graph
    max_iterations = graph_module.get_settings().max_iterations

    # First query: the hallucination grader always reports "yes", so the
    # generation-retry loop runs until the cap is reached.
    holder["hallucination"] = types.SimpleNamespace(binary_score="yes")
    first_state = graph_module.run_query("first question", thread_id="shared")
    assert first_state["iteration_count"] == max_iterations

    # Second query on the same thread: a grounded, on-target answer needs no
    # recovery, so the counter must read zero. Under an ``operator.add``
    # reducer it would still read ``max_iterations``.
    holder["hallucination"] = types.SimpleNamespace(binary_score="no")
    holder["answer"] = types.SimpleNamespace(binary_score="yes")
    second_state = graph_module.run_query("second question", thread_id="shared")
    assert second_state["iteration_count"] == 0


def test_recovery_still_runs_on_a_reused_thread(wired_graph):
    """The cap must not be pre-saturated, so recovery is reachable again."""
    holder = wired_graph
    max_iterations = graph_module.get_settings().max_iterations

    holder["hallucination"] = types.SimpleNamespace(binary_score="yes")
    graph_module.run_query("first question", thread_id="shared")

    # Same failing grader on the same thread: the retry loop must run its full
    # budget again rather than routing straight to END on the first check.
    second_state = graph_module.run_query("second question", thread_id="shared")
    assert second_state["iteration_count"] == max_iterations


def test_stale_derived_fields_are_cleared(wired_graph):
    """A previous turn's rewritten query and web context must not carry over."""
    holder = wired_graph

    # Drive the first query into the rewrite loop so ``rewritten_query`` is set.
    holder["hallucination"] = types.SimpleNamespace(binary_score="no")
    holder["answer"] = types.SimpleNamespace(binary_score="no")
    first_state = graph_module.run_query("first question", thread_id="shared")
    assert first_state["rewritten_query"] == "rewritten"

    holder["answer"] = types.SimpleNamespace(binary_score="yes")
    second_state = graph_module.run_query("second question", thread_id="shared")
    assert second_state["rewritten_query"] == ""
    assert second_state["source"] == "corpus"
    assert second_state["web_search_result"] == ""


def test_add_or_reset_semantics():
    """The reducer increments on non-zero and resets on an explicit zero."""
    assert add_or_reset(0, 1) == 1
    assert add_or_reset(2, 1) == 3
    assert add_or_reset(3, 0) == 0
    assert add_or_reset(0, 0) == 0
