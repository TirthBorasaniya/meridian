"""StateGraph assembly, checkpointing, and a query convenience wrapper.

Wires nodes and conditional edges per the routing table and compiles the graph
with a SQLite checkpointer so that graph state persists across API restarts.
"""

import os
import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from meridian.config import get_settings
from meridian.graph.edges import (
    decide_after_answer,
    decide_after_grading,
    decide_after_hallucination,
)
from meridian.graph.nodes import (
    check_answer,
    check_hallucination,
    generate,
    grade_documents,
    retrieve,
    rewrite_query,
    route_query,
    web_search,
)
from meridian.graph.state import GraphState
from meridian.memory.session_store import get_session_context, save_turn


def build_graph() -> StateGraph:
    """Construct the uncompiled state graph with all nodes and edges.

    Returns
    -------
    langgraph.graph.StateGraph
        The graph builder, ready to compile.
    """
    builder = StateGraph(GraphState)

    builder.add_node("route_query", route_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("web_search", web_search)
    builder.add_node("generate", generate)
    builder.add_node("check_hallucination", check_hallucination)
    builder.add_node("check_answer", check_answer)
    builder.add_node("rewrite_query", rewrite_query)

    builder.add_edge(START, "route_query")
    builder.add_edge("route_query", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate", "web_search": "web_search"},
    )
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", "check_hallucination")
    builder.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination,
        {"generate": "generate", "check_answer": "check_answer", END: END},
    )
    builder.add_conditional_edges(
        "check_answer",
        decide_after_answer,
        {"rewrite_query": "rewrite_query", END: END},
    )
    builder.add_edge("rewrite_query", "retrieve")

    return builder


def get_checkpointer() -> SqliteSaver:
    """Return a SqliteSaver backed by the configured on-disk database.

    A plain ``sqlite3`` connection is constructed explicitly rather than using
    ``SqliteSaver.from_conn_string`` because the latter is a context manager in
    current LangGraph versions; an explicit connection is stable across
    versions. ``check_same_thread`` is disabled so the saver is usable from the
    threaded server.
    """
    settings = get_settings()
    os.makedirs(os.path.dirname(settings.checkpoint_db_path), exist_ok=True)
    connection = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    return SqliteSaver(connection)


@lru_cache(maxsize=1)
def get_graph():
    """Return the process-wide compiled graph with checkpointing attached."""
    return build_graph().compile(checkpointer=get_checkpointer())


def run_query(query: str, thread_id: str = "default", session_id: str | None = None) -> dict:
    """Execute the graph for a single query and return the final state.

    Parameters
    ----------
    query : str
        The user question.
    thread_id : str, optional
        Checkpoint thread identifier. Reusing a thread id resumes that
        conversation's persisted state within a running process. Defaults to
        ``"default"``.
    session_id : str or None, optional
        Cross-session memory key. Distinct from ``thread_id``: it persists
        conversation turns across process restarts via
        :mod:`meridian.memory.session_store`, whereas ``thread_id`` only
        resumes mid-graph state via ``SqliteSaver``. Defaults to ``thread_id``
        when not given.

    Returns
    -------
    dict
        The final graph state, including ``generation`` and ``source``.
    """
    session_id = session_id or thread_id
    config_dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }
    # Every field derived from a previous turn is cleared explicitly. Reusing a
    # ``thread_id`` resumes that thread's checkpointed channel values, so any
    # field a node does not write on every run would otherwise carry forward
    # into this query. Three carried forward with visible consequences:
    # ``rewritten_query`` (``_active_query`` would retrieve for the previous
    # question), and ``source`` together with ``web_search_result`` (once a
    # thread fell back to the web, ``generate`` re-read its own ``source`` and
    # kept serving the stale web context in place of freshly retrieved
    # documents). ``iteration_count`` resets through the ``add_or_reset``
    # reducer, for which zero means reset rather than add nothing.
    initial_state = {
        "query": query,
        "session_id": session_id,
        "conversation_history": get_session_context(session_id),
        "rewritten_query": "",
        "query_type": "",
        "retrieved_docs": [],
        "graded_docs": [],
        "generation": "",
        "hallucination_score": "",
        "answer_score": "",
        "web_search_result": "",
        "source": "",
        "iteration_count": 0,
    }
    final_state = get_graph().invoke(initial_state, config=config_dict)
    save_turn(session_id, "user", query)
    save_turn(session_id, "assistant", final_state.get("generation", ""))
    return final_state
