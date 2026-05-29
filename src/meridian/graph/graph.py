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


def run_query(query: str, thread_id: str = "default") -> dict:
    """Execute the graph for a single query and return the final state.

    Parameters
    ----------
    query : str
        The user question.
    thread_id : str, optional
        Checkpoint thread identifier. Reusing a thread id resumes that
        conversation's persisted state. Defaults to ``"default"``.

    Returns
    -------
    dict
        The final graph state, including ``generation`` and ``source``.
    """
    config_dict = {"configurable": {"thread_id": thread_id}}
    initial_state = {"query": query, "iteration_count": 0}
    return get_graph().invoke(initial_state, config=config_dict)
