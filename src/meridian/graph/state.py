"""GraphState schema for the agentic RAG state machine.

The state accumulates as the graph executes. ``iteration_count`` uses the
:func:`add_or_reset` reducer so that any node returning
``{"iteration_count": 1}`` increments the running total rather than overwriting
it; this bounds the two recovery loops (generation retry and query rewrite)
against ``max_iterations``.
"""

from typing import Annotated, TypedDict


def add_or_reset(current: int, update: int) -> int:
    """Add ``update`` to ``current``, treating an explicit zero as a reset.

    A plain ``operator.add`` reducer cannot express "start this query over".
    Because the checkpointer resumes a thread's channel values, adding zero
    preserved the previous query's count, so a returning caller on a reused
    ``thread_id`` began at or past ``max_iterations`` and every conditional
    edge routed straight to ``END`` without running any recovery path.

    Nodes still increment by returning ``1`` and never need to read the
    current value. Only the per-query entry point passes ``0``, which this
    reducer interprets as a deliberate reset rather than a no-op addition.

    Parameters
    ----------
    current : int
        The count already accumulated on this thread.
    update : int
        The increment returned by a node, or ``0`` to reset.

    Returns
    -------
    int
        ``0`` when ``update`` is zero, otherwise ``current + update``.
    """
    if update == 0:
        return 0
    return current + update


class GraphState(TypedDict, total=False):
    """Mutable state threaded through every node of the graph."""

    query: str  # original user query
    session_id: str  # cross-session memory key, distinct from the checkpoint thread id
    conversation_history: str  # prior turns for this session, recalled at graph start
    rewritten_query: str  # query after the rewrite node
    query_type: str  # "factual" | "comparative" | "methodological"
    retrieved_docs: list[dict]  # post-RRF, pre-cross-encoder candidates
    graded_docs: list[dict]  # cross-encoder filtered documents (score >= 0.5)
    generation: str  # generator output
    hallucination_score: str  # "yes" | "no"
    answer_score: str  # "yes" | "no"
    web_search_result: str  # Tavily result if CRAG fallback triggered
    source: str  # "corpus" | "web"
    iteration_count: Annotated[int, add_or_reset]  # increments by 1; an explicit 0 resets
