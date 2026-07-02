"""Graph node functions.

Each node receives the current :class:`GraphState` and returns a partial state
update. Nodes are wrapped with the Langfuse ``@observe`` decorator so that each
emits its own span with timing and token counts; the LangChain callback
integration is deliberately not used because it does not surface node-level
detail. When Langfuse is unavailable the decorator degrades to a no-op.
"""

from functools import lru_cache

from tavily import TavilyClient

from meridian.config import get_settings
from meridian.graph.chains import (
    get_answer_chain,
    get_generation_chain,
    get_hallucination_chain,
    get_rewrite_chain,
    get_router_chain,
)
from meridian.graph.state import GraphState
from meridian.retrieval.dense import dense_retrieve
from meridian.retrieval.fusion import reciprocal_rank_fusion
from meridian.retrieval.reranker import get_reranker
from meridian.retrieval.sparse import get_sparse_retriever

try:
    from langfuse.decorators import observe
except ImportError:  # pragma: no cover - tracing is optional

    def observe(*args, **kwargs):  # type: ignore[no-redef]
        """No-op stand-in for the Langfuse decorator when it is unavailable."""

        def _decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return _decorator


@lru_cache(maxsize=1)
def get_tavily_client() -> TavilyClient:
    """Return the process-wide cached Tavily client."""
    return TavilyClient(api_key=get_settings().tavily_api_key)


def _active_query(state: GraphState) -> str:
    """Return the rewritten query if present, otherwise the original query."""
    return state.get("rewritten_query") or state["query"]


def _format_documents(doc_list: list[dict]) -> str:
    """Render graded documents into a numbered, citable context block."""
    part_list: list[str] = []
    for index, doc in enumerate(doc_list, start=1):
        payload_dict = doc.get("payload", {})
        title = payload_dict.get("title", "")
        part_list.append(f"[{index}] {title}\n{doc.get('text', '')}")
    return "\n\n".join(part_list)


def _resolve_context(state: GraphState) -> str:
    """Return the context string for generation and grading.

    Uses the Tavily web result when the CRAG fallback fired, otherwise the
    graded corpus documents.
    """
    if state.get("source") == "web" and state.get("web_search_result"):
        return state["web_search_result"]
    return _format_documents(state.get("graded_docs", []))


@observe(name="route_query")
def route_query(state: GraphState) -> dict:
    """Classify the query type using the 8B model."""
    result = get_router_chain().invoke({"query": state["query"]})
    return {"query_type": result.query_type}


@observe(name="retrieve")
def retrieve(state: GraphState) -> dict:
    """Run dense and sparse retrieval and fuse the results with RRF."""
    query = _active_query(state)
    dense_list = dense_retrieve(query)
    sparse_list = get_sparse_retriever().retrieve(query)
    fused_list = reciprocal_rank_fusion([dense_list, sparse_list])
    return {"retrieved_docs": fused_list}


@observe(name="grade_documents")
def grade_documents(state: GraphState) -> dict:
    """Score retrieved documents with the cross-encoder and filter below 0.5."""
    query = _active_query(state)
    graded_list = get_reranker().rerank(query, state.get("retrieved_docs", []))
    return {"graded_docs": graded_list}


@observe(name="web_search")
def web_search(state: GraphState) -> dict:
    """CRAG fallback: retrieve web context via Tavily when no corpus doc passed."""
    query = _active_query(state)
    response = get_tavily_client().search(
        query=query, max_results=5, search_depth="advanced"
    )
    combined_result = "\n\n".join(
        item.get("content", "") for item in response.get("results", [])
    )
    return {"web_search_result": combined_result, "source": "web"}


@observe(name="generate")
def generate(state: GraphState) -> dict:
    """Generate an answer from the graded documents or the web result."""
    query = _active_query(state)
    context = _resolve_context(state)
    conversation_history = state.get("conversation_history") or "(none)"
    generation = get_generation_chain().invoke(
        {"query": query, "context": context, "conversation_history": conversation_history}
    )
    source = "web" if state.get("source") == "web" else "corpus"
    return {"generation": generation, "source": source}


@observe(name="check_hallucination")
def check_hallucination(state: GraphState) -> dict:
    """Grade whether the generation is grounded in the context.

    When the generation is ungrounded ("yes"), the iteration counter is
    incremented so that the generation-retry loop is bounded by
    ``max_iterations``. A grounded generation ("no") leaves the counter
    unchanged.
    """
    context = _resolve_context(state)
    result = get_hallucination_chain().invoke(
        {"context": context, "generation": state["generation"]}
    )
    update_dict: dict = {"hallucination_score": result.binary_score}
    if result.binary_score == "yes":
        update_dict["iteration_count"] = 1
    return update_dict


@observe(name="check_answer")
def check_answer(state: GraphState) -> dict:
    """Grade whether the generation resolves the original query."""
    result = get_answer_chain().invoke(
        {"query": state["query"], "generation": state["generation"]}
    )
    return {"answer_score": result.binary_score}


@observe(name="rewrite_query")
def rewrite_query(state: GraphState) -> dict:
    """Reformulate the query and increment the iteration counter."""
    result = get_rewrite_chain().invoke({"query": state["query"]})
    return {"rewritten_query": result.rewritten_query, "iteration_count": 1}
