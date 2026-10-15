"""Dependencies injected into the agent's nodes, so tests can substitute every external call."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from meridian.agent.web import WebSearch
from meridian.common.config import AgentConfig
from meridian.common.llm_client import LLMClient
from meridian.retrieval.hybrid import RetrievalResult


class Retriever(Protocol):
    """The retrieval interface the agent needs."""

    def search(
        self,
        query: str,
        mode: str = ...,
        k: int | None = ...,
        paper_filter: list[str] | None = ...,
    ) -> RetrievalResult: ...


@dataclass
class AgentDeps:
    """Everything the nodes call."""

    llm: LLMClient
    retriever: Retriever
    agent_config: AgentConfig
    topic_name: str
    topic_description: str
    final_k: int
    web: WebSearch | None = None


def timed(state: Mapping[str, Any], name: str, started: float) -> dict[str, float]:
    """Return the state's timings with this node's elapsed seconds added."""
    timing_dict = dict(state.get("timings") or {})
    timing_dict[name] = timing_dict.get(name, 0.0) + (time.perf_counter() - started)
    return timing_dict


def stream_writer() -> Callable[[Any], None]:
    """Return LangGraph's custom stream writer, or a no-op outside a streaming run."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (RuntimeError, LookupError, ImportError):
        return lambda _chunk: None
