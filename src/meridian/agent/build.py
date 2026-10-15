"""Graph assembly, the SQLite checkpointer keyed by session id, and the `Agent` facade used by the
API and the evaluations."""

import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from meridian.agent.deps import AgentDeps
from meridian.agent.nodes.generate import (
    insufficient,
    make_finalize,
    make_generate,
    make_verify,
    route_after_verify,
)
from meridian.agent.nodes.retrieve import (
    make_decide,
    make_grade,
    make_retrieve,
    make_rewrite,
    make_web_search,
    route_after_decide,
    route_after_web,
)
from meridian.agent.nodes.route import (
    make_respond_without_search,
    make_route,
    route_after_router,
)
from meridian.agent.state import AgentState, per_turn_reset
from meridian.common.llm_client import UsageMeter, usage_scope

RECURSION_LIMIT = 60


def start_turn(state: AgentState) -> dict:
    """Reset every per-turn field; only messages and the pinned paper filter survive."""
    return per_turn_reset(state["question"])


def build_graph(deps: AgentDeps):
    """Return the uncompiled state graph."""
    builder = StateGraph(AgentState)
    builder.add_node("start_turn", start_turn)
    builder.add_node("route", make_route(deps))
    builder.add_node("respond_without_search", make_respond_without_search(deps))
    builder.add_node("rewrite", make_rewrite(deps))
    builder.add_node("retrieve", make_retrieve(deps))
    builder.add_node("grade", make_grade(deps))
    builder.add_node("decide", make_decide(deps))
    builder.add_node("web_search", make_web_search(deps))
    builder.add_node("insufficient", insufficient)
    builder.add_node("generate", make_generate(deps))
    builder.add_node("verify", make_verify(deps))
    builder.add_node("finalize", make_finalize(deps))

    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "route")
    builder.add_conditional_edges(
        "route", route_after_router, ["rewrite", "respond_without_search"]
    )
    builder.add_edge("respond_without_search", "finalize")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_edge("grade", "decide")
    builder.add_conditional_edges(
        "decide", route_after_decide, ["generate", "rewrite", "web_search", "insufficient"]
    )
    builder.add_conditional_edges("web_search", route_after_web, ["generate", "insufficient"])
    builder.add_edge("insufficient", "finalize")
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges("verify", route_after_verify, ["generate", "finalize"])
    builder.add_edge("finalize", END)
    return builder


def sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    """Return a SQLite checkpointer on its own connection."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


@dataclass
class AnswerResult:
    """Outcome of one turn."""

    session_id: str
    question: str
    answer: str
    status: str
    citations: list[dict]
    route: str
    standalone_query: str
    fallback: str
    retries: int
    regenerations: int
    grade_stats: dict
    verification: dict
    contexts: list[str]
    graded: list[dict]
    timings: dict[str, float]
    latency_s: float
    usage: UsageMeter = field(default_factory=UsageMeter)


class Agent:
    """Compiled graph plus helpers to ask one question per turn."""

    def __init__(self, deps: AgentDeps, checkpointer=None) -> None:
        self.deps = deps
        self.graph = build_graph(deps).compile(
            checkpointer=checkpointer or sqlite_checkpointer(deps.agent_config.checkpoint_path)
        )

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT}

    @staticmethod
    def _input(question: str, pinned_papers: list[str] | None) -> dict:
        state_input: dict = {"question": question}
        if pinned_papers is not None:
            state_input["pinned_papers"] = pinned_papers
        return state_input

    def history_length(self, session_id: str) -> int:
        """Return how many messages the session already holds."""
        snapshot = self.graph.get_state(self._config(session_id))
        return len((snapshot.values or {}).get("messages", [])) if snapshot else 0

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        pinned_papers: list[str] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> AnswerResult:
        """
        Answer one question in a session.

        Parameters
        ----------
        question : str
            The user's message.
        session_id : str or None
            Conversation key; a new one is created when None.
        pinned_papers : list of str or None
            Restrict retrieval to these arXiv ids; persists in the session when given.
        on_event : callable or None
            Receives streamed events: `{"type": "token", "text": ...}` and `{"type": "reset"}`.

        Returns
        -------
        result : AnswerResult
            Answer, citations, diagnostics, and LLM usage of the turn.
        """
        session = session_id or uuid.uuid4().hex
        started = time.perf_counter()
        with usage_scope() as meter:
            final_state: dict = {}
            for mode, chunk in self.graph.stream(
                self._input(question, pinned_papers),
                self._config(session),
                stream_mode=["custom", "values"],
            ):
                if mode == "custom" and on_event is not None:
                    on_event(chunk)
                elif mode == "values":
                    final_state = chunk
        return AnswerResult(
            session_id=session,
            question=question,
            answer=final_state.get("answer", ""),
            status=final_state.get("status", ""),
            citations=final_state.get("citations", []),
            route=final_state.get("route", ""),
            standalone_query=final_state.get("standalone_query", ""),
            fallback=final_state.get("fallback", ""),
            retries=final_state.get("retries", 0),
            regenerations=final_state.get("regenerations", 0),
            grade_stats=final_state.get("grade_stats", {}),
            verification=final_state.get("verification", {}),
            contexts=final_state.get("contexts", []),
            graded=final_state.get("graded", []),
            timings=final_state.get("timings", {}),
            latency_s=time.perf_counter() - started,
            usage=meter,
        )


def build_agent(
    agent_config=None,
    o_web_fallback: bool | None = None,
    verify_sample_rate: float | None = None,
    retriever=None,
    llm=None,
    checkpointer=None,
) -> Agent:
    """
    Build the production agent from configs and settings.

    Parameters
    ----------
    agent_config : AgentConfig or None
        Defaults to configs/agent.yaml.
    o_web_fallback : bool or None
        Override of `decide.o_web_fallback` (the public demo disables it).
    verify_sample_rate : float or None
        Override of `verify.sample_rate` (0.3 in serving).
    retriever, llm, checkpointer : optional
        Injected dependencies; defaults are the live retriever and the Groq client.

    Returns
    -------
    agent : Agent
        Ready agent.
    """
    from meridian.agent.web import TavilyWebSearch
    from meridian.common.config import get_agent_config, get_corpus_config, get_index_config
    from meridian.common.llm_client import build_llm_client
    from meridian.common.settings import get_settings
    from meridian.retrieval.factory import build_retriever

    config = agent_config or get_agent_config()
    if o_web_fallback is not None:
        config = config.model_copy(
            update={"decide": config.decide.model_copy(update={"o_web_fallback": o_web_fallback})}
        )
    if verify_sample_rate is not None:
        config = config.model_copy(
            update={"verify": config.verify.model_copy(update={"sample_rate": verify_sample_rate})}
        )
    corpus_config = get_corpus_config()
    tavily_key = get_settings().tavily_api_key.get_secret_value()
    web = TavilyWebSearch(tavily_key) if config.decide.o_web_fallback and tavily_key else None
    deps = AgentDeps(
        llm=llm or build_llm_client(),
        retriever=retriever or build_retriever(),
        agent_config=config,
        topic_name=corpus_config.topic_name,
        topic_description=corpus_config.topic_description,
        final_k=get_index_config().retrieval.final_k,
        web=web,
    )
    return Agent(deps, checkpointer)
