"""Scripted LLM and retriever for agent tests."""

import json

from meridian.common.config import get_agent_config, get_models_config
from meridian.common.llm_client import LLMClient, MemoryResponseCache
from meridian.retrieval.hybrid import RetrievalResult, RetrievedChunk
from tests.fakes import ScriptedTransport


def make_chunk(
    index: int, score: float, arxiv_id: str = "2201.11903", text: str = ""
) -> RetrievedChunk:
    """Return a retrieved chunk with a given reranker score."""
    return RetrievedChunk(
        chunk_id=f"00000000-0000-0000-0000-{index:012d}",
        payload={
            "arxiv_id": arxiv_id,
            "version": 6,
            "title": f"Paper {arxiv_id}",
            "section_title": f"{index} Section",
            "section_index": index,
            "page_start": 1,
            "page_end": 2,
            "text": text or f"Fact number {index} is stated here. A second sentence. A third one.",
            "char_start": 0,
            "char_end": 10,
            "index_version": 1,
        },
        score=score,
        rerank_score=score,
    )


class FakeRetriever:
    """Returns queued results; repeats the last one when the queue runs out."""

    def __init__(self, chunk_list_queue: list[list[RetrievedChunk]]) -> None:
        self.queue = list(chunk_list_queue)
        self.call_list: list[dict] = []

    def search(self, query, mode="hybrid_rerank", k=None, paper_filter=None):
        self.call_list.append({"query": query, "paper_filter": paper_filter})
        chunk_list = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return RetrievalResult(
            query=query, mode=mode, chunk_list=list(chunk_list), timing_dict={"total": 0.01}
        )


class FakeWeb:
    """Returns one fixed web result."""

    def __init__(self) -> None:
        self.call_count = 0

    def search(self, query: str, max_results: int) -> list[dict]:
        self.call_count += 1
        return [{"title": "Web page", "url": "https://example.org/a", "content": "Web fact."}]


class Script:
    """Per-prompt replies; each value is a string or a callable of the user message."""

    def __init__(self, **reply_dict) -> None:
        self.reply_dict = {
            "route": json.dumps({"route": "answer", "clarifying_question": ""}),
            "rewrite": json.dumps({"query": "standalone query"}),
            "grade": json.dumps({"relevant": True}),
            "generate": "Fact one holds [c:1]. Fact two holds [c:2].",
            "verify": json.dumps({"supported": True}),
        }
        self.reply_dict.update(reply_dict)
        self.count_dict: dict[str, int] = {}

    def __call__(self, model, message_list, param_dict):
        system = message_list[0]["content"]
        if system.startswith("You route requests"):
            kind = "route"
        elif "standalone search query" in system:
            kind = "rewrite"
        elif system.startswith("You judge whether a passage"):
            kind = "grade"
        elif system.startswith("You check whether a sentence"):
            kind = "verify"
        else:
            kind = "generate"
        self.count_dict[kind] = self.count_dict.get(kind, 0) + 1
        reply = self.reply_dict[kind]
        return reply(message_list[-1]["content"]) if callable(reply) else reply


def make_agent(script: Script, retriever: FakeRetriever, web=None, **config_update):
    """Return an agent with scripted dependencies and an in-memory checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver

    from meridian.agent.build import Agent
    from meridian.agent.deps import AgentDeps

    agent_config = get_agent_config()
    for section, update in config_update.items():
        agent_config = agent_config.model_copy(
            update={section: getattr(agent_config, section).model_copy(update=update)}
        )
    llm = LLMClient(get_models_config(), ScriptedTransport(script), MemoryResponseCache())
    deps = AgentDeps(
        llm=llm,
        retriever=retriever,
        agent_config=agent_config,
        topic_name="LLM reasoning and evaluation",
        topic_description="papers on reasoning",
        final_k=6,
        web=web,
    )
    return Agent(deps, InMemorySaver())
