"""End-to-end agent test: five fixture questions answered with recorded LLM responses; every
citation must resolve to a chunk that exists in the index.

The responses live in `tests/fixtures/e2e_llm_cache.sqlite`, keyed by the client's SHA-256 cache
key. Record them once with a Groq key: `MERIDIAN_RECORD_E2E=1 make test-e2e`. Without the
recording flag the transport refuses network calls, so the test is deterministic.
"""

import os
import shutil
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient

from meridian.agent.build import Agent
from meridian.agent.deps import AgentDeps
from meridian.common.config import (
    get_agent_config,
    get_corpus_config,
    get_index_config,
    get_models_config,
)
from meridian.common.llm_client import LLMClient, SqliteResponseCache
from meridian.ingest import index as index_module
from meridian.ingest.build import build_collection
from meridian.ingest.manifest import Manifest
from meridian.ingest.pipeline import IngestContext
from meridian.retrieval.hybrid import HybridRetriever
from tests.fakes import FakeDenseEncoder, FakeReranker, FakeSparseEncoder
from tests.helpers import acquire_fixture_papers, fake_indexer_factory, lifecycle_papers

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

CACHE_PATH = Path(__file__).parents[1] / "fixtures" / "e2e_llm_cache.sqlite"
QUESTION_LIST = [
    "What does chain of thought prompting ask a model to write?",
    "How is expected calibration error computed?",
    "How does self consistency choose its final answer?",
    "Which biases affect language model judges?",
    "Does reranking improve support for retrieval augmented answers?",
]


class CacheOnlyTransport:
    """Fails on any request, so a cache miss cannot reach the network."""

    def complete(self, model, message_list, param_dict, on_delta):
        raise AssertionError("LLM cache miss; re-record with MERIDIAN_RECORD_E2E=1")


@pytest.fixture
def agent_and_client(tmp_path):
    corpus_config = get_corpus_config()
    paths = corpus_config.paths.model_copy(
        update={
            "pdf_dir": str(tmp_path / "pdfs"),
            "parsed_dir": str(tmp_path / "parsed"),
            "manifest": str(tmp_path / "manifest.sqlite"),
        }
    )
    corpus_config = corpus_config.model_copy(update={"paths": paths})
    index_config = get_index_config()
    client = QdrantClient(":memory:")
    manifest = Manifest(paths.manifest)
    context = IngestContext(
        corpus_config=corpus_config,
        index_config=index_config,
        gate_config=None,  # type: ignore[arg-type]
        manifest=manifest,
        client=client,
        collection="",
        index_version=1,
        arxiv=None,  # type: ignore[arg-type]
        baseline_path="",
        retrieval_set_path="",
        batch_dir=tmp_path,
        alert_path=tmp_path / "a",
    )
    acquire_fixture_papers(context, lifecycle_papers())
    report = build_collection(
        client, manifest, index_config, corpus_config, indexer_factory=fake_indexer_factory
    )
    index_module.switch_alias(client, index_config.alias, report.collection)

    if os.environ.get("MERIDIAN_RECORD_E2E") == "1":
        from meridian.common.llm_client import GroqTransport
        from meridian.common.settings import get_settings

        transport = GroqTransport(
            get_settings().groq_api_key.get_secret_value(),
            get_models_config().request_timeout_seconds,
        )
        cache_path = CACHE_PATH
    else:
        if not CACHE_PATH.exists():
            pytest.fail(f"{CACHE_PATH} is missing; record it with MERIDIAN_RECORD_E2E=1")
        transport = CacheOnlyTransport()
        cache_path = tmp_path / "cache.sqlite"
        shutil.copy2(CACHE_PATH, cache_path)
    llm = LLMClient(get_models_config(), transport, SqliteResponseCache(cache_path))
    retriever = HybridRetriever(
        client,
        index_config.alias,
        FakeDenseEncoder(),
        FakeSparseEncoder(),
        FakeReranker(),
        index_config.retrieval,
    )
    agent_config = get_agent_config()
    agent_config = agent_config.model_copy(
        update={"decide": agent_config.decide.model_copy(update={"o_web_fallback": False})}
    )
    deps = AgentDeps(
        llm=llm,
        retriever=retriever,
        agent_config=agent_config,
        topic_name=corpus_config.topic_name,
        topic_description=corpus_config.topic_description,
        final_k=index_config.retrieval.final_k,
    )
    return Agent(deps, InMemorySaver()), client


def test_fixture_questions_cite_existing_chunks(agent_and_client):
    agent, client = agent_and_client
    alias = get_index_config().alias
    answered = 0
    for question in QUESTION_LIST:
        result = agent.ask(question)
        assert result.status in {"answered", "insufficient"}
        for citation in result.citations:
            assert citation["kind"] == "corpus"
            point_list = client.retrieve(alias, ids=[citation["chunk_id"]], with_payload=True)
            assert point_list, citation
            assert point_list[0].payload["arxiv_id"] == citation["arxiv_id"]
        answered += result.status == "answered"
    assert answered >= 3
