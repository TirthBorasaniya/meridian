"""Unit tests for training-data construction: exclusion list, pair splits, negatives, card."""

import json
from pathlib import Path

import pytest

from meridian.common.config import EmbedderConfig, get_models_config, load_config
from meridian.common.llm_client import LLMClient, MemoryResponseCache
from meridian.embedder.build_pairs import PoolChunk, build_pairs, corpus_pool, split_papers
from meridian.embedder.card import render_card
from meridian.embedder.exclusion import LeakageError, assert_no_leakage
from meridian.embedder.mine_negatives import select_negative
from meridian.embedder.train import recall_key
from meridian.eval.datasets import evaluation_paper_ids, read_jsonl
from tests.fakes import ScriptedTransport

pytestmark = pytest.mark.unit


def _config(**update) -> EmbedderConfig:
    return load_config(EmbedderConfig, "configs/embedder.yaml").model_copy(update=update)


def _pool(n_papers: int, chunks_per_paper: int) -> list[PoolChunk]:
    return [
        PoolChunk(
            chunk_id=f"{paper}-{index}",
            arxiv_id=f"p{paper}",
            section_key=f"p{paper}v1:{index % 2}",
            text=f"text of paper {paper} chunk {index}",
            source="corpus",
        )
        for paper in range(n_papers)
        for index in range(chunks_per_paper)
    ]


def test_leakage_assertion_rejects_evaluation_papers():
    assert assert_no_leakage([{"arxiv_id": "ok"}], {"bad"}) == 1
    with pytest.raises(LeakageError):
        assert_no_leakage([{"arxiv_id": "bad"}], {"bad"})
    with pytest.raises(LeakageError):
        assert_no_leakage([{"arxiv_id": "ok", "negatives": [{"arxiv_id": "bad"}]}], {"bad"})


def test_pool_excludes_evaluation_papers_and_short_chunks():
    payload_list = [
        {
            "chunk_id": "a",
            "arxiv_id": "eval",
            "version": 1,
            "section_index": 0,
            "text": "t",
            "token_count": 300,
        },
        {
            "chunk_id": "b",
            "arxiv_id": "train",
            "version": 1,
            "section_index": 0,
            "text": "t",
            "token_count": 300,
        },
        {
            "chunk_id": "c",
            "arxiv_id": "train",
            "version": 1,
            "section_index": 1,
            "text": "t",
            "token_count": 10,
        },
    ]
    pool = corpus_pool(payload_list, {"eval"}, min_tokens=80)
    assert [chunk.chunk_id for chunk in pool] == ["b"]


def test_dev_split_holds_whole_papers_and_pair_counts():
    pool = _pool(n_papers=10, chunks_per_paper=4)
    dev_set = split_papers(pool, n_dev_chunks=8, seed=3)
    assert 2 <= len(dev_set) <= 3

    def responder(model, messages, params):
        return json.dumps({"queries": ["q one", "q two", "q three"]})

    llm = LLMClient(get_models_config(), ScriptedTransport(responder), MemoryResponseCache())
    config = _config(n_pairs=30, n_dev_pairs=6, queries_per_chunk=3)
    train_list, dev_list = build_pairs(pool, llm, config)
    assert (len(train_list), len(dev_list)) == (30, 6)
    assert not {pair["arxiv_id"] for pair in train_list} & {pair["arxiv_id"] for pair in dev_list}
    model, _messages, _params = llm.transport.request_list[0]
    assert model == get_models_config().role("pair_writer").model
    assert "llama" not in model.lower()


def test_negative_from_the_same_section_is_skipped():
    section_by_text = {"pos": "s1", "same": "s1", "other": "s2"}
    arxiv_by_text = {"pos": "a", "same": "a", "other": "b"}
    negative = select_negative("pos", ["pos", "same", "other"], section_by_text, arxiv_by_text)
    assert negative == {"text": "other", "arxiv_id": "b", "section_key": "s2"}
    assert select_negative("pos", ["same"], section_by_text, arxiv_by_text) is None


def test_recall_key_and_card_render():
    assert (
        recall_key({"dev_cosine_ndcg@10": 0.1, "dev_cosine_recall@10": 0.2})
        == "dev_cosine_recall@10"
    )
    report = {
        "dev_metric": "dev_cosine_recall@10",
        "batch_size": 32,
        "hardware": "CPU arm64",
        "wall_clock_s": 120.0,
        "dev_before": {"dev_cosine_recall@10": 0.5},
        "dev_after": {"dev_cosine_recall@10": 0.6},
    }
    card = render_card(
        _config(), report, {"train_pairs": 10000, "pair_writer": "openai/gpt-oss-120b"}, None
    )
    assert "E5 has not been run yet." in card
    assert "base_model: BAAI/bge-small-en-v1.5" in card


def test_generated_training_pairs_contain_no_evaluation_paper():
    train_path = Path("data/embedder/pairs_train.jsonl")
    if not train_path.exists():
        pytest.skip("training pairs are generated in Phase 2 (meridian embedder pairs)")
    excluded_set = evaluation_paper_ids()
    assert_no_leakage(read_jsonl(train_path), excluded_set)
    assert_no_leakage(read_jsonl("data/embedder/pairs_dev.jsonl"), excluded_set)
