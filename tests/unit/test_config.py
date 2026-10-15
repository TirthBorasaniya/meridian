"""Unit tests for YAML config loading and validation rules."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from meridian.common.config import (
    ChunkerConfig,
    ModelsConfig,
    RetrievalConfig,
    get_corpus_config,
    get_gate_config,
    get_index_config,
    get_models_config,
    load_config,
)

pytestmark = pytest.mark.unit


def test_every_config_loads_with_version_and_hash():
    for config in (
        get_corpus_config(),
        get_index_config(),
        get_models_config(),
        get_gate_config(),
    ):
        assert config.version >= 1
        assert len(config.sha256) == 64
        assert Path(config.source_path).exists()


def test_every_yaml_under_configs_has_a_version(repo_root):
    for yaml_path in sorted((repo_root / "configs").rglob("*.yaml")):
        raw_dict = yaml.safe_load(yaml_path.read_text())
        assert isinstance(raw_dict.get("version"), int), yaml_path


def test_judge_must_differ_from_generator(tmp_path, repo_root):
    raw_dict = yaml.safe_load((repo_root / "configs" / "models.yaml").read_text())
    raw_dict["roles"]["judge"]["model"] = raw_dict["roles"]["generator"]["model"]
    bad_path = tmp_path / "models.yaml"
    bad_path.write_text(yaml.safe_dump(raw_dict))
    with pytest.raises(ValidationError, match="judge"):
        load_config(ModelsConfig, bad_path)


def test_pair_writer_rejects_llama(tmp_path, repo_root):
    raw_dict = yaml.safe_load((repo_root / "configs" / "models.yaml").read_text())
    raw_dict["roles"]["pair_writer"]["model"] = "llama-3.3-70b-versatile"
    bad_path = tmp_path / "models.yaml"
    bad_path.write_text(yaml.safe_dump(raw_dict))
    with pytest.raises(ValidationError, match="Llama"):
        load_config(ModelsConfig, bad_path)


def test_chunker_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError):
        ChunkerConfig(chunk_tokens=100, overlap_tokens=100, min_section_tokens=0)


def test_retrieval_depths_are_ordered():
    with pytest.raises(ValidationError):
        RetrievalConfig(prefetch_limit=50, fusion_limit=30, rerank_depth=30, final_k=31, rrf_k=60)
    with pytest.raises(ValidationError):
        RetrievalConfig(prefetch_limit=50, fusion_limit=20, rerank_depth=30, final_k=6, rrf_k=60)


def test_spec_defaults_are_configured():
    index_config = get_index_config()
    assert index_config.chunker.chunk_tokens == 400
    assert index_config.chunker.overlap_tokens == 60
    assert index_config.retrieval.prefetch_limit == 50
    assert index_config.retrieval.rerank_depth == 30
    assert index_config.retrieval.final_k == 6
    assert index_config.alias == "meridian_live"
    assert get_gate_config().max_regression == 0.02
    assert get_corpus_config().throttle.min_interval_seconds >= 3.0


def test_seed_id_list_has_the_recorded_corpus(repo_root):
    seed_path = repo_root / get_corpus_config().arxiv.backfill.seed_ids_path
    id_list = [
        line.strip()
        for line in seed_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(id_list) == 199
    assert len(set(id_list)) == 199
    assert "2309.12481v2" not in id_list
