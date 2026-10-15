"""Unit tests for README rendering from results files, and for CLI handler resolution."""

import importlib.util
import json

import pytest

from meridian.cli import COMMAND_LIST, resolve_handler
from meridian.common.results import build_result

pytestmark = pytest.mark.unit


def _renderer():
    spec = importlib.util.spec_from_file_location("render_results", "scripts/render_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(value: float) -> dict:
    latency = {"count": 1, "p50_s": 0.1, "p95_s": 0.2, "mean_s": 0.1}
    return {
        "recall_at_5": value,
        "recall_at_10": value,
        "mrr_at_10": value,
        "ndcg_at_10": value,
        "n_queries": 1,
        "latency": {
            stage: latency for stage in ("embed_dense", "embed_sparse", "qdrant", "rerank", "total")
        },
    }


def test_missing_results_render_as_not_measured():
    module = _renderer()
    text = "<!-- results:e2:start -->\nold\n<!-- results:e2:end -->\n"
    rendered = module.render_readme(text, {})
    assert "Not yet measured. Run `make eval-answers`" in rendered
    assert "old" not in rendered
    headline = module.render_block("headline", {})
    assert headline.count("not yet measured") == 11


def test_e1_renders_from_a_results_file(tmp_path):
    module = _renderer()
    result = build_result(
        "e1_retrieval",
        {
            "dataset": {"split": "full", "n_items": 150, "n_invalid_excluded": 2},
            "index": {"collection": "meridian_v1", "papers": 199, "points": 11000},
            "configurations": {"hybrid_rerank": _summary(0.8125)},
            "prefetch_sensitivity": {"by_limit": {"50": _summary(0.8)}},
            "chunk_size_sensitivity": {
                "by_chunk_tokens": {"400": {**_summary(0.8), "build": {"reused_live": True}}}
            },
        },
        "configs/eval/e1.yaml",
    )
    (tmp_path / "e1_retrieval.json").write_text(json.dumps(result))
    results = module.load_results(tmp_path)
    block = module.render_block("e1", results)
    assert "| hybrid_rerank | 0.8125 |" in block
    assert "2 reviewed invalid and excluded" in block
    assert "0.8125" in module.render_block("headline", results)


def test_committed_readme_is_current():
    module = _renderer()
    text = module.README_PATH.read_text()
    assert module.render_readme(text, module.load_results()) == text


def test_every_cli_handler_resolves():
    for _path, handler, _help, _adder in COMMAND_LIST:
        assert callable(resolve_handler(handler)), handler
