"""Unit tests for the results artifact schema and latency summaries."""

import json
from datetime import UTC, datetime

import pytest

from meridian.common.results import build_result, latency_summary, percentile, write_result

pytestmark = pytest.mark.unit


def test_build_result_has_the_shared_schema(repo_root):
    moment = datetime(2026, 10, 15, 14, 2, 11, tzinfo=UTC)
    result_dict = build_result(
        "e1_retrieval",
        {"x": 1},
        str(repo_root / "configs" / "index.yaml"),
        notes="n",
        now=moment,
    )
    assert set(result_dict) == {
        "schema_version",
        "run_id",
        "git_sha",
        "timestamp_utc",
        "config_path",
        "config_sha256",
        "environment",
        "metrics",
        "notes",
    }
    assert result_dict["run_id"] == "2026-10-15T14-02-11Z_e1_retrieval"
    assert result_dict["timestamp_utc"] == "2026-10-15T14:02:11Z"
    assert len(result_dict["config_sha256"]) == 64
    assert set(result_dict["environment"]) == {"python", "packages", "hardware"}


def test_write_result_writes_json(tmp_path, repo_root):
    output_path = write_result(
        "e0_test", {"a": 2}, str(repo_root / "configs" / "gate.yaml"), results_dir=tmp_path
    )
    loaded_dict = json.loads(output_path.read_text())
    assert loaded_dict["metrics"] == {"a": 2}


def test_percentile_interpolates():
    assert percentile([], 0.5) is None
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)


def test_latency_summary_handles_empty():
    assert latency_summary([])["count"] == 0
    summary_dict = latency_summary([1.0, 3.0])
    assert summary_dict["mean_s"] == pytest.approx(2.0)
