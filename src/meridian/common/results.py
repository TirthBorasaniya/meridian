"""Results artifacts in the shared schema: one JSON file per evaluation, written to results/.

Numbers in results/ come only from the scripts that call `write_result`; nothing is edited by
hand.
"""

import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from meridian.common.config import file_sha256

logger = logging.getLogger(__name__)

# ============= Constants =============

SCHEMA_VERSION = 1
RESULTS_DIR = Path("results")
DEFAULT_PACKAGE_LIST = (
    "qdrant-client",
    "fastembed",
    "sentence-transformers",
    "torch",
    "transformers",
    "langgraph",
    "ragas",
    "groq",
    "pymupdf4llm",
)


# ============= Environment capture =============


def git_sha() -> str:
    """Return the short git SHA of HEAD, or 'unknown' outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git sha unavailable", extra={"error": str(exc)})
        return "unknown"
    return completed.stdout.strip()


def hardware_description() -> str:
    """Return a short description of the CPU and, when present, the CUDA device."""
    description = f"{platform.system()} {platform.machine()}, {platform.processor() or 'cpu'}"
    try:
        import torch

        if torch.cuda.is_available():
            description += f", CUDA {torch.cuda.get_device_name(0)}"
        elif torch.backends.mps.is_available():
            description += ", MPS"
    except ImportError:
        pass
    return description


def package_versions(package_list: tuple[str, ...] = DEFAULT_PACKAGE_LIST) -> dict[str, str]:
    """Return installed versions for the given distribution names."""
    version_dict: dict[str, str] = {}
    for package_name in package_list:
        try:
            version_dict[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            version_dict[package_name] = "not installed"
    return version_dict


def environment_record() -> dict:
    """Return the environment block of a results artifact."""
    return {
        "python": platform.python_version(),
        "packages": package_versions(),
        "hardware": hardware_description(),
    }


# ============= Writer =============


def make_run_id(experiment: str, now: datetime | None = None) -> str:
    """Return a run id such as `2026-10-15T14-02-11Z_e1_retrieval`."""
    moment = now or datetime.now(UTC)
    return f"{moment.strftime('%Y-%m-%dT%H-%M-%SZ')}_{experiment}"


def build_result(
    experiment: str,
    metrics: dict,
    config_path: str,
    notes: str = "",
    now: datetime | None = None,
) -> dict:
    """
    Assemble a results artifact without writing it.

    Parameters
    ----------
    experiment : str
        Short experiment name, for example `e1_retrieval`.
    metrics : dict
        Measured values. Contents are experiment-specific.
    config_path : str
        Path of the experiment config, hashed into `config_sha256`.
    notes : str
        Free text describing caveats of the run.
    now : datetime or None
        Timestamp override, used by tests.

    Returns
    -------
    result : dict
        Artifact in the shared schema.
    """
    moment = now or datetime.now(UTC)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": make_run_id(experiment, moment),
        "git_sha": git_sha(),
        "timestamp_utc": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": config_path,
        "config_sha256": file_sha256(config_path),
        "environment": environment_record(),
        "metrics": metrics,
        "notes": notes,
    }


def write_result(
    experiment: str,
    metrics: dict,
    config_path: str,
    notes: str = "",
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Write `results/<experiment>.json` and return its path."""
    result_dict = build_result(experiment, metrics, config_path, notes)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{experiment}.json"
    output_path.write_text(json.dumps(result_dict, indent=2, sort_keys=False) + "\n")
    logger.info(
        "results written", extra={"path": str(output_path), "run_id": result_dict["run_id"]}
    )
    return output_path


def percentile(value_list: list[float], fraction: float) -> float | None:
    """
    Return the linear-interpolated percentile of a list, or None when empty.

    Parameters
    ----------
    value_list : list of float
        Observations, any order.
    fraction : float
        Quantile in [0, 1], for example 0.95.

    Returns
    -------
    value : float or None
        The interpolated quantile.
    """
    if not value_list:
        return None
    ordered_list = sorted(value_list)
    position = (len(ordered_list) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_list) - 1)
    weight = position - lower_index
    return ordered_list[lower_index] * (1 - weight) + ordered_list[upper_index] * weight


def latency_summary(value_list: list[float]) -> dict:
    """Return count, p50, p95, and mean of latencies in seconds."""
    if not value_list:
        return {"count": 0, "p50_s": None, "p95_s": None, "mean_s": None}
    return {
        "count": len(value_list),
        "p50_s": percentile(value_list, 0.5),
        "p95_s": percentile(value_list, 0.95),
        "mean_s": sum(value_list) / len(value_list),
    }
