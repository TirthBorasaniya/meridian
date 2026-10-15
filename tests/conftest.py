"""Shared pytest fixtures: repository-relative paths and isolated config caches."""

import os
from pathlib import Path

# tests never read a developer's .env, which may hold real keys
os.environ["MERIDIAN_ENV_FILE"] = os.devnull

import pytest  # noqa: E402

from meridian.common import config as config_module  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _clear_config_caches():
    """Clear cached config accessors so a test that patches configs cannot leak."""
    config_module.clear_config_caches()
    yield
    config_module.clear_config_caches()


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root."""
    return REPO_ROOT


@pytest.fixture
def fixture_dir() -> Path:
    """Return the test fixture directory."""
    return FIXTURE_DIR
