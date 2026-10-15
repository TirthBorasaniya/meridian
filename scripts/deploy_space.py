"""Deploy the public demo to a Hugging Face Docker Space.

Uploads the Space files (Dockerfile and README at the Space root), the package, configs, and the
evaluation sets. Secrets are set only when present in the environment and are never printed:
`GROQ_API_KEY` (a spend-capped key) and `SPACE_HF_TOKEN` (read-only access to the index dataset
repo), plus the `HF_INDEX_REPO` variable.

Usage::

    HF_TOKEN=... HF_SPACE_REPO=user/meridian uv run python scripts/deploy_space.py
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_LIST = ["pyproject.toml", "uv.lock", "LICENSE", "src", "configs", "eval", "space"]


def stage(target: Path) -> None:
    """Assemble the Space repository contents in `target`."""
    for name in COPY_LIST:
        source = REPO_ROOT / name
        if source.is_dir():
            shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target / name)
    shutil.copy2(REPO_ROOT / "space" / "Dockerfile", target / "Dockerfile")
    shutil.copy2(REPO_ROOT / "space" / "README.md", target / "README.md")


def main() -> None:
    """Create or update the Space."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default=os.environ.get("HF_SPACE_REPO", ""))
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "")
    if not args.space or not token:
        raise SystemExit("set HF_SPACE_REPO (or --space) and HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    if os.environ.get("HF_INDEX_REPO"):
        api.add_space_variable(args.space, "HF_INDEX_REPO", os.environ["HF_INDEX_REPO"])
    for secret_name, env_name in (("GROQ_API_KEY", "GROQ_API_KEY"), ("HF_TOKEN", "SPACE_HF_TOKEN")):
        if os.environ.get(env_name):
            api.add_space_secret(args.space, secret_name, os.environ[env_name])
    with tempfile.TemporaryDirectory(dir=REPO_ROOT / ".tmp") as staging:
        stage(Path(staging))
        api.upload_folder(folder_path=staging, repo_id=args.space, repo_type="space")
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
