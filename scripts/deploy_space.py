"""Entrypoint for deploying Meridian to a HuggingFace Space.

Uploads the Gradio app, its trimmed requirements, and the Space card from
``spaces/`` to the Space root, plus the ``meridian`` package (``src/`` and
``pyproject.toml``) so the app's ``from meridian...`` imports resolve after
the Space installs ``requirements.txt`` (which pins ``-e .``).

Requires ``HF_TOKEN`` in the environment with write access to the target
Space. This script does not run automatically; invoke it manually:

    python scripts/deploy_space.py --space-id TirthBorasaniya/meridian
"""

import argparse
import os


def deploy_to_space(space_id: str, space_dir: str, token: str) -> str:
    """Upload the Space app and the meridian package, and return the Space URL.

    Parameters
    ----------
    space_id : str
        Target Space identifier, for example ``"TirthBorasaniya/meridian"``.
    space_dir : str
        Local directory containing ``app.py``, ``requirements.txt``, and
        ``README.md`` for the Space (the repo's ``spaces/`` directory).
    token : str
        HuggingFace access token with write permission on ``space_id``.

    Returns
    -------
    str
        The public URL of the deployed Space.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for filename in ("app.py", "requirements.txt", "README.md"):
        api.upload_file(
            path_or_fileobj=os.path.join(space_dir, filename),
            path_in_repo=filename,
            repo_id=space_id,
            repo_type="space",
        )

    api.upload_file(
        path_or_fileobj=os.path.join(repo_root, "pyproject.toml"),
        path_in_repo="pyproject.toml",
        repo_id=space_id,
        repo_type="space",
    )
    api.upload_folder(
        folder_path=os.path.join(repo_root, "src"),
        path_in_repo="src",
        repo_id=space_id,
        repo_type="space",
    )

    return f"https://huggingface.co/spaces/{space_id}"


def main() -> None:
    """Parse arguments and deploy the Space, or print setup instructions."""
    parser = argparse.ArgumentParser(description="Deploy Meridian to a HuggingFace Space.")
    parser.add_argument("--space-id", default="TirthBorasaniya/meridian")
    parser.add_argument(
        "--space-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spaces"),
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print(
            "HF_TOKEN is not set. Create a token with write access at "
            "https://huggingface.co/settings/tokens, then run:\n"
            "  export HF_TOKEN=hf_...\n"
            "  python scripts/deploy_space.py"
        )
        return

    space_url = deploy_to_space(args.space_id, args.space_dir, token)
    print(f"Deployed to {space_url}")


if __name__ == "__main__":
    main()
