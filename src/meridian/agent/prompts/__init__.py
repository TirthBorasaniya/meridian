"""Prompt templates of the agent, stored as text files next to this module."""

from functools import cache
from importlib import resources


@cache
def load_prompt(name: str) -> str:
    """Return the text of `prompts/<name>.txt`."""
    return resources.files(__package__).joinpath(f"{name}.txt").read_text(encoding="utf-8")
