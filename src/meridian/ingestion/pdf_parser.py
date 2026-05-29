"""Docling-based PDF parsing.

Converts a downloaded arXiv PDF into structured Markdown and caches the result
to the parsed corpus directory so that re-runs do not re-parse unchanged PDFs.
"""

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from meridian.config import get_settings

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter


@lru_cache(maxsize=1)
def get_converter() -> "DocumentConverter":
    """Return a cached Docling document converter.

    The converter loads layout and OCR models on first construction, so it is
    instantiated once and reused across papers. Docling is imported lazily so
    importing this module stays cheap.
    """
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def parse_pdf(pdf_path: str, arxiv_id: str, *, o_overwrite: bool = False) -> str:
    """Parse a PDF into structured Markdown and cache it on disk.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF.
    arxiv_id : str
        Identifier used to name the cached parsed output.
    o_overwrite : bool, optional
        If False and a cached parse exists, return it without re-parsing.
        Defaults to False.

    Returns
    -------
    str
        The parsed document as Markdown text.

    Raises
    ------
    FileNotFoundError
        If ``pdf_path`` does not exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    settings = get_settings()
    os.makedirs(settings.corpus_parsed_dir, exist_ok=True)
    safe_id = arxiv_id.replace("/", "_")
    parsed_path = os.path.join(settings.corpus_parsed_dir, f"{safe_id}.md")

    if os.path.exists(parsed_path) and not o_overwrite:
        with open(parsed_path, encoding="utf-8") as parsed_file:
            return parsed_file.read()

    result = get_converter().convert(pdf_path)
    text = result.document.export_to_markdown()

    with open(parsed_path, "w", encoding="utf-8") as parsed_file:
        parsed_file.write(text)
    return text


def parsed_path_for(arxiv_id: str) -> str:
    """Return the cached parsed-output path for an arxiv_id."""
    settings = get_settings()
    safe_id = arxiv_id.replace("/", "_")
    return os.path.join(settings.corpus_parsed_dir, f"{safe_id}.md")
