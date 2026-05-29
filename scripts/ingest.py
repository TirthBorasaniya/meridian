"""Entrypoint for the corpus ingestion pipeline.

Runs the Prefect fetch flow followed by the parse-and-index flow. Use
``--max-papers`` to cap the corpus size for a smoke test (the sequencing
guidance recommends 10 papers first).

Examples
--------
Run a smoke ingestion of 10 papers::

    python scripts/ingest.py --max-papers 10

Run the full ingestion at the configured target corpus size::

    python scripts/ingest.py
"""

import argparse

from meridian.config import get_settings
from meridian.ingestion.flows import fetch_papers_flow, parse_and_index_flow


def main() -> None:
    """Parse arguments and run both ingestion flows."""
    parser = argparse.ArgumentParser(description="Run the Meridian ingestion pipeline.")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum papers to fetch. Defaults to the configured target corpus size.",
    )
    args = parser.parse_args()

    settings = get_settings()
    max_papers = args.max_papers if args.max_papers is not None else settings.target_corpus_size

    print(f"Fetching up to {max_papers} papers from arXiv...")
    downloaded_count = fetch_papers_flow(max_papers)
    print(f"Downloaded {downloaded_count} papers. Parsing and indexing...")
    indexed_chunk_count = parse_and_index_flow()
    print(f"Indexed {indexed_chunk_count} chunks into Qdrant.")


if __name__ == "__main__":
    main()
