"""Recursive character chunking with token-aware boundaries.

Splits parsed paper text into overlapping chunks of approximately 512 tokens
with 64 tokens of overlap. Token counts are measured with the embedding
model's own tokenizer so that chunks respect the embedder's context window.
The overlap preserves sentence boundaries that fall at chunk edges and ensures
a concept split across a boundary remains retrievable from either chunk.
"""

from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from meridian.config import get_settings


@lru_cache(maxsize=1)
def _get_splitter() -> TextSplitter:
    """Return a cached token-aware recursive character splitter.

    Annotated with the ``TextSplitter`` base rather than the concrete subclass:
    ``from_huggingface_tokenizer`` is inherited and declared to return the base
    type, so pinning the subclass here fails type checking on the
    langchain-text-splitters versions that resolve a fresh install. Only
    ``split_text`` is called, which the base type provides.
    """
    # Imported lazily so importing this module does not load the transformers
    # tokenizer stack until chunking is actually performed.
    from transformers import AutoTokenizer

    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )


def chunk_text(text: str, arxiv_id: str) -> list[dict]:
    """Split parsed text into chunk records with deterministic chunk IDs.

    Parameters
    ----------
    text : str
        Parsed paper text (Markdown).
    arxiv_id : str
        Identifier of the source paper, used to construct chunk IDs.

    Returns
    -------
    list of dict
        One record per chunk with keys ``chunk_id``, ``arxiv_id``,
        ``chunk_index``, and ``text``. The ``chunk_id`` is
        ``{arxiv_id}_{chunk_index}`` so re-ingestion overwrites rather than
        duplicates.
    """
    splitter = _get_splitter()
    piece_list = splitter.split_text(text)
    chunk_list: list[dict] = []
    for chunk_index, piece in enumerate(piece_list):
        chunk_list.append(
            {
                "chunk_id": f"{arxiv_id}_{chunk_index}",
                "arxiv_id": arxiv_id,
                "chunk_index": chunk_index,
                "text": piece,
            }
        )
    return chunk_list
