"""Section-aware token chunking with deterministic chunk ids and the chunk payload schema.

Sections are split first; within a section, windows of `chunk_tokens` tokens with
`overlap_tokens` overlap are cut using the embedding model's tokenizer offsets, so every chunk
keeps an exact character span into the parsed text and never exceeds the model's maximum
sequence length.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from meridian.common.config import ChunkerConfig
from meridian.ingest.parse import ParsedPaper
from meridian.ingest.records import PaperMeta

# ============= Types =============


class TokenizerLike(Protocol):
    """The subset of a Hugging Face fast tokenizer used by the chunker."""

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass
class Chunk:
    """One retrievable chunk with its span in the parsed text."""

    chunk_id: str
    arxiv_id: str
    version: int
    section_index: int
    chunk_index: int
    section_title: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    token_count: int
    text: str


class ChunkPayload(BaseModel):
    """Qdrant payload of one chunk; the first fourteen fields are the SPEC S3.4 schema."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    arxiv_id: str
    version: int = Field(ge=1)
    title: str
    authors: list[str]
    published: str
    primary_category: str
    section_title: str
    section_index: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    ingest_batch_id: str
    index_version: int = Field(ge=1)
    # span fields keep labeled evaluation spans valid across rebuilds
    chunk_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_count: int = Field(ge=1)
    parser_version: int = Field(ge=1)


# ============= Identifiers =============


def make_chunk_id(arxiv_id: str, version: int, section_index: int, chunk_index: int) -> str:
    """Return `uuid5(NAMESPACE_URL, "{arxiv_id}v{version}:{section_index}:{chunk_index}")`."""
    name = f"{arxiv_id}v{version}:{section_index}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


# ============= Tokenizer =============


def load_tokenizer(model_name: str) -> tuple[TokenizerLike, int]:
    """
    Load the embedding model's fast tokenizer and its usable token budget.

    Parameters
    ----------
    model_name : str
        Hugging Face model id or local path of the dense embedding model.

    Returns
    -------
    tokenizer, max_tokens : tuple
        The tokenizer and the maximum number of content tokens per input, which is the model
        maximum length minus the special tokens the model adds.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        raise ValueError(f"{model_name} has no fast tokenizer; offsets are required")
    max_tokens = int(tokenizer.model_max_length) - int(tokenizer.num_special_tokens_to_add())
    # whole sections are tokenized only for offsets, so the length warning does not apply
    tokenizer.model_max_length = 1 << 30
    return tokenizer, max_tokens


def token_offsets(tokenizer: TokenizerLike, text: str) -> list[tuple[int, int]]:
    """Return the character offsets of each content token of `text`."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(start), int(end)) for start, end in encoding["offset_mapping"]]


# ============= Section grouping =============


def group_sections(
    paper: ParsedPaper, tokenizer: TokenizerLike, min_section_tokens: int
) -> list[tuple[int, str, int, int]]:
    """
    Merge sections shorter than `min_section_tokens` into the next section.

    Parameters
    ----------
    paper : ParsedPaper
        Parsed paper.
    tokenizer : TokenizerLike
        Tokenizer used to measure section length.
    min_section_tokens : int
        Sections below this token count are merged forward; a short final section is merged
        into the previous group.

    Returns
    -------
    group_list : list of tuple
        (section_index, section_title, char_start, char_end) per group. The index and title
        are those of the section that absorbs the short ones.
    """
    group_list: list[tuple[int, str, int, int]] = []
    pending_start: int | None = None
    for section in paper.section_list:
        token_count = len(token_offsets(tokenizer, paper.section_text(section)))
        group_start = section.char_start if pending_start is None else pending_start
        if token_count < min_section_tokens:
            pending_start = group_start
            continue
        group_list.append((section.index, section.title, group_start, section.char_end))
        pending_start = None
    if pending_start is not None and paper.section_list:
        if group_list:
            last_index, last_title, last_start, _ = group_list[-1]
            group_list[-1] = (last_index, last_title, last_start, paper.section_list[-1].char_end)
        else:
            first = paper.section_list[0]
            group_list.append(
                (first.index, first.title, pending_start, paper.section_list[-1].char_end)
            )
    return group_list


# ============= Windowing =============


def window_spans(
    offset_list: list[tuple[int, int]], chunk_tokens: int, overlap_tokens: int
) -> list[tuple[int, int, int]]:
    """
    Cut token windows and return their character spans.

    Parameters
    ----------
    offset_list : list of tuple of (int, int)
        Character offsets of the tokens of one section, in order.
    chunk_tokens : int
        Window size in tokens.
    overlap_tokens : int
        Tokens shared by consecutive windows.

    Returns
    -------
    span_list : list of tuple of (int, int, int)
        (char_start, char_end, token_count) per window, relative to the section text.
    """
    if not offset_list:
        return []
    span_list: list[tuple[int, int, int]] = []
    stride = chunk_tokens - overlap_tokens
    token_start = 0
    while True:
        token_end = min(token_start + chunk_tokens, len(offset_list))
        span_list.append(
            (
                offset_list[token_start][0],
                offset_list[token_end - 1][1],
                token_end - token_start,
            )
        )
        if token_end >= len(offset_list):
            return span_list
        token_start += stride


def chunk_paper(
    paper: ParsedPaper,
    meta: PaperMeta,
    tokenizer: TokenizerLike,
    max_tokens: int,
    chunker_config: ChunkerConfig,
) -> list[Chunk]:
    """
    Split a parsed paper into section-aware token chunks.

    Parameters
    ----------
    paper : ParsedPaper
        Parsed paper.
    meta : PaperMeta
        Metadata supplying the id and version for chunk ids.
    tokenizer : TokenizerLike
        The embedding model's fast tokenizer.
    max_tokens : int
        Maximum content tokens the embedding model accepts.
    chunker_config : ChunkerConfig
        Window and overlap sizes.

    Returns
    -------
    chunk_list : list of Chunk
        Chunks in document order.

    Raises
    ------
    ValueError
        If `chunk_tokens` exceeds `max_tokens`.
    """
    if chunker_config.chunk_tokens > max_tokens:
        raise ValueError(
            f"chunk_tokens={chunker_config.chunk_tokens} exceeds the model limit of {max_tokens}"
        )
    chunk_list: list[Chunk] = []
    for section_index, section_title, group_start, group_end in group_sections(
        paper, tokenizer, chunker_config.min_section_tokens
    ):
        section_text = paper.text[group_start:group_end]
        span_list = window_spans(
            token_offsets(tokenizer, section_text),
            chunker_config.chunk_tokens,
            chunker_config.overlap_tokens,
        )
        for chunk_index, (rel_start, rel_end, token_count) in enumerate(span_list):
            char_start = group_start + rel_start
            char_end = group_start + rel_end
            chunk_list.append(
                Chunk(
                    chunk_id=make_chunk_id(meta.arxiv_id, meta.version, section_index, chunk_index),
                    arxiv_id=meta.arxiv_id,
                    version=meta.version,
                    section_index=section_index,
                    chunk_index=chunk_index,
                    section_title=section_title,
                    char_start=char_start,
                    char_end=char_end,
                    page_start=paper.page_of(char_start),
                    page_end=paper.page_of(max(char_start, char_end - 1)),
                    token_count=token_count,
                    text=paper.text[char_start:char_end],
                )
            )
    return chunk_list


# ============= Payload =============


def build_payload(
    chunk: Chunk,
    meta: PaperMeta,
    ingest_batch_id: str,
    index_version: int,
    parser_version: int,
) -> dict:
    """Return the validated Qdrant payload of one chunk."""
    payload = ChunkPayload(
        chunk_id=chunk.chunk_id,
        arxiv_id=chunk.arxiv_id,
        version=chunk.version,
        title=meta.title,
        authors=list(meta.authors),
        published=meta.published,
        primary_category=meta.primary_category,
        section_title=chunk.section_title,
        section_index=chunk.section_index,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        ingest_batch_id=ingest_batch_id,
        index_version=index_version,
        chunk_index=chunk.chunk_index,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        token_count=chunk.token_count,
        parser_version=parser_version,
    )
    return payload.model_dump()
