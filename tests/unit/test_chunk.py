"""Unit tests for chunk ids, token windows, section merging, and the payload schema."""

import re
import uuid

import pytest
from pydantic import ValidationError

from meridian.common.config import ChunkerConfig, get_index_config
from meridian.ingest.chunk import (
    ChunkPayload,
    build_payload,
    chunk_paper,
    group_sections,
    make_chunk_id,
    window_spans,
)
from meridian.ingest.parse import assemble_paper
from meridian.ingest.records import PaperMeta, split_versioned_id

pytestmark = pytest.mark.unit

WORD_PATTERN = re.compile(r"\S+")


class WhitespaceTokenizer:
    """Fake fast tokenizer: one token per whitespace-separated word."""

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [match.span() for match in WORD_PATTERN.finditer(text)]}


def _meta() -> PaperMeta:
    return PaperMeta(
        arxiv_id="2201.11903",
        version=6,
        title="Chain-of-Thought Prompting",
        authors=["Jason Wei", "Denny Zhou"],
        published="2022-01-28T00:00:00Z",
        updated="2023-01-10T00:00:00Z",
        primary_category="cs.CL",
    )


def _paper(word_count_per_section: list[int]):
    page_list = []
    for index, word_count in enumerate(word_count_per_section):
        words = " ".join(f"w{index}_{position}" for position in range(word_count))
        page_list.append(f"## {index + 1} Section{index}\n\n{words}")
    return assemble_paper(page_list, get_index_config().parser, "test", o_fallback=False)


def test_chunk_id_is_deterministic_uuid5():
    first = make_chunk_id("2201.11903", 6, 3, 1)
    assert first == make_chunk_id("2201.11903", 6, 3, 1)
    assert first == str(uuid.uuid5(uuid.NAMESPACE_URL, "2201.11903v6:3:1"))
    assert first != make_chunk_id("2201.11903", 5, 3, 1)
    assert first != make_chunk_id("2201.11903", 6, 3, 2)


def test_window_spans_overlap_and_cover_everything():
    offset_list = [(index * 2, index * 2 + 1) for index in range(25)]
    span_list = window_spans(offset_list, chunk_tokens=10, overlap_tokens=3)
    assert [token_count for _s, _e, token_count in span_list] == [10, 10, 10, 4]
    assert span_list[0][0] == 0
    assert span_list[-1][1] == offset_list[-1][1]
    # the second window starts at token 7
    assert span_list[1][0] == offset_list[7][0]
    assert window_spans([], 10, 3) == []


def test_chunks_never_exceed_the_window_or_the_model_limit():
    paper = _paper([950, 120])
    config = ChunkerConfig(chunk_tokens=400, overlap_tokens=60, min_section_tokens=10)
    tokenizer = WhitespaceTokenizer()
    chunk_list = chunk_paper(paper, _meta(), tokenizer, max_tokens=510, chunker_config=config)
    for chunk in chunk_list:
        assert chunk.token_count <= 400
        assert len(tokenizer(chunk.text)["offset_mapping"]) <= 400
        assert paper.text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_size_above_model_limit_is_rejected():
    paper = _paper([50])
    config = ChunkerConfig(chunk_tokens=600, overlap_tokens=60, min_section_tokens=0)
    with pytest.raises(ValueError, match="exceeds"):
        chunk_paper(paper, _meta(), WhitespaceTokenizer(), max_tokens=510, chunker_config=config)


def test_short_sections_merge_forward_and_ids_follow_sections():
    paper = _paper([5, 200, 3])
    group_list = group_sections(paper, WhitespaceTokenizer(), min_section_tokens=20)
    assert len(group_list) == 1
    section_index, title, char_start, char_end = group_list[0]
    assert title == "2 Section1"
    assert section_index == 1
    assert char_start == 0
    assert char_end == len(paper.text)


def test_section_chunk_indices_restart_per_section():
    paper = _paper([500, 500])
    config = ChunkerConfig(chunk_tokens=400, overlap_tokens=60, min_section_tokens=10)
    chunk_list = chunk_paper(paper, _meta(), WhitespaceTokenizer(), 510, config)
    key_list = [(chunk.section_index, chunk.chunk_index) for chunk in chunk_list]
    assert key_list == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len({chunk.chunk_id for chunk in chunk_list}) == len(chunk_list)


def test_payload_schema_has_the_spec_fields():
    paper = _paper([120])
    config = ChunkerConfig(chunk_tokens=400, overlap_tokens=60, min_section_tokens=10)
    chunk = chunk_paper(paper, _meta(), WhitespaceTokenizer(), 510, config)[0]
    payload_dict = build_payload(chunk, _meta(), "batch-1", 1, parser_version=1)
    spec_field_set = {
        "chunk_id",
        "arxiv_id",
        "version",
        "title",
        "authors",
        "published",
        "primary_category",
        "section_title",
        "section_index",
        "page_start",
        "page_end",
        "text",
        "ingest_batch_id",
        "index_version",
    }
    assert spec_field_set <= set(payload_dict)
    assert payload_dict["authors"] == ["Jason Wei", "Denny Zhou"]
    with pytest.raises(ValidationError):
        ChunkPayload(**{**payload_dict, "unexpected": 1})


def test_split_versioned_id():
    assert split_versioned_id("2201.11903v6") == ("2201.11903", 6)
    with pytest.raises(ValueError):
        split_versioned_id("2201.11903")


@pytest.mark.slow
def test_real_tokenizer_chunks_fit_the_model():
    from meridian.ingest.chunk import load_tokenizer, token_offsets

    index_config = get_index_config()
    tokenizer, max_tokens = load_tokenizer(index_config.dense.model)
    paper = _paper([1500, 300])
    chunk_list = chunk_paper(paper, _meta(), tokenizer, max_tokens, index_config.chunker)
    assert chunk_list
    for chunk in chunk_list:
        assert len(token_offsets(tokenizer, chunk.text)) <= max_tokens
