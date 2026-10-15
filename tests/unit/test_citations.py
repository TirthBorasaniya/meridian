"""Unit tests for the citation parser, alias resolution, validator, and excerpts."""

import pytest

from meridian.agent.citations import (
    cited_chunk_ids,
    excerpt,
    extract_citations,
    invalid_citations,
    normalize_markers,
    resolve_aliases,
    split_sentences,
    strip_citations,
)

pytestmark = pytest.mark.unit


def test_markers_after_the_period_move_into_the_sentence():
    text = "Self-consistency votes over paths. [c:1] It beats greedy decoding [c:2, c:3]."
    normalized = normalize_markers(text)
    assert (
        normalized == "Self-consistency votes over paths[c:1]. It beats greedy decoding [c:2][c:3]."
    )
    sentence_list = split_sentences(normalized)
    assert len(sentence_list) == 2
    assert extract_citations(sentence_list[1]) == [("c", "2"), ("c", "3")]


def test_sentence_split_protects_abbreviations_and_decimals():
    text = (
        "Models such as GPT-3, e.g. the 175B one, score 3.5 points higher [c:1]. "
        "Wei et al. agree [c:2]."
    )
    sentence_list = split_sentences(text)
    assert len(sentence_list) == 2
    assert sentence_list[1].startswith("Wei et al. agree")


def test_resolve_aliases_maps_ids_and_drops_out_of_context_markers():
    alias_dict = {"1": "chunk-a", "2": "chunk-b"}
    resolved, invalid_list = resolve_aliases(
        "A holds [c:1]. B holds [c:9]. C holds [w:1]. D holds [w:4].", alias_dict, web_count=1
    )
    assert resolved == "A holds [c:chunk-a]. B holds . C holds [w:1]. D holds ."
    assert invalid_list == ["[c:9]", "[w:4]"]
    assert cited_chunk_ids(resolved) == ["chunk-a"]


def test_validator_checks_ids_against_the_context():
    text = "X [c:chunk-a][c:chunk-z]. Y [w:2]."
    assert invalid_citations(text, {"chunk-a"}, web_count=1) == ["[c:chunk-z]", "[w:2]"]
    assert invalid_citations("Z [c:chunk-a].", {"chunk-a"}, 0) == []


def test_strip_citations_and_excerpt():
    assert strip_citations("Fact holds [c:x] here [w:1].") == "Fact holds here."
    text = "## 3 Results\nFirst sentence here. Second sentence here. Third sentence here."
    assert excerpt(text, 2) == "3 Results First sentence here. Second sentence here."
    assert len(excerpt("word " * 500, 2, max_chars=50)) <= 50
