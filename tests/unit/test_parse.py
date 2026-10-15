"""Unit tests for PDF cleaning, references removal, and section splitting."""

import pymupdf
import pytest

from meridian.common.config import get_index_config
from meridian.ingest import parse as parse_module
from meridian.ingest.parse import (
    METHOD_PYPDF,
    assemble_paper,
    clean_page,
    find_references_offset,
    find_repeated_edge_lines,
    is_number_only,
    load_parsed,
    parse_pdf,
    save_parsed,
    split_sections,
)

pytestmark = pytest.mark.unit

BODY_SENTENCE = (
    "Chain-of-thought prompting asks the model to write intermediate reasoning steps before "
    "the final answer, which improves accuracy on arithmetic word problems. "
)


def _page(number: int, body: str) -> str:
    return f"Preprint. Under review.\n\n{body}\n\n{number}\n"


def test_number_only_lines():
    assert is_number_only("12")
    assert is_number_only("**3**")
    assert is_number_only(" 4 ")
    assert not is_number_only("3 Results")
    assert not is_number_only("| 1 | 2 |")
    assert not is_number_only("")


def test_repeated_edge_lines_found_on_most_pages():
    body_list = ["alpha results", "beta method", "gamma setup", "delta analysis", "epsilon"]
    page_list = [_page(index, body) for index, body in enumerate(body_list, start=1)]
    repeated_set = find_repeated_edge_lines(page_list, edge_lines=3, page_fraction=0.5)
    assert repeated_set == {"preprint. under review."}


def test_headings_are_never_removed_as_running_headers():
    page_list = [f"## {index} Section\n\nText {word}" for index, word in enumerate("abcd")]
    assert find_repeated_edge_lines(page_list, edge_lines=3, page_fraction=0.5) == set()


def test_clean_page_drops_header_footer_figure_text_and_numbers():
    page_text = (
        "Preprint. Under review.\n\n## 1 Introduction\n\nReal text.\n\n"
        "<!-- Start of picture text -->axis label<br><!-- End of picture text -->\n\n7\n"
    )
    cleaned = clean_page(page_text, {"preprint. under review."}, edge_lines=3)
    assert "Preprint" not in cleaned
    assert "axis label" not in cleaned
    assert "\n7" not in cleaned
    assert "## 1 Introduction" in cleaned
    assert "Real text." in cleaned


def test_references_heading_cuts_everything_after():
    text = "## 1 Introduction\n\nBody.\n\n## **References**\n\nSmith 2020.\n\n## A Appendix\n\nX"
    offset = find_references_offset(text, ["references", "bibliography"], o_fallback=False)
    assert text[offset:].startswith("## **References**")


def test_references_numbered_and_plain_line_variants():
    numbered = "## 1 Intro\n\nBody.\n\n## 7 REFERENCES\n\nA. B."
    assert find_references_offset(numbered, ["references"], o_fallback=False) > 0
    plain = "Intro text.\n\nReferences\n\nA. B."
    assert plain[find_references_offset(plain, ["references"], o_fallback=True) :].startswith(
        "References"
    )
    assert find_references_offset("no refs here", ["references"], o_fallback=False) == -1


def test_split_sections_with_front_matter():
    text = "Title block\n\n## Abstract\n\nA.\n\n## 1 Introduction\n\nB."
    start_list = split_sections(text, o_fallback=False)
    assert [title for _level, title, _start in start_list] == [
        "Front matter",
        "Abstract",
        "1 Introduction",
    ]
    assert text[start_list[1][2] :].startswith("## Abstract")


def test_split_sections_fallback_numbered_headings():
    text = "Abstract\nWe study X.\n1 Introduction\nBody.\n2.1 Setup\nMore."
    title_list = [title for _level, title, _start in split_sections(text, o_fallback=True)]
    assert title_list == ["Abstract", "1 Introduction", "2.1 Setup"]


def test_assemble_paper_offsets_and_pages():
    page_list = [
        _page(1, "## Abstract\n\n" + BODY_SENTENCE),
        _page(2, "## 1 Introduction\n\n" + BODY_SENTENCE),
        _page(3, "## 2 Method\n\n" + BODY_SENTENCE + "\n\n## References\n\nDoe 2021."),
    ]
    paper = assemble_paper(page_list, get_index_config().parser, "test", o_fallback=False)
    assert paper.o_references_removed
    assert "Doe 2021" not in paper.text
    assert "Preprint" not in paper.text
    title_list = [section.title for section in paper.section_list]
    assert title_list == ["Abstract", "1 Introduction", "2 Method"]
    method_section = paper.section_list[2]
    assert paper.section_text(method_section).startswith("## 2 Method")
    assert method_section.page_start == 3
    assert paper.section_list[0].page_start == 1


def _write_pdf(path, page_text_list):
    document = pymupdf.open()
    for page_number, (heading, body) in enumerate(page_text_list, start=1):
        page = document.new_page()
        page.insert_text((72, 40), "Preprint. Under review.", fontsize=8)
        page.insert_text((72, 90), heading, fontsize=18)
        y = 130
        for line_start in range(0, len(body), 90):
            page.insert_text((72, y), body[line_start : line_start + 90], fontsize=10)
            y += 14
        page.insert_text((300, 800), str(page_number), fontsize=8)
    document.save(str(path))
    document.close()


def test_parse_generated_pdf_end_to_end(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    body = BODY_SENTENCE * 6
    _write_pdf(
        pdf_path,
        [
            ("1 Introduction", body),
            ("2 Method", body),
            ("3 Results", body),
            ("References", "Smith, J. A paper about things. 2020. " * 4),
        ],
    )
    paper = parse_pdf(pdf_path, get_index_config().parser)
    assert paper.page_count == 4
    assert paper.o_references_removed
    assert "Smith, J." not in paper.text
    assert "Preprint. Under review." not in paper.text
    assert paper.char_count > 500
    assert paper.section_count >= 3


def test_fallback_to_pypdf_is_flagged(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    _write_pdf(pdf_path, [("1 Introduction", BODY_SENTENCE * 8)] * 2)

    def _fail(_path):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(parse_module, "extract_pages_pymupdf4llm", _fail)
    paper = parse_pdf(pdf_path, get_index_config().parser)
    assert paper.parse_method == METHOD_PYPDF
    assert paper.o_fallback


def test_save_and_load_roundtrip(tmp_path):
    page_list = [_page(1, "## Abstract\n\n" + BODY_SENTENCE), _page(2, BODY_SENTENCE)]
    paper = assemble_paper(page_list, get_index_config().parser, "test", o_fallback=False)
    save_parsed(paper, tmp_path, "2201.00001v2")
    loaded = load_parsed(tmp_path, "2201.00001v2")
    assert loaded is not None
    assert loaded.text == paper.text
    assert loaded.section_list == paper.section_list
    assert loaded.stats() == paper.stats()
