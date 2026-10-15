"""PDF parsing into cleaned, section-split Markdown with character offsets.

`pymupdf4llm` keeps headings; `pypdf` plain text is the fallback, flagged in the manifest. The
references section and everything after it, repeated page headers and footers, number-only
lines, and figure-internal text are removed. Section offsets index into the cleaned text, which
is the coordinate system for labeled evaluation spans.
"""

import bisect
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from meridian.common.config import ParserConfig

logger = logging.getLogger(__name__)

# ============= Constants =============

METHOD_PYMUPDF4LLM = "pymupdf4llm"
METHOD_PYPDF = "pypdf"

MIN_PARSED_CHARS = 500
PICTURE_TEXT_PATTERN = re.compile(
    r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.DOTALL
)
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
OMITTED_PICTURE_PATTERN = re.compile(r"^\*{0,2}==> picture .* <==\*{0,2}\s*$")
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBER_ONLY_PATTERN = re.compile(r"^[\s\-–—]*\d+(?:[.,:/]\d+)*[\s\-–—]*$")
INLINE_MARKUP_PATTERN = re.compile(r"(\*\*|__|`|</?u>|</?b>|</?i>|<br>|\*|_(?=\W)|(?<=\W)_)")
HEADING_NUMBER_PREFIX = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*\.?|[IVXLC]+\.|[A-Z]\.?)\s+")
FALLBACK_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+){0,2}\.?)\s+[A-Z][A-Za-z0-9 ,:;()/&'\-–]{2,80}$"
)
FALLBACK_NAMED_HEADINGS = {"abstract", "introduction", "conclusion", "conclusions"}
MULTI_BLANK_PATTERN = re.compile(r"\n{3,}")
DIGIT_PATTERN = re.compile(r"\d+")


# ============= Data structures =============


@dataclass
class Section:
    """One section of a parsed paper; offsets index the cleaned full text."""

    index: int
    title: str
    level: int
    char_start: int
    char_end: int
    page_start: int
    page_end: int


@dataclass
class ParsedPaper:
    """Cleaned paper text, its sections, and parse statistics."""

    text: str
    section_list: list[Section]
    page_count: int
    parse_method: str
    o_fallback: bool
    parser_version: int
    o_references_removed: bool
    page_offset_list: list[int] = field(default_factory=list)

    @property
    def section_count(self) -> int:
        """Number of sections."""
        return len(self.section_list)

    @property
    def char_count(self) -> int:
        """Number of characters in the cleaned text."""
        return len(self.text)

    def section_text(self, section: Section) -> str:
        """Return the text of one section."""
        return self.text[section.char_start : section.char_end]

    def page_of(self, char_offset: int) -> int:
        """Return the 1-based page containing a character offset."""
        return max(1, bisect.bisect_right(self.page_offset_list, char_offset))

    def stats(self) -> dict:
        """Return the per-paper statistics recorded in the manifest."""
        return {
            "page_count": self.page_count,
            "section_count": self.section_count,
            "char_count": self.char_count,
            "parse_method": self.parse_method,
            "o_fallback": self.o_fallback,
            "parser_version": self.parser_version,
            "o_references_removed": self.o_references_removed,
        }


class ParseError(Exception):
    """Raised when neither parser yields usable text."""


# ============= Raw extraction =============


def extract_pages_pymupdf4llm(pdf_path: str | Path) -> list[str]:
    """Return per-page Markdown from `pymupdf4llm.to_markdown`."""
    import pymupdf4llm

    page_dict_list = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    return [page_dict.get("text", "") for page_dict in page_dict_list]


def extract_pages_pypdf(pdf_path: str | Path) -> list[str]:
    """Return per-page plain text from `pypdf`."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


# ============= Cleaning =============


def strip_markup(text: str) -> str:
    """Remove inline Markdown and HTML emphasis from a short string."""
    return " ".join(INLINE_MARKUP_PATTERN.sub("", text).split())


def normalize_edge_line(line: str) -> str:
    """Normalize a line for repeated header and footer detection."""
    return DIGIT_PATTERN.sub("#", strip_markup(line).lower())


def is_number_only(line: str) -> bool:
    """Return whether a line contains only a number, such as a page number."""
    stripped = strip_markup(line)
    return bool(stripped) and bool(NUMBER_ONLY_PATTERN.match(stripped))


def find_repeated_edge_lines(
    page_list: list[str], edge_lines: int, page_fraction: float
) -> set[str]:
    """
    Return normalized lines that appear at the top or bottom of most pages.

    Parameters
    ----------
    page_list : list of str
        Per-page text.
    edge_lines : int
        Number of non-empty lines inspected at each page edge.
    page_fraction : float
        Minimum share of pages on which a line must recur.

    Returns
    -------
    repeated_set : set of str
        Normalized lines to remove from page edges.
    """
    if len(page_list) < 3:
        return set()
    count_dict: Counter[str] = Counter()
    for page_text in page_list:
        line_list = [line for line in page_text.splitlines() if line.strip()]
        # headings are never treated as running headers, even when they differ only in digits
        edge_set = {
            normalize_edge_line(line)
            for line in line_list[:edge_lines] + line_list[-edge_lines:]
            if not is_number_only(line) and not MARKDOWN_HEADING_PATTERN.match(line.strip())
        }
        count_dict.update(edge_set - {""})
    threshold = page_fraction * len(page_list)
    return {line for line, count in count_dict.items() if count > threshold}


def clean_page(page_text: str, repeated_set: set[str], edge_lines: int) -> str:
    """Remove figure text, comments, number-only lines, and repeated edge lines from a page."""
    page_text = PICTURE_TEXT_PATTERN.sub("", page_text)
    page_text = HTML_COMMENT_PATTERN.sub("", page_text)
    line_list = page_text.splitlines()
    nonempty_index_list = [index for index, line in enumerate(line_list) if line.strip()]
    edge_index_set = set(nonempty_index_list[:edge_lines] + nonempty_index_list[-edge_lines:])
    kept_list: list[str] = []
    for index, line in enumerate(line_list):
        if is_number_only(line) or OMITTED_PICTURE_PATTERN.match(line.strip()):
            continue
        if (
            index in edge_index_set
            and normalize_edge_line(line) in repeated_set
            and not MARKDOWN_HEADING_PATTERN.match(line.strip())
        ):
            continue
        kept_list.append(line.rstrip())
    cleaned = "\n".join(kept_list)
    return MULTI_BLANK_PATTERN.sub("\n\n", cleaned).strip()


# ============= Headings and references =============


def heading_key(title: str) -> str:
    """Return a lowercase heading without markup or a section number prefix."""
    plain = strip_markup(title).strip().rstrip(":").strip()
    return HEADING_NUMBER_PREFIX.sub("", plain).strip().lower()


def match_heading(line: str, o_fallback: bool) -> tuple[int, str] | None:
    """
    Return (level, title) when a line is a section heading, else None.

    Parameters
    ----------
    line : str
        One line of cleaned text.
    o_fallback : bool
        True for `pypdf` text, which has no Markdown heading markers.

    Returns
    -------
    heading : tuple of (int, str) or None
        Heading level and cleaned title.
    """
    markdown_match = MARKDOWN_HEADING_PATTERN.match(line)
    if markdown_match:
        title = strip_markup(markdown_match.group(2))
        return (len(markdown_match.group(1)), title) if title else None
    if not o_fallback:
        return None
    stripped = line.strip()
    if FALLBACK_HEADING_PATTERN.match(stripped):
        return (stripped.split()[0].count(".") + 2, stripped)
    if stripped.lower().rstrip(":") in FALLBACK_NAMED_HEADINGS:
        return (2, stripped)
    return None


def find_references_offset(text: str, references_heading_list: list[str], o_fallback: bool) -> int:
    """
    Return the character offset where the references section starts, or -1.

    A Markdown heading is preferred; a standalone line equal to a references heading is
    accepted as well, because both parsers sometimes emit it as plain bold text.
    """
    heading_set = {heading.lower() for heading in references_heading_list}
    offset = 0
    candidate_offset = -1
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        heading = match_heading(stripped, o_fallback)
        if heading and heading_key(heading[1]) in heading_set:
            return offset
        if candidate_offset < 0 and stripped and heading_key(stripped) in heading_set:
            candidate_offset = offset
        offset += len(line)
    return candidate_offset


def split_sections(text: str, o_fallback: bool) -> list[tuple[int, str, int]]:
    """Return (level, title, char_start) for every section start, front matter first."""
    start_list: list[tuple[int, str, int]] = [(0, "Front matter", 0)]
    offset = 0
    for line in text.splitlines(keepends=True):
        heading = match_heading(line.strip(), o_fallback)
        if heading:
            if offset == 0:
                start_list = []
            start_list.append((heading[0], heading[1], offset))
        offset += len(line)
    return start_list


# ============= Assembly =============


def assemble_paper(
    page_list: list[str], parser_config: ParserConfig, method: str, o_fallback: bool
) -> ParsedPaper:
    """
    Clean raw pages and split them into sections.

    Parameters
    ----------
    page_list : list of str
        Raw per-page text from one of the extractors.
    parser_config : ParserConfig
        Cleaning options.
    method : str
        Name of the extractor that produced `page_list`.
    o_fallback : bool
        Whether this is the fallback extractor.

    Returns
    -------
    paper : ParsedPaper
        Cleaned paper with sections whose offsets index `paper.text`.
    """
    repeated_set = find_repeated_edge_lines(
        page_list,
        parser_config.repeated_line_edge_lines,
        parser_config.repeated_line_page_fraction,
    )
    cleaned_page_list = [
        clean_page(page_text, repeated_set, parser_config.repeated_line_edge_lines)
        for page_text in page_list
    ]
    page_offset_list: list[int] = []
    part_list: list[str] = []
    offset = 0
    for page_text in cleaned_page_list:
        page_offset_list.append(offset)
        if not page_text:
            continue
        if part_list:
            part_list.append("\n\n")
            offset += 2
            page_offset_list[-1] = offset
        part_list.append(page_text)
        offset += len(page_text)
    text = "".join(part_list)

    references_offset = find_references_offset(text, parser_config.references_headings, o_fallback)
    o_references_removed = references_offset > 0
    if o_references_removed:
        text = text[:references_offset].rstrip()

    paper = ParsedPaper(
        text=text,
        section_list=[],
        page_count=len(page_list),
        parse_method=method,
        o_fallback=o_fallback,
        parser_version=parser_config.parser_version,
        o_references_removed=o_references_removed,
        page_offset_list=page_offset_list,
    )
    start_list = split_sections(text, o_fallback)
    for index, (level, title, char_start) in enumerate(start_list):
        char_end = start_list[index + 1][2] if index + 1 < len(start_list) else len(text)
        if not text[char_start:char_end].strip():
            continue
        paper.section_list.append(
            Section(
                index=len(paper.section_list),
                title=title,
                level=level,
                char_start=char_start,
                char_end=char_end,
                page_start=paper.page_of(char_start),
                page_end=paper.page_of(max(char_start, char_end - 1)),
            )
        )
    return paper


def parse_pdf(pdf_path: str | Path, parser_config: ParserConfig) -> ParsedPaper:
    """
    Parse a PDF with `pymupdf4llm`, falling back to `pypdf` when it fails or yields no text.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF.
    parser_config : ParserConfig
        Cleaning options.

    Returns
    -------
    paper : ParsedPaper
        Parsed paper; `o_fallback` is True when `pypdf` was used.

    Raises
    ------
    ParseError
        If neither parser produces at least `MIN_PARSED_CHARS` characters.
    """
    try:
        page_list = extract_pages_pymupdf4llm(pdf_path)
        paper = assemble_paper(page_list, parser_config, METHOD_PYMUPDF4LLM, o_fallback=False)
        if paper.char_count >= MIN_PARSED_CHARS:
            return paper
        logger.warning(
            "pymupdf4llm produced too little text, using pypdf",
            extra={"pdf_path": str(pdf_path), "chars": paper.char_count},
        )
    except (RuntimeError, ValueError, IndexError, KeyError, TypeError, AttributeError) as exc:
        logger.warning(
            "pymupdf4llm failed, using pypdf", extra={"pdf_path": str(pdf_path), "error": str(exc)}
        )

    try:
        page_list = extract_pages_pypdf(pdf_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ParseError(f"pypdf could not read {pdf_path}: {exc}") from exc
    paper = assemble_paper(page_list, parser_config, METHOD_PYPDF, o_fallback=True)
    if paper.char_count < MIN_PARSED_CHARS:
        raise ParseError(f"no usable text in {pdf_path} ({paper.char_count} characters)")
    return paper


# ============= Persistence =============


def parsed_paths(parsed_dir: str | Path, versioned_id: str) -> tuple[Path, Path]:
    """Return the text and metadata paths for a parsed paper."""
    safe_id = versioned_id.replace("/", "_")
    base_dir = Path(parsed_dir)
    return base_dir / f"{safe_id}.md", base_dir / f"{safe_id}.json"


def save_parsed(paper: ParsedPaper, parsed_dir: str | Path, versioned_id: str) -> Path:
    """Write the cleaned text and the section metadata; return the text path."""
    text_path, meta_path = parsed_paths(parsed_dir, versioned_id)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(paper.text, encoding="utf-8")
    meta_dict = {
        **paper.stats(),
        "page_offset_list": paper.page_offset_list,
        "section_list": [asdict(section) for section in paper.section_list],
    }
    meta_path.write_text(json.dumps(meta_dict, indent=1), encoding="utf-8")
    return text_path


def load_parsed(parsed_dir: str | Path, versioned_id: str) -> ParsedPaper | None:
    """Load a previously saved parse, or None when absent."""
    text_path, meta_path = parsed_paths(parsed_dir, versioned_id)
    if not (text_path.exists() and meta_path.exists()):
        return None
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    return ParsedPaper(
        text=text_path.read_text(encoding="utf-8"),
        section_list=[Section(**section_dict) for section_dict in meta_dict["section_list"]],
        page_count=meta_dict["page_count"],
        parse_method=meta_dict["parse_method"],
        o_fallback=meta_dict["o_fallback"],
        parser_version=meta_dict["parser_version"],
        o_references_removed=meta_dict["o_references_removed"],
        page_offset_list=meta_dict["page_offset_list"],
    )


def parse_and_cache(
    pdf_path: str | Path,
    parsed_dir: str | Path,
    versioned_id: str,
    parser_config: ParserConfig,
    o_run_from_scratch: bool = False,
) -> ParsedPaper:
    """Return the cached parse when it matches the parser version, else parse and save."""
    if not o_run_from_scratch:
        cached = load_parsed(parsed_dir, versioned_id)
        if cached is not None and cached.parser_version == parser_config.parser_version:
            return cached
    paper = parse_pdf(pdf_path, parser_config)
    save_parsed(paper, parsed_dir, versioned_id)
    return paper
