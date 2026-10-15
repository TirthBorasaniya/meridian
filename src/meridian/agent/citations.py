"""Citation handling: sentence splitting, `[c:<chunk_id>]` and `[w:<n>]` markers, alias resolution,
validation against the context, and short excerpts.

The generator cites short aliases (`[c:1]`); they are resolved to chunk ids after generation so
the model never has to copy UUIDs, and an alias outside the context is an invalid citation.
"""

import re

# ============= Patterns =============

MARKER_GROUP_PATTERN = re.compile(
    r"\[\s*((?:c|w)\s*:\s*[^\]\s,]+(?:\s*,\s*(?:c|w)\s*:\s*[^\]\s,]+)*)\s*\]"
)
SINGLE_MARKER_PATTERN = re.compile(r"\[(c|w):([^\]\s]+)\]")
TRAILING_MARKERS_PATTERN = re.compile(r"([.!?])((?:\s*\[(?:c|w):[^\]\s]+\])+)")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*-])")
ABBREVIATION_LIST = [
    "e.g.",
    "i.e.",
    "et al.",
    "etc.",
    "vs.",
    "cf.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Sec.",
    "Tab.",
    "No.",
    "approx.",
]
PLACEHOLDER = "⁣"
WEB_KIND = "w"
CORPUS_KIND = "c"


# ============= Normalization and splitting =============


def normalize_markers(text: str) -> str:
    """
    Rewrite markers into one canonical form.

    `[c:1, c:2]` becomes `[c:1][c:2]`, and markers placed after a sentence's final punctuation
    move before it, so that each marker stays with the sentence it supports.
    """

    def expand(match: re.Match) -> str:
        part_list = [part.replace(" ", "") for part in match.group(1).split(",")]
        return "".join(f"[{part}]" for part in part_list)

    expanded = MARKER_GROUP_PATTERN.sub(expand, text)
    return TRAILING_MARKERS_PATTERN.sub(
        lambda match: f"{match.group(2).strip()}{match.group(1)}", expanded
    )


def split_sentences(text: str) -> list[str]:
    """Split normalized answer text into sentences, protecting common abbreviations."""
    protected = text
    for abbreviation in ABBREVIATION_LIST:
        protected = protected.replace(abbreviation, abbreviation.replace(".", PLACEHOLDER))
    sentence_list: list[str] = []
    for paragraph in protected.split("\n"):
        for piece in SENTENCE_BOUNDARY_PATTERN.split(paragraph.strip()):
            restored = piece.replace(PLACEHOLDER, ".").strip()
            if restored:
                sentence_list.append(restored)
    return sentence_list


def extract_citations(sentence: str) -> list[tuple[str, str]]:
    """Return (kind, key) for each marker in a sentence, in order, without duplicates."""
    seen_list: list[tuple[str, str]] = []
    for match in SINGLE_MARKER_PATTERN.finditer(sentence):
        pair = (match.group(1), match.group(2))
        if pair not in seen_list:
            seen_list.append(pair)
    return seen_list


def strip_citations(sentence: str) -> str:
    """Return a sentence without markers."""
    stripped = SINGLE_MARKER_PATTERN.sub("", sentence)
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(stripped.split()))


# ============= Aliases =============


def resolve_aliases(text: str, alias_dict: dict[str, str], web_count: int) -> tuple[str, list[str]]:
    """
    Replace `[c:<alias>]` by `[c:<chunk_id>]` and drop markers that point outside the context.

    Parameters
    ----------
    text : str
        Generator output with alias markers.
    alias_dict : dict
        Maps an alias such as `"1"` to a chunk id.
    web_count : int
        Number of web results; `[w:n]` is valid for 1 <= n <= web_count.

    Returns
    -------
    resolved, invalid_list : tuple of (str, list of str)
        Text with resolved markers, and the invalid markers that were removed.
    """
    invalid_list: list[str] = []
    chunk_id_set = set(alias_dict.values())

    def replace(match: re.Match) -> str:
        kind, key = match.group(1), match.group(2)
        if kind == CORPUS_KIND:
            if key in alias_dict:
                return f"[c:{alias_dict[key]}]"
            if key in chunk_id_set:
                return f"[c:{key}]"
        elif key.isdigit() and 1 <= int(key) <= web_count:
            return f"[w:{int(key)}]"
        invalid_list.append(match.group(0))
        return ""

    return SINGLE_MARKER_PATTERN.sub(replace, normalize_markers(text)), invalid_list


def invalid_citations(text: str, context_chunk_ids: set[str], web_count: int) -> list[str]:
    """Return markers whose chunk id is not in the context or whose web index is out of range."""
    invalid_list: list[str] = []
    for kind, key in extract_citations(text):
        if kind == CORPUS_KIND and key not in context_chunk_ids:
            invalid_list.append(f"[c:{key}]")
        if kind == WEB_KIND and not (key.isdigit() and 1 <= int(key) <= web_count):
            invalid_list.append(f"[w:{key}]")
    return invalid_list


def cited_chunk_ids(text: str) -> list[str]:
    """Return cited corpus chunk ids in order of first appearance."""
    return [key for kind, key in extract_citations(text) if kind == CORPUS_KIND]


def cited_web_indices(text: str) -> list[int]:
    """Return cited web result numbers in order of first appearance."""
    return [int(key) for kind, key in extract_citations(text) if kind == WEB_KIND and key.isdigit()]


# ============= Excerpts =============


def excerpt(text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
    """Return at most `max_sentences` sentences of a chunk, cut at `max_chars`."""
    flat = " ".join(line.lstrip("#* ").strip() for line in text.splitlines() if line.strip())
    sentence_list = split_sentences(flat)[:max_sentences]
    joined = " ".join(sentence_list)
    return joined if len(joined) <= max_chars else joined[: max_chars - 3].rstrip() + "..."
