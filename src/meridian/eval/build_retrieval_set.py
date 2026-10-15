"""Construction of the labeled retrieval set R from the live collection (SPEC S6.1).

Chunks are sampled stratified by paper and section, one question is written per chunk by the
`question_writer` role, questions whose content tokens overlap the chunk by more than the
configured share are dropped and replaced, and the chunk's character span is recorded as the
relevant span. R-fast is a seeded subset of R-full.
"""

import json
import logging
import random
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from meridian.common.config import VersionedConfig, get_index_config, load_config
from meridian.common.llm_client import LLMClient, build_llm_client
from meridian.eval.datasets import write_jsonl
from meridian.eval.spans import LabeledSpan

logger = logging.getLogger(__name__)

# ============= Constants =============

DEFAULT_CONFIG_PATH = "configs/eval/retrieval_set.yaml"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORD_SET = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "by",
        "with",
        "as",
        "at",
        "from",
        "that",
        "this",
        "these",
        "those",
        "which",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "does",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "may",
        "might",
        "it",
        "its",
        "their",
        "there",
        "than",
        "then",
        "into",
        "over",
        "under",
        "about",
        "between",
        "such",
        "using",
        "used",
        "use",
        "based",
    ]
)
QUESTION_SYSTEM_PROMPT = (
    "You write evaluation questions for a search engine over research papers. Given one passage "
    "from a paper, write exactly one question that a researcher could ask and that this passage "
    "answers on its own. Rules: ask about a specific finding, method detail, number, or "
    "definition stated in the passage; do not mention 'the passage', 'the paper', or 'the "
    "authors'; paraphrase instead of copying distinctive phrases from the passage; use at most "
    '30 words. Reply with a JSON object {"question": "..."}.'
)
RETRY_SUFFIX = (
    " Your previous question reused too many words from the passage. Use different words "
    "(synonyms and a different sentence structure) while asking about the same content."
)


class RetrievalSetConfig(VersionedConfig):
    """configs/eval/retrieval_set.yaml."""

    n_items: int = Field(gt=0)
    n_fast: int = Field(gt=0)
    seed: int
    max_token_overlap: float = Field(gt=0.0, le=1.0)
    attempts_per_chunk: int = Field(ge=1)
    min_chunk_tokens: int = Field(ge=1)
    excluded_section_patterns: list[str]
    review_sample_size: int = Field(gt=0)
    review_seed: int
    review_top_k: int = Field(gt=0)
    output_path: str
    meta_path: str


# ============= Overlap filter =============


def content_tokens(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens that are not stopwords."""
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORD_SET]


def token_overlap(question: str, chunk_text: str) -> float:
    """
    Return the share of the question's content tokens that also occur in the chunk.

    Parameters
    ----------
    question : str
        Generated question.
    chunk_text : str
        Source chunk.

    Returns
    -------
    overlap : float
        In [0, 1]; 1.0 for a question with no content tokens, so such questions are dropped.
    """
    question_token_list = content_tokens(question)
    if not question_token_list:
        return 1.0
    chunk_token_set = set(content_tokens(chunk_text))
    shared = sum(1 for token in question_token_list if token in chunk_token_set)
    return shared / len(question_token_list)


# ============= Sampling =============


def eligible(payload: dict, config: RetrievalSetConfig) -> bool:
    """Return whether a chunk may be sampled."""
    title = str(payload.get("section_title", "")).lower()
    if any(pattern in title for pattern in config.excluded_section_patterns):
        return False
    return int(payload.get("token_count", 0)) >= config.min_chunk_tokens


def stratified_order(payload_list: list[dict], seed: int) -> list[dict]:
    """
    Order chunks so that consecutive draws spread over papers first and sections second.

    Papers are visited round-robin in a seeded random order. Within a paper, each visit takes a
    chunk from the section used least so far, so the first draw from every paper comes before
    any second draw, and a paper's sections are covered before any section repeats.

    Parameters
    ----------
    payload_list : list of dict
        Eligible chunk payloads.
    seed : int
        Random seed.

    Returns
    -------
    ordered_list : list of dict
        Every input payload exactly once, in draw order.
    """
    rng = random.Random(seed)
    section_dict: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for payload in payload_list:
        section_dict[payload["arxiv_id"]][int(payload["section_index"])].append(payload)
    for paper_sections in section_dict.values():
        for chunk_list in paper_sections.values():
            chunk_list.sort(key=lambda payload: payload["chunk_id"])
            rng.shuffle(chunk_list)
    paper_list = sorted(section_dict)
    rng.shuffle(paper_list)
    queue_dict: dict[str, list[list[dict]]] = {}
    for arxiv_id in paper_list:
        section_list = [section_dict[arxiv_id][index] for index in sorted(section_dict[arxiv_id])]
        rng.shuffle(section_list)
        queue_dict[arxiv_id] = section_list
    ordered_list: list[dict] = []
    while queue_dict:
        for arxiv_id in list(paper_list):
            section_queue = queue_dict.get(arxiv_id)
            if not section_queue:
                queue_dict.pop(arxiv_id, None)
                continue
            # take one chunk from the first section, then rotate that section to the back
            section_chunks = section_queue.pop(0)
            ordered_list.append(section_chunks.pop())
            if section_chunks:
                section_queue.append(section_chunks)
            if not section_queue:
                queue_dict.pop(arxiv_id, None)
        paper_list = [arxiv_id for arxiv_id in paper_list if arxiv_id in queue_dict]
    return ordered_list


# ============= Question writing =============


def write_question(llm: LLMClient, chunk_text: str, attempt: int) -> str:
    """Ask the question writer for one question about a chunk."""
    system_prompt = QUESTION_SYSTEM_PROMPT + (RETRY_SUFFIX if attempt > 0 else "")
    reply = llm.chat_json(
        "question_writer",
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Passage:\n{chunk_text}"},
        ],
    )
    return str(reply.get("question", "")).strip()


def build_items(
    ordered_list: list[dict],
    llm: LLMClient,
    config: RetrievalSetConfig,
) -> tuple[list[dict], dict]:
    """
    Generate questions in draw order until `n_items` pass the overlap filter.

    Parameters
    ----------
    ordered_list : list of dict
        Chunk payloads from `stratified_order`.
    llm : LLMClient
        Client with a `question_writer` role.
    config : RetrievalSetConfig
        Set construction parameters.

    Returns
    -------
    record_list, stats : tuple
        Dataset records and drop statistics.
    """
    record_list: list[dict] = []
    stats = {"chunks_drawn": 0, "questions_written": 0, "dropped_overlap": 0, "dropped_empty": 0}
    used_paper_count: dict[str, int] = defaultdict(int)
    for payload in ordered_list:
        if len(record_list) >= config.n_items:
            break
        stats["chunks_drawn"] += 1
        for attempt in range(config.attempts_per_chunk):
            try:
                question = write_question(llm, payload["text"], attempt)
            except ValueError as exc:
                logger.warning("question writer reply unusable", extra={"error": str(exc)})
                stats["dropped_empty"] += 1
                continue
            stats["questions_written"] += 1
            if not question:
                stats["dropped_empty"] += 1
                continue
            overlap = token_overlap(question, payload["text"])
            if overlap > config.max_token_overlap:
                stats["dropped_overlap"] += 1
                continue
            used_paper_count[payload["arxiv_id"]] += 1
            span = LabeledSpan(
                arxiv_id=payload["arxiv_id"],
                version=int(payload["version"]),
                start=int(payload["char_start"]),
                end=int(payload["char_end"]),
            )
            record_list.append(
                {
                    "qid": f"r{len(record_list) + 1:03d}",
                    "question": question,
                    "spans": [{**span.to_dict(), "source": "generated"}],
                    "source_chunk_id": payload["chunk_id"],
                    "section_title": payload.get("section_title", ""),
                    "parser_version": int(payload.get("parser_version", 0)),
                    "token_overlap": round(overlap, 4),
                    "in_fast": False,
                    "review": {"reviewed": False, "valid": None, "notes": ""},
                }
            )
            break
    stats["distinct_papers"] = len(used_paper_count)
    return record_list, stats


def assign_fast_split(record_list: list[dict], n_fast: int, seed: int) -> None:
    """Mark a seeded random subset of `n_fast` items as R-fast, in place."""
    rng = random.Random(seed)
    for index in rng.sample(range(len(record_list)), min(n_fast, len(record_list))):
        record_list[index]["in_fast"] = True


# ============= Entry point =============


def build_retrieval_set(
    payload_list: list[dict],
    llm: LLMClient,
    config: RetrievalSetConfig,
) -> tuple[list[dict], dict]:
    """Run sampling, question writing, filtering, and the split; return records and stats."""
    eligible_list = [payload for payload in payload_list if eligible(payload, config)]
    ordered_list = stratified_order(eligible_list, config.seed)
    record_list, stats = build_items(ordered_list, llm, config)
    if len(record_list) < config.n_items:
        raise RuntimeError(
            f"only {len(record_list)} of {config.n_items} questions passed the overlap filter"
        )
    assign_fast_split(record_list, config.n_fast, config.seed)
    stats["eligible_chunks"] = len(eligible_list)
    return record_list, stats


def cli_build(args) -> int:
    """`meridian build-retrieval-set`: generate R from the live collection."""
    from meridian.ingest.index import alias_target, get_qdrant_client, iter_payloads

    config = load_config(RetrievalSetConfig, DEFAULT_CONFIG_PATH)
    config = config.model_copy(
        update={"n_items": args.n_items, "n_fast": args.n_fast, "seed": args.seed}
    )
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"{output_path} exists; R is built once and then only reviewed")
    index_config = get_index_config()
    client = get_qdrant_client()
    payload_list = [record.payload or {} for record in iter_payloads(client, index_config.alias)]
    llm = build_llm_client()
    record_list, stats = build_retrieval_set(payload_list, llm, config)
    write_jsonl(output_path, record_list)
    meta_dict = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collection": alias_target(client, index_config.alias),
        "question_writer": llm.models_config.role("question_writer").model,
        "config_path": DEFAULT_CONFIG_PATH,
        "config_sha256": config.sha256,
        "stats": stats,
        "review": {"sample_size": config.review_sample_size, "reviewed": 0, "invalid": 0},
    }
    Path(config.meta_path).write_text(json.dumps(meta_dict, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0
