"""Synthetic (query, positive chunk) training pairs (SPEC S3.8).

The pool is every live-collection chunk from a paper outside the exclusion list (R, A, O); when
it is smaller than `min_pool_chunks`, relevance-ranked papers beyond the corpus are downloaded,
parsed, and chunked into a separate training pool that is never indexed. The `pair_writer`
role (`openai/gpt-oss-120b`, Apache 2.0, never a Llama model) writes queries per chunk. The dev
split holds whole papers, so no dev chunk's paper appears in training.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from meridian.common.config import EmbedderConfig, get_corpus_config, get_index_config, load_config
from meridian.common.llm_client import LLMClient
from meridian.embedder.exclusion import assert_no_leakage, build_exclusion_list
from meridian.eval.datasets import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/embedder.yaml"
PAIR_PROMPT = (
    "You write search queries for training a retrieval model over research papers. Given one "
    "passage, write {n} different questions or keyword queries that a researcher might type and "
    "that this passage answers. Vary the phrasing and specificity; do not copy long phrases from "
    'the passage. Reply with a JSON object {{"queries": ["...", ...]}}.'
)


@dataclass
class PoolChunk:
    """One chunk available for training."""

    chunk_id: str
    arxiv_id: str
    section_key: str
    text: str
    source: str


# ============= Pool =============


def corpus_pool(
    payload_list: list[dict], excluded_set: set[str], min_tokens: int
) -> list[PoolChunk]:
    """Return eligible live-collection chunks from papers outside the exclusion list."""
    return [
        PoolChunk(
            chunk_id=payload["chunk_id"],
            arxiv_id=payload["arxiv_id"],
            section_key=f"{payload['arxiv_id']}v{payload['version']}:{payload['section_index']}",
            text=payload["text"],
            source="corpus",
        )
        for payload in payload_list
        if payload["arxiv_id"] not in excluded_set
        and int(payload.get("token_count", 0)) >= min_tokens
    ]


def extra_pool(
    config: EmbedderConfig, excluded_set: set[str], known_id_set: set[str]
) -> list[PoolChunk]:
    """
    Download, parse, and chunk relevance-ranked papers outside the corpus and the exclusions.

    The papers come from the recorded backfill query beyond the corpus; they are chunked with
    the index chunker but never written to Qdrant.
    """
    from meridian.ingest.arxiv_client import ArxivClient, PdfUnavailableError, backfill_query
    from meridian.ingest.chunk import chunk_paper, load_tokenizer
    from meridian.ingest.parse import ParseError, parse_and_cache

    corpus_config = get_corpus_config()
    index_config = get_index_config()
    tokenizer, max_tokens = load_tokenizer(index_config.dense.model)
    client = ArxivClient(corpus_config)
    pool_dir = Path(config.paths.pool_dir)
    chunk_list: list[PoolChunk] = []
    taken = 0
    for meta in client.iter_search(
        backfill_query(corpus_config.arxiv), "relevance", "descending", limit=None
    ):
        if taken >= config.extra_pool_papers:
            break
        if meta.arxiv_id in excluded_set or meta.arxiv_id in known_id_set or meta.o_withdrawn:
            continue
        pdf_path = pool_dir / "pdfs" / f"{meta.versioned_id}.pdf"
        try:
            if not pdf_path.exists():
                client.download_pdf(meta, pdf_path)
            paper = parse_and_cache(
                pdf_path, pool_dir / "parsed", meta.versioned_id, index_config.parser
            )
        except (PdfUnavailableError, ParseError, OSError) as exc:
            logger.warning(
                "training pool paper skipped", extra={"paper": meta.versioned_id, "error": str(exc)}
            )
            continue
        taken += 1
        for chunk in chunk_paper(paper, meta, tokenizer, max_tokens, index_config.chunker):
            if chunk.token_count >= config.min_chunk_tokens:
                chunk_list.append(
                    PoolChunk(
                        chunk_id=chunk.chunk_id,
                        arxiv_id=meta.arxiv_id,
                        section_key=f"{meta.versioned_id}:{chunk.section_index}",
                        text=chunk.text,
                        source="extra",
                    )
                )
    client.close()
    return chunk_list


# ============= Pairs =============


def write_queries(llm: LLMClient, text: str, n: int) -> list[str]:
    """Ask the pair writer for `n` queries answered by one passage."""
    try:
        reply = llm.chat_json(
            "pair_writer",
            [
                {"role": "system", "content": PAIR_PROMPT.format(n=n)},
                {"role": "user", "content": f"Passage:\n{text}"},
            ],
        )
    except ValueError as exc:
        logger.warning("pair writer reply unusable", extra={"error": str(exc)})
        return []
    query_list = [str(query).strip() for query in reply.get("queries", []) if str(query).strip()]
    return list(dict.fromkeys(query_list))[:n]


def split_papers(pool: list[PoolChunk], n_dev_chunks: int, seed: int) -> set[str]:
    """Pick whole papers for the dev split until they hold at least `n_dev_chunks` chunks."""
    paper_count: dict[str, int] = {}
    for chunk in pool:
        paper_count[chunk.arxiv_id] = paper_count.get(chunk.arxiv_id, 0) + 1
    paper_list = sorted(paper_count)
    random.Random(seed).shuffle(paper_list)
    dev_set: set[str] = set()
    total = 0
    for arxiv_id in paper_list:
        if total >= n_dev_chunks:
            break
        dev_set.add(arxiv_id)
        total += paper_count[arxiv_id]
    return dev_set


def build_pairs(
    pool: list[PoolChunk], llm: LLMClient, config: EmbedderConfig
) -> tuple[list[dict], list[dict]]:
    """
    Generate train and dev pairs.

    Parameters
    ----------
    pool : list of PoolChunk
        Eligible chunks.
    llm : LLMClient
        Client with a `pair_writer` role.
    config : EmbedderConfig
        Target counts and seed.

    Returns
    -------
    train_list, dev_list : tuple of list of dict
        Pairs with `query`, `positive`, `chunk_id`, `arxiv_id`, `section_key`, `source`.
    """
    n_dev_chunks = -(-config.n_dev_pairs // config.queries_per_chunk)
    dev_paper_set = split_papers(pool, n_dev_chunks * 2, config.seed)
    order = list(pool)
    random.Random(config.seed).shuffle(order)
    train_list: list[dict] = []
    dev_list: list[dict] = []
    for chunk in order:
        is_dev = chunk.arxiv_id in dev_paper_set
        target = dev_list if is_dev else train_list
        limit = config.n_dev_pairs if is_dev else config.n_pairs
        if len(target) >= limit:
            if len(train_list) >= config.n_pairs and len(dev_list) >= config.n_dev_pairs:
                break
            continue
        for query in write_queries(llm, chunk.text, config.queries_per_chunk):
            if len(target) >= limit:
                break
            target.append(
                {
                    "query": query,
                    "positive": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "arxiv_id": chunk.arxiv_id,
                    "section_key": chunk.section_key,
                    "source": chunk.source,
                }
            )
    return train_list, dev_list


def pair_paths(config: EmbedderConfig) -> dict[str, Path]:
    """Return the working files of the embedder pipeline."""
    work_dir = Path(config.paths.work_dir)
    return {
        "pool": work_dir / "pool.jsonl",
        "train": work_dir / "pairs_train.jsonl",
        "dev": work_dir / "pairs_dev.jsonl",
        "triplets": work_dir / "triplets_train.jsonl",
        "meta": work_dir / "pairs_meta.json",
        "report": work_dir / "train_report.json",
    }


def load_pool(path: Path) -> list[PoolChunk]:
    """Read the pool file."""
    return [PoolChunk(**record) for record in read_jsonl(path)]


def cli_build_pairs(args) -> int:
    """`meridian embedder pairs`: build the pool and the pairs, refusing on any leakage."""
    from meridian.common.llm_client import build_llm_client
    from meridian.ingest.index import get_qdrant_client, iter_payloads

    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(EmbedderConfig, config_path)
    excluded_set = build_exclusion_list()
    payload_list = [
        record.payload or {}
        for record in iter_payloads(get_qdrant_client(), get_index_config().alias)
    ]
    pool = corpus_pool(payload_list, excluded_set, config.min_chunk_tokens)
    if len(pool) < config.min_pool_chunks:
        known_id_set = {payload["arxiv_id"] for payload in payload_list}
        pool += extra_pool(config, excluded_set, known_id_set)
    llm = build_llm_client()
    train_list, dev_list = build_pairs(pool, llm, config)
    if len(train_list) < config.n_pairs or len(dev_list) < config.n_dev_pairs:
        raise RuntimeError(f"pool too small: {len(train_list)} train and {len(dev_list)} dev pairs")
    assert_no_leakage(train_list + dev_list, excluded_set)
    paths = pair_paths(config)
    write_jsonl(paths["pool"], [asdict(chunk) for chunk in pool])
    write_jsonl(paths["train"], train_list)
    write_jsonl(paths["dev"], dev_list)
    meta = {
        "pair_writer": llm.models_config.role("pair_writer").model,
        "excluded_papers": len(excluded_set),
        "pool_chunks": len(pool),
        "pool_by_source": {
            source: sum(1 for chunk in pool if chunk.source == source)
            for source in ("corpus", "extra")
        },
        "train_pairs": len(train_list),
        "dev_pairs": len(dev_list),
        "dev_papers": len({pair["arxiv_id"] for pair in dev_list}),
        "config_sha256": config.sha256,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0
