"""Hard negatives with `sentence_transformers.util.mine_hard_negatives` and the current embedding
model: candidates ranked 10 to 50, a relative margin so near-duplicates of the positive are not
negatives, and no negative from the positive's own section (SPEC S3.8).
"""

import json
import logging

from meridian.common.config import EmbedderConfig, load_config
from meridian.embedder.build_pairs import DEFAULT_CONFIG_PATH, PoolChunk, load_pool, pair_paths
from meridian.embedder.exclusion import assert_no_leakage, build_exclusion_list
from meridian.eval.datasets import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def select_negative(
    positive: str,
    candidate_list: list[str],
    section_by_text: dict[str, str],
    arxiv_by_text: dict[str, str],
) -> dict | None:
    """
    Return the first mined candidate that is not the positive and not from its section.

    Parameters
    ----------
    positive : str
        Positive chunk text.
    candidate_list : list of str
        Mined negatives, hardest first.
    section_by_text, arxiv_by_text : dict
        Pool metadata keyed by chunk text.

    Returns
    -------
    negative : dict or None
        `{"text", "arxiv_id", "section_key"}`, or None when every candidate is excluded.
    """
    positive_section = section_by_text.get(positive)
    for candidate in candidate_list:
        if candidate == positive or section_by_text.get(candidate) == positive_section:
            continue
        return {
            "text": candidate,
            "arxiv_id": arxiv_by_text.get(candidate, ""),
            "section_key": section_by_text.get(candidate, ""),
        }
    return None


def mine(
    pair_list: list[dict], pool: list[PoolChunk], config: EmbedderConfig, model
) -> tuple[list[dict], dict]:
    """
    Mine one hard negative per pair.

    Returns
    -------
    triplet_list, stats : tuple
        Triplets with `query`, `positive`, `negative`, and provenance; and drop counts.
    """
    from datasets import Dataset
    from sentence_transformers.util import mine_hard_negatives

    negative_config = config.hard_negatives
    section_by_text = {chunk.text: chunk.section_key for chunk in pool}
    arxiv_by_text = {chunk.text: chunk.arxiv_id for chunk in pool}
    pair_by_key = {(pair["query"], pair["positive"]): pair for pair in pair_list}
    mined = mine_hard_negatives(
        Dataset.from_dict(
            {
                "anchor": [pair["query"] for pair in pair_list],
                "positive": [pair["positive"] for pair in pair_list],
            }
        ),
        model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=list(dict.fromkeys(chunk.text for chunk in pool)),
        range_min=negative_config.range_min,
        range_max=negative_config.range_max,
        relative_margin=negative_config.relative_margin,
        num_negatives=negative_config.num_candidates,
        sampling_strategy="top",
        output_format="n-tuple",
        query_prompt=config.query_prefix,
        batch_size=negative_config.batch_size,
        verbose=False,
    )
    negative_columns = [name for name in mined.column_names if name.startswith("negative")]
    triplet_list: list[dict] = []
    dropped_same_section = 0
    for row in mined:
        negative = select_negative(
            row["positive"],
            [row[name] for name in negative_columns],
            section_by_text,
            arxiv_by_text,
        )
        if negative is None:
            dropped_same_section += 1
            continue
        pair = pair_by_key[(row["anchor"], row["positive"])]
        triplet_list.append({**pair, "negative": negative["text"], "negatives": [negative]})
    stats = {
        "pairs": len(pair_list),
        "mined_rows": len(mined),
        "dropped_without_candidates": len(pair_list) - len(mined),
        "dropped_same_section": dropped_same_section,
        "triplets": len(triplet_list),
    }
    return triplet_list, stats


def cli_mine(args) -> int:
    """`meridian embedder negatives`: mine with the base model and write triplets."""
    from sentence_transformers import SentenceTransformer

    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(EmbedderConfig, config_path)
    paths = pair_paths(config)
    pair_list = read_jsonl(paths["train"])
    pool = load_pool(paths["pool"])
    triplet_list, stats = mine(pair_list, pool, config, SentenceTransformer(config.base_model))
    assert_no_leakage(triplet_list, build_exclusion_list())
    write_jsonl(paths["triplets"], triplet_list)
    print(json.dumps(stats, indent=2))
    return 0
