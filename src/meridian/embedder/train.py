"""Fine-tuning with `SentenceTransformerTrainer` and `MultipleNegativesRankingLoss`, starting from
the current embedding model, with early stopping on dev recall@10 (SPEC S3.8). Batch size 64 on a
GPU, 32 on CPU; the report records the hardware and wall-clock time.
"""

import json
import logging
import platform
import time
from pathlib import Path

from meridian.common.config import EmbedderConfig, load_config
from meridian.embedder.build_pairs import DEFAULT_CONFIG_PATH, PoolChunk, load_pool, pair_paths
from meridian.embedder.exclusion import assert_no_leakage, build_exclusion_list
from meridian.eval.datasets import read_jsonl

logger = logging.getLogger(__name__)


def device_description() -> tuple[str, str]:
    """Return (device type, human-readable hardware)."""
    import torch

    if torch.cuda.is_available():
        return "cuda", f"CUDA {torch.cuda.get_device_name(0)}"
    return "cpu", f"CPU {platform.machine()} {platform.processor() or ''}".strip()


def dev_evaluator(dev_pair_list: list[dict], pool: list[PoolChunk], query_prefix: str):
    """
    Build the dev information-retrieval evaluator over the whole pool.

    Each dev query's relevant document is its source chunk; the corpus is every pool chunk.
    """
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    corpus: dict = {chunk.chunk_id: chunk.text for chunk in pool}
    queries = {f"q{index}": pair["query"] for index, pair in enumerate(dev_pair_list)}
    relevant_docs = {f"q{index}": {pair["chunk_id"]} for index, pair in enumerate(dev_pair_list)}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="dev",
        query_prompt=query_prefix,
        precision_recall_at_k=[1, 5, 10],
        show_progress_bar=False,
    )


def recall_key(metric_dict: dict) -> str:
    """Return the evaluator's recall@10 metric name."""
    for key in metric_dict:
        if key.endswith("recall@10"):
            return key
    raise KeyError(f"no recall@10 among {sorted(metric_dict)}")


def train(
    config: EmbedderConfig,
    triplet_list: list[dict],
    dev_pair_list: list[dict],
    pool: list[PoolChunk],
) -> dict:
    """
    Fine-tune the base model and save it to `paths.output_dir`.

    Returns
    -------
    report : dict
        Hardware, wall-clock seconds, dataset sizes, dev metrics before and after, and the
        selected checkpoint.
    """
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    from transformers import EarlyStoppingCallback

    device, hardware = device_description()
    training = config.training
    batch_size = training.batch_size_gpu if device == "cuda" else training.batch_size_cpu
    model = SentenceTransformer(config.base_model, device=device)
    evaluator = dev_evaluator(dev_pair_list, pool, config.query_prefix)
    before = evaluator(model)
    metric_name = recall_key(before)
    train_dataset = Dataset.from_dict(
        {
            "anchor": [triplet["query"] for triplet in triplet_list],
            "positive": [triplet["positive"] for triplet in triplet_list],
            "negative": [triplet["negative"] for triplet in triplet_list],
        }
    )
    work_dir = Path(config.paths.work_dir)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(work_dir / "checkpoints"),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=training.learning_rate,
        warmup_ratio=training.warmup_ratio,
        seed=config.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"anchor": config.query_prefix},
        eval_strategy="steps",
        eval_steps=training.eval_steps,
        save_strategy="steps",
        save_steps=training.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{metric_name}",
        greater_is_better=True,
        logging_steps=training.eval_steps,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=MultipleNegativesRankingLoss(model),
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training.early_stopping_patience)],
    )
    started = time.perf_counter()
    trainer.train()
    wall_clock_s = time.perf_counter() - started
    after = evaluator(model)
    output_dir = Path(config.paths.output_dir)
    model.save_pretrained(str(output_dir))
    return {
        "base_model": config.base_model,
        "device": device,
        "hardware": hardware,
        "batch_size": batch_size,
        "wall_clock_s": wall_clock_s,
        "train_triplets": len(triplet_list),
        "dev_pairs": len(dev_pair_list),
        "pool_chunks": len(pool),
        "dev_metric": metric_name,
        "dev_before": before,
        "dev_after": after,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "global_steps": trainer.state.global_step,
        "output_dir": str(output_dir),
        "config_sha256": config.sha256,
    }


def cli_train(args) -> int:
    """`meridian embedder train`: train after re-checking the exclusion list."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(EmbedderConfig, config_path)
    paths = pair_paths(config)
    triplet_list = read_jsonl(paths["triplets"])
    dev_pair_list = read_jsonl(paths["dev"])
    excluded_set = build_exclusion_list()
    assert_no_leakage(triplet_list + dev_pair_list, excluded_set)
    report = train(config, triplet_list, dev_pair_list, load_pool(paths["pool"]))
    paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("hardware", "wall_clock_s", "dev_metric")}, indent=2
        )
    )
    return 0
