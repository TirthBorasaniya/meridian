"""Model card generation and the Hugging Face Hub release of the adapted embedder."""

import json
from pathlib import Path

from meridian.common.config import EmbedderConfig, load_config
from meridian.embedder.build_pairs import DEFAULT_CONFIG_PATH, pair_paths

E5_RESULT_PATH = Path("results/e5_embedder.json")


def render_card(config: EmbedderConfig, report: dict, pair_meta: dict, e5: dict | None) -> str:
    """Return the model card Markdown, with metrics only from the training report and E5."""
    dev_metric = report["dev_metric"]
    evaluation = "E5 has not been run yet."
    if e5 is not None:
        evaluation = "Held-out retrieval set R-full (`results/e5_embedder.json`):\n\n"
        evaluation += (
            "| Pipeline | Model | recall@5 | recall@10 | MRR@10 | nDCG@10 |\n"
            "|---|---|---|---|---|---|\n"
        )
        for pipeline, model_dict in e5["metrics"]["comparison"].items():
            for model_name, metrics in model_dict.items():
                evaluation += (
                    f"| {pipeline} | {model_name} | {metrics['recall_at_5']:.4f} | "
                    f"{metrics['recall_at_10']:.4f} | {metrics['mrr_at_10']:.4f} | "
                    f"{metrics['ndcg_at_10']:.4f} |\n"
                )
    return f"""---
license: mit
base_model: {config.base_model}
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- retrieval
- arxiv
---

# {config.hub.repo_id.split("/")[-1]}

Domain-adapted version of `{config.base_model}` for retrieval over arXiv papers on LLM reasoning
and evaluation, trained for the Meridian research assistant.

## Usage

Queries take the prefix `{config.query_prefix}`; passages take no prefix.

## Training data

- {pair_meta.get("train_pairs")} synthetic (query, passage) pairs and {pair_meta.get("dev_pairs")}
  dev pairs, written by `{pair_meta.get("pair_writer")}` (Apache 2.0) from chunks of papers that
  appear in no evaluation set ({pair_meta.get("excluded_papers")} papers excluded).
- Hard negatives mined with the base model from candidate ranks
  {config.hard_negatives.range_min} to {config.hard_negatives.range_max}, relative margin
  {config.hard_negatives.relative_margin}, excluding the positive's own section.
- Passage text is not redistributed; most arXiv papers carry a license without redistribution
  rights.

## Training

`MultipleNegativesRankingLoss`, {config.training.epochs} epoch, learning rate
{config.training.learning_rate}, batch size {report["batch_size"]}, warmup ratio
{config.training.warmup_ratio}, seed {config.seed}, early stopping on dev `{dev_metric}`.
Hardware: {report["hardware"]}. Wall-clock training time: {report["wall_clock_s"]:.0f} seconds.

Dev `{dev_metric}`: {report["dev_before"][dev_metric]:.4f} before training,
{report["dev_after"][dev_metric]:.4f} after.

## Evaluation

{evaluation}
"""


def cli_push(args) -> int:
    """`meridian embedder push`: write the card into the model folder and upload it."""
    from huggingface_hub import HfApi

    from meridian.common.settings import get_settings

    config = load_config(EmbedderConfig, args.config or DEFAULT_CONFIG_PATH)
    paths = pair_paths(config)
    report = json.loads(paths["report"].read_text())
    pair_meta = json.loads(paths["meta"].read_text())
    e5 = json.loads(E5_RESULT_PATH.read_text()) if E5_RESULT_PATH.exists() else None
    output_dir = Path(config.paths.output_dir)
    (output_dir / "README.md").write_text(render_card(config, report, pair_meta, e5))
    token = get_settings().hf_token.get_secret_value()
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi(token=token)
    api.create_repo(
        config.hub.repo_id, repo_type="model", private=config.hub.private, exist_ok=True
    )
    api.upload_folder(folder_path=str(output_dir), repo_id=config.hub.repo_id, repo_type="model")
    print(f"https://huggingface.co/{config.hub.repo_id}")
    return 0
