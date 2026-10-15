"""Command-line entry point `meridian`, dispatching subcommands to lazily imported modules.

Each subcommand names a `module:function` handler that receives the parsed arguments, so that
`meridian --help` does not import torch or the Qdrant client.
"""

import argparse
import importlib
import sys
from collections.abc import Callable

from meridian.common.logging import configure_logging
from meridian.common.settings import get_settings

# ============= Command table =============

# each entry: (command path, handler "module:function", help, argument adder)
CommandSpec = tuple[tuple[str, ...], str, str, Callable[[argparse.ArgumentParser], None]]


def _no_args(parser: argparse.ArgumentParser) -> None:
    """Add no arguments."""


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", dest="o_dry_run", action="store_true")


def _add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--since",
        default="last_success",
        help="'last_success' or an ISO date; papers submitted on or after it are fetched",
    )
    parser.add_argument("--until", default=None, help="ISO date upper bound, inclusive")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--deadline-minutes", type=float, default=None)
    parser.add_argument("--no-gate", dest="o_skip_gate", action="store_true")
    parser.add_argument("--batch-id", default=None)
    _add_dry_run(parser)


def _add_backfill_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=["seed-ids", "query"],
        default="seed-ids",
        help="seed-ids reproduces the recorded corpus; query re-runs the relevance query",
    )
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument(
        "--legacy-pdf-dir",
        default=None,
        help="reuse PDFs already downloaded under this directory instead of re-fetching",
    )


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-config", default=None, help="override configs/index.yaml")
    parser.add_argument("--collection", default=None, help="explicit collection name")
    parser.add_argument(
        "--switch-alias",
        dest="o_switch_alias",
        action="store_true",
        help="point meridian_live at the new collection (initial build only)",
    )


def _add_rebuild_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-config", default=None, help="index config for the new version")
    parser.add_argument("--skip-answer-eval", dest="o_skip_answer_eval", action="store_true")
    _add_dry_run(parser)


def _add_switch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("collection", help="collection name, for example meridian_v3")


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="experiment config path")


def _add_build_set_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-items", type=int, default=150)
    parser.add_argument("--n-fast", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="eval/retrieval_set.jsonl")


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)


def _add_release_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", default=".tmp/release")


COMMAND_LIST: list[CommandSpec] = [
    (
        ("backfill",),
        "meridian.ingest.pipeline:cli_backfill",
        "fetch, parse, and record the initial corpus in the manifest",
        _add_backfill_args,
    ),
    (
        ("ingest",),
        "meridian.ingest.pipeline:cli_ingest",
        "incremental ingestion into the live collection with the fast regression gate",
        _add_ingest_args,
    ),
    (
        ("index", "build"),
        "meridian.ingest.build:cli_build",
        "build a new versioned collection from the parsed papers in the manifest",
        _add_build_args,
    ),
    (
        ("index", "status"),
        "meridian.ingest.index:cli_status",
        "show collections, alias target, and point counts",
        _no_args,
    ),
    (
        ("index", "switch"),
        "meridian.ingest.rebuild:cli_switch",
        "atomically point the live alias at a collection",
        _add_switch_args,
    ),
    (
        ("rebuild",),
        "meridian.ingest.rebuild:cli_rebuild",
        "build meridian_v{N+1}, run the full gate, and switch the alias on pass",
        _add_rebuild_args,
    ),
    (
        ("rollback",),
        "meridian.ingest.rebuild:cli_rollback",
        "point the live alias back at the previous kept version",
        _no_args,
    ),
    (
        ("scheduler",),
        "meridian.ingest.scheduler:cli_scheduler",
        "run the daily ingestion scheduler in the foreground",
        _no_args,
    ),
    (
        ("build-retrieval-set",),
        "meridian.eval.build_retrieval_set:cli_build",
        "generate the labeled retrieval set R from the live collection",
        _add_build_set_args,
    ),
    (
        ("eval", "retrieval"),
        "meridian.eval.retrieval_eval:cli_eval",
        "E1: retrieval quality and stage latency per configuration",
        _add_eval_args,
    ),
    (
        ("eval", "answers"),
        "meridian.eval.answer_eval:cli_eval",
        "E2: RAGAS, citation precision, and refusal accuracy",
        _add_eval_args,
    ),
    (
        ("eval", "serving"),
        "meridian.eval.serving_eval:cli_eval",
        "E3: end-to-end latency, cost, and cache behavior",
        _add_eval_args,
    ),
    (
        ("eval", "ingest"),
        "meridian.eval.ingest_eval:cli_eval",
        "E4: simulated daily batches and recorded CI runs",
        _add_eval_args,
    ),
    (
        ("eval", "embedder"),
        "meridian.eval.embedder_eval:cli_eval",
        "E5: base versus domain-adapted embedding model",
        _add_eval_args,
    ),
    (
        ("eval", "demo"),
        "meridian.eval.demo_eval:cli_eval",
        "E6: public demo cold start, latency, and rate limiting",
        _add_eval_args,
    ),
    (
        ("embedder", "pairs"),
        "meridian.embedder.build_pairs:cli_build_pairs",
        "generate synthetic (query, chunk) training pairs",
        _add_eval_args,
    ),
    (
        ("embedder", "negatives"),
        "meridian.embedder.mine_negatives:cli_mine",
        "mine hard negatives with the current embedding model",
        _add_eval_args,
    ),
    (
        ("embedder", "train"),
        "meridian.embedder.train:cli_train",
        "fine-tune the embedding model with MultipleNegativesRankingLoss",
        _add_eval_args,
    ),
    (
        ("embedder", "push"),
        "meridian.embedder.card:cli_push",
        "write the model card and push the adapted model to the Hub",
        _add_eval_args,
    ),
    (
        ("release", "publish"),
        "meridian.release.publish:cli_publish",
        "snapshot the live collection and publish it to the index dataset repo",
        _add_release_args,
    ),
    (
        ("release", "restore"),
        "meridian.release.restore:cli_restore",
        "restore the latest accepted snapshot listed in latest.json",
        _add_release_args,
    ),
    (
        ("serve",),
        "meridian.serving.app:cli_serve",
        "run the FastAPI service",
        _add_serve_args,
    ),
]


# ============= Parser construction =============


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree from `COMMAND_LIST`."""
    parser = argparse.ArgumentParser(prog="meridian", description="Meridian command line")
    subparsers = parser.add_subparsers(dest="command", required=True)
    group_dict: dict[str, argparse._SubParsersAction] = {}
    for command_path, handler, help_text, add_arguments in COMMAND_LIST:
        if len(command_path) == 1:
            command_parser = subparsers.add_parser(command_path[0], help=help_text)
        else:
            group_name, leaf_name = command_path
            if group_name not in group_dict:
                group_parser = subparsers.add_parser(group_name, help=f"{group_name} commands")
                group_dict[group_name] = group_parser.add_subparsers(
                    dest="subcommand", required=True
                )
            command_parser = group_dict[group_name].add_parser(leaf_name, help=help_text)
        command_parser.set_defaults(handler=handler)
        add_arguments(command_parser)
    return parser


def resolve_handler(handler: str) -> Callable[[argparse.Namespace], int | None]:
    """Import and return the function named by a `module:function` string."""
    module_name, function_name = handler.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, configure logging, and run the selected handler."""
    settings = get_settings()
    configure_logging(settings.service_name, settings.log_level)
    args = build_parser().parse_args(argv)
    exit_code = resolve_handler(args.handler)(args)
    return int(exit_code or 0)


if __name__ == "__main__":
    sys.exit(main())
