"""Unit tests for the command-line parser."""

import pytest

from meridian.cli import COMMAND_LIST, build_parser

pytestmark = pytest.mark.unit


def test_parser_accepts_every_registered_command():
    parser = build_parser()
    for command_path, handler, _help, _adder in COMMAND_LIST:
        argv = list(command_path)
        if command_path == ("index", "switch"):
            argv.append("meridian_v2")
        if command_path == ("release", "runlog"):
            argv += ["--exit-code", "0", "--run-id", "1", "--started-utc", "2026-10-14T00:00:00Z"]
        args = parser.parse_args(argv)
        assert args.handler == handler


def test_ingest_defaults_to_last_success():
    args = build_parser().parse_args(["ingest"])
    assert args.since == "last_success"
    assert args.o_skip_gate is False
