"""The scheduled run log `results/ingest_runs.jsonl`: one record per CI run, appended and committed
by the daily workflow, read by E4.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from meridian.common.config import get_release_config
from meridian.common.results import git_sha

EXIT_DECISION_DICT = {0: "accepted", 2: "rejected", 3: "aborted"}


def load_json(path: str | None) -> dict | None:
    """Read a JSON file written by an earlier workflow step, or None when absent or empty."""
    if not path or not Path(path).exists() or not Path(path).read_text().strip():
        return None
    return json.loads(Path(path).read_text())


def build_run_record(
    ingest: dict | None,
    publish: dict | None,
    exit_code: int,
    run_id: str,
    started_utc: str,
    finished_utc: str | None = None,
) -> dict:
    """
    Return one run log record.

    Parameters
    ----------
    ingest : dict or None
        The batch record printed by `meridian ingest`.
    publish : dict or None
        The release manifest printed by `meridian release publish`, when a release was made.
    exit_code : int
        Exit code of `meridian ingest`.
    run_id : str
        CI run id.
    started_utc, finished_utc : str
        Run boundaries.

    Returns
    -------
    record : dict
        The run record.
    """
    finished = finished_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    duration_s = (
        datetime.fromisoformat(finished.replace("Z", "+00:00"))
        - datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    ).total_seconds()
    ingest = ingest or {}
    gate = ingest.get("eval") or {}
    return {
        "run_id": run_id,
        "started_utc": started_utc,
        "finished_utc": finished,
        "duration_s": duration_s,
        "exit_code": exit_code,
        "decision": ingest.get("decision") or EXIT_DECISION_DICT.get(exit_code, "error"),
        "batch_id": ingest.get("batch_id"),
        "papers_fetched": ingest.get("papers_fetched"),
        "papers_skipped": ingest.get("papers_skipped"),
        "papers_failed": ingest.get("papers_failed"),
        "papers_indexed": ingest.get("papers_indexed"),
        "chunks_added": ingest.get("chunks_added"),
        "recall_at_5": gate.get("value"),
        "baseline_recall_at_5": gate.get("baseline"),
        "snapshot": publish.get("name") if publish else None,
        "index_version": publish.get("index_version") if publish else ingest.get("index_version"),
        "git_sha": git_sha(),
    }


def gate_report_markdown(record: dict, ingest: dict | None) -> str:
    """Return the GitHub issue body for a failed or rejected run."""
    gate = (ingest or {}).get("eval") or {}
    line_list = [
        f"Daily index run `{record['run_id']}` finished with decision `{record['decision']}` "
        f"(exit code {record['exit_code']}). No snapshot was published.",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in ("batch_id", "papers_fetched", "papers_indexed", "papers_failed", "chunks_added"):
        line_list.append(f"| {key} | {record.get(key)} |")
    if gate:
        line_list.append(f"| recall@5 (R-fast) | {gate.get('value')} |")
        line_list.append(f"| baseline | {gate.get('baseline')} |")
        line_list.append(f"| max regression | {gate.get('max_regression')} |")
    failure_dict = (ingest or {}).get("failure_dict") or {}
    if failure_dict:
        line_list += ["", "Failed papers:", ""] + [
            f"- {key}: {value}" for key, value in failure_dict.items()
        ]
    return "\n".join(line_list) + "\n"


def append_record(record: dict, path: str | None = None) -> Path:
    """Append a record to the run log."""
    log_path = Path(path or get_release_config().run_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return log_path


def cli_runlog(args) -> int:
    """`meridian release runlog`: append this run's record and write the issue body."""
    ingest = load_json(args.ingest_output)
    record = build_run_record(
        ingest, load_json(args.publish_output), args.exit_code, args.run_id, args.started_utc
    )
    append_record(record)
    if record["decision"] != "accepted":
        Path(args.report_output).write_text(gate_report_markdown(record, ingest))
    print(json.dumps(record, indent=2))
    return 0
