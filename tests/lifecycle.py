"""The index lifecycle scenario of SPEC S7, shared by the in-memory unit test and the
integration test against the compose Qdrant server: build v1, run R-fast, switch the alias,
rebuild into v2 through the full gate, ingest a fixture batch, force a failing gate, assert the
batch points are deleted, and roll back.
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient

from meridian.common.config import IndexConfig, get_corpus_config, get_gate_config
from meridian.ingest import index as index_module
from meridian.ingest.build import build_collection, record_build_in_manifest
from meridian.ingest.gate import run_fast_gate, save_baseline
from meridian.ingest.manifest import STATUS_REJECTED, Manifest
from meridian.ingest.pipeline import IngestContext, run_increment
from meridian.ingest.rebuild import rollback, run_rebuild
from tests.helpers import (
    FakeArxiv,
    acquire_fixture_papers,
    fake_indexer_factory,
    fake_search_factory,
    lifecycle_papers,
    paper_meta,
    write_retrieval_set_for,
)

QUESTION_DICT = {
    "2490.00001": "intermediate reasoning steps before the final answer",
    "2490.00002": "stated confidence matches observed accuracy",
    "2490.00003": "most frequent final answer over sampled reasoning paths",
    "2490.00004": "position bias of a language model judge",
}


def run_lifecycle_scenario(client: QdrantClient, tmp_path: Path, index_config: IndexConfig) -> None:
    """Run the scenario and assert every step; collections use `index_config` names."""
    paper_list = lifecycle_papers()
    corpus_config = get_corpus_config()
    paths = corpus_config.paths.model_copy(
        update={
            "pdf_dir": str(tmp_path / "pdfs"),
            "parsed_dir": str(tmp_path / "parsed"),
            "manifest": str(tmp_path / "manifest.sqlite"),
        }
    )
    corpus_config = corpus_config.model_copy(update={"paths": paths})
    retrieval_set_path = tmp_path / "retrieval_set.jsonl"
    baseline_path = tmp_path / "baseline.json"
    gate_config = get_gate_config()
    search_factory = fake_search_factory(client)
    body_by_id = {
        f"{paper['arxiv_id']}v1": " ".join(body for _heading, body in paper["sections"]) * 10
        for paper in paper_list
    }
    arxiv = FakeArxiv(body_by_id)
    manifest = Manifest(paths.manifest)
    v1_name = index_module.collection_name(index_config.collection_prefix, 1)
    v2_name = index_module.collection_name(index_config.collection_prefix, 2)

    context = IngestContext(
        corpus_config=corpus_config,
        index_config=index_config,
        gate_config=gate_config,
        manifest=manifest,
        client=client,
        collection=v1_name,
        index_version=1,
        arxiv=arxiv,  # type: ignore[arg-type]
        baseline_path=str(baseline_path),
        retrieval_set_path=str(retrieval_set_path),
        batch_dir=tmp_path / "batches",
        alert_path=tmp_path / "alerts.jsonl",
    )

    # build meridian_v1 from four fixture papers
    acquire_fixture_papers(context, paper_list[:4])
    report = build_collection(
        client,
        manifest,
        index_config,
        corpus_config,
        build_batch_id="build-v1",
        indexer_factory=fake_indexer_factory,
    )
    assert report.collection == v1_name
    assert report.papers_indexed == 4
    assert index_module.count_papers(client, v1_name) == 4

    # switch the alias and record the build
    assert index_module.switch_alias(client, index_config.alias, v1_name) is None
    record_build_in_manifest(manifest, report, "build-v1")

    # run R-fast and store the baseline
    write_retrieval_set_for(client, v1_name, retrieval_set_path, QUESTION_DICT)
    fast_result = run_fast_gate(
        search_factory(v1_name, index_config),
        gate_config,
        v1_name,
        baseline_path,
        str(retrieval_set_path),
    )
    assert fast_result.o_pass and fast_result.o_baseline_created
    assert fast_result.value > 0.5
    save_baseline(baseline_path, fast_result, str(retrieval_set_path))

    # rebuild into meridian_v2 through the full gate
    rebuild_report = run_rebuild(
        client,
        manifest,
        index_config,
        corpus_config,
        gate_config,
        search_factory,
        answer_evaluator=lambda collection, config: 0.9,
        indexer_factory=fake_indexer_factory,
        baseline_path=str(baseline_path),
        retrieval_set_path=str(retrieval_set_path),
    )
    assert rebuild_report.o_promoted, rebuild_report.reason_list
    assert rebuild_report.candidate == v2_name
    assert index_module.alias_target(client, index_config.alias) == v2_name
    assert client.collection_exists(v1_name)

    # a worse candidate is declined and the alias stays
    declined = run_rebuild(
        client,
        manifest,
        index_config,
        corpus_config,
        gate_config,
        search_factory,
        answer_evaluator=lambda collection, config: 0.5 if collection != v2_name else 0.9,
        indexer_factory=fake_indexer_factory,
        baseline_path=str(baseline_path),
        retrieval_set_path=str(retrieval_set_path),
    )
    assert not declined.o_promoted
    assert any("faithfulness" in reason for reason in declined.reason_list)
    assert index_module.alias_target(client, index_config.alias) == v2_name
    client.delete_collection(declined.candidate)

    # ingest a fixture batch into the live version and force a failing gate
    context.collection = v2_name
    context.index_version = 2
    context.indexer = fake_indexer_factory(client, v2_name, 2, index_config)
    context.search = search_factory(v2_name, index_config)
    baseline_path.write_text(json.dumps({"metric": gate_config.fast_metric, "value": 1.5}))
    arxiv.search_list = [paper_meta(paper_list[4]["arxiv_id"])]
    points_before = index_module.count_points(client, v2_name)
    record = run_increment(context, since="2024-11-04", until="2024-11-05")
    assert record.decision == "rejected"
    assert record.chunks_added > 0
    assert record.chunks_deleted == record.chunks_added
    assert index_module.count_points(client, v2_name) == points_before
    assert (
        index_module.count_points(
            client, v2_name, index_module.field_filter("ingest_batch_id", record.batch_id)
        )
        == 0
    )
    assert manifest.get_paper(paper_list[4]["arxiv_id"], 1).status == STATUS_REJECTED

    # roll back to the kept previous version
    previous, target = rollback(client, index_config)
    assert (previous, target) == (v2_name, v1_name)
    assert index_module.alias_target(client, index_config.alias) == v1_name
