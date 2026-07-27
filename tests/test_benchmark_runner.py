from __future__ import annotations

import json
from pathlib import Path

from hcmaic.benchmark.runner import run_benchmark


def test_benchmark_runner_freezes_inputs_and_writes_required_reports(
    sample_root: Path, tmp_path: Path
):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        f"""
dataset_adapter: {{name: btc-style, version: v1}}
ingestion_backend: {{name: supplied-keyframes, version: v1}}
shot_detector: {{name: supplied, version: v1}}
embedding_provider: {{name: mock, version: mock-palette-v1}}
index_provider: {{name: exact-numpy, version: v1}}
fusion: {{name: single-stage, version: v1}}
reranker: {{name: identity, version: v1}}
benchmark_inputs:
  name: proxy-fixture
  version: v1
  params:
    dataset_root: "{sample_root.as_posix()}"
    queries: "{(sample_root / 'queries.jsonl').as_posix()}"
    qrels: "{(sample_root / 'qrels.jsonl').as_posix()}"
    warmups: 1
    repeats: 2
    top_k: 100
device: cpu
batch_size: 16
seed: 7
""",
        encoding="utf-8",
    )
    out = tmp_path / "benchmark-run"
    result = run_benchmark(config, out)

    required = {
        "benchmark_summary.json",
        "per_query_results.jsonl",
        "run_manifest.json",
        "failure_cases.md",
    }
    assert required <= {path.name for path in out.iterdir()}
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (out / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_hash"]
    assert manifest["config_hash"]
    assert manifest["query_hash"]
    assert manifest["qrels_hash"]
    assert manifest["evidence_level"] == "FIXTURE_VERIFIED"
    assert summary["recall_at"]["100"] == 1.0
    assert summary["disclaimer"]
    assert result["run_manifest"] == out / "run_manifest.json"
