from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from hcmaic.indexing.scale_benchmark import ScaleBenchmarkConfig, run_scale_benchmark


def test_synthetic_scale_benchmark_is_reproducible_and_measures_ann_recall():
    config = ScaleBenchmarkConfig(
        vector_count=300,
        dimension=32,
        query_count=10,
        top_k=10,
        seed=7,
        hnsw_m=16,
        ef_search=64,
    )
    first = run_scale_benchmark(config)
    second = run_scale_benchmark(config)
    assert first["config"] == second["config"]
    assert first["ann_recall_at_k"] == second["ann_recall_at_k"]
    assert first["ann_recall_at_k"] >= 0.9
    assert first["p95_latency_ms"] >= 0
    assert first["evidence_level"] == "SYNTHETIC_SCALE_VERIFIED"
