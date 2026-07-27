"""Typed configuration and provenance tests."""

from pathlib import Path

import pytest

from hcmaic.config import (
    CompetitiveFoundationConfig,
    ProviderSpec,
    artifact_provenance,
    config_hash,
    load_config,
)


def test_config_hash_is_stable_across_field_order():
    a = CompetitiveFoundationConfig(
        dataset_adapter=ProviderSpec(name="btc-adapter", version="v1"),
        ingestion_backend=ProviderSpec(name="ffmpeg", version="2026-07"),
        shot_detector=ProviderSpec(name="uniform", version="fallback"),
        embedding_provider=ProviderSpec(name="mock", version="mock-palette-v1"),
        index_provider=ProviderSpec(name="exact-numpy", version="v1"),
        fusion=ProviderSpec(name="late-fusion", version="v1"),
        reranker=ProviderSpec(name="identity", version="v1"),
        benchmark_inputs=ProviderSpec(name="fixture", version="sample"),
        device="cpu",
        batch_size=16,
        seed=7,
    )
    b = CompetitiveFoundationConfig(
        dataset_adapter=ProviderSpec(name="btc-adapter", version="v1"),
        ingestion_backend=ProviderSpec(name="ffmpeg", version="2026-07"),
        shot_detector=ProviderSpec(name="uniform", version="fallback"),
        embedding_provider=ProviderSpec(name="mock", version="mock-palette-v1"),
        index_provider=ProviderSpec(name="exact-numpy", version="v1"),
        fusion=ProviderSpec(name="late-fusion", version="v1"),
        reranker=ProviderSpec(name="identity", version="v1"),
        benchmark_inputs=ProviderSpec(name="fixture", version="sample"),
        device="cpu",
        batch_size=16,
        seed=7,
    )
    assert config_hash(a) == config_hash(b)


def test_artifact_provenance_records_config_and_hash():
    cfg = CompetitiveFoundationConfig(
        dataset_adapter=ProviderSpec(name="btc-adapter", version="v1"),
        ingestion_backend=ProviderSpec(name="opencv", version="5.0"),
        shot_detector=ProviderSpec(name="uniform", version="fallback"),
        embedding_provider=ProviderSpec(name="mock", version="mock-palette-v1"),
        index_provider=ProviderSpec(name="exact-numpy", version="v1"),
        fusion=ProviderSpec(name="late-fusion", version="v1"),
        reranker=ProviderSpec(name="identity", version="v1"),
        benchmark_inputs=ProviderSpec(name="fixture", version="sample"),
        device="cpu",
        batch_size=8,
        seed=123,
    )
    provenance = artifact_provenance(cfg, code_version="abc1234")
    assert provenance["config_hash"] == config_hash(cfg)
    assert provenance["config"]["device"] == "cpu"
    assert provenance["code_version"] == "abc1234"


def test_load_config_covers_all_competitive_foundation_controls(tmp_path: Path):
    path = tmp_path / "competitive.yaml"
    path.write_text(
        """
dataset_adapter: {name: btc-style, version: v1}
ingestion_backend: {name: opencv, version: "5.0"}
shot_detector: {name: uniform, version: fallback}
sampling_policy: {name: representative-plus-long-shot, version: v1}
modality_extractors:
  - {name: mock-ocr, version: v1}
  - {name: mock-asr, version: v1}
embedding_provider: {name: mock, version: mock-palette-v1}
index_provider: {name: exact-numpy, version: v1}
fusion:
  name: weighted-late-fusion
  version: v1
  params:
    weights: {visual: 1.0, ocr: 0.4}
reranker: {name: identity, version: v1}
benchmark_inputs: {name: proxy-fixture, version: v1}
device: cpu
batch_size: 16
seed: 7
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.sampling_policy.name == "representative-plus-long-shot"
    assert [provider.name for provider in config.modality_extractors] == [
        "mock-ocr",
        "mock-asr",
    ]
    assert config.fusion.to_dict()["params"]["weights"]["ocr"] == 0.4
    assert config_hash(config) == config_hash(load_config(path))


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_size", 0), ("batch_size", -1), ("seed", -1)],
)
def test_load_config_rejects_invalid_runtime_controls(
    tmp_path: Path, field: str, value: int
):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        f"""
dataset_adapter: {{name: fixture}}
ingestion_backend: {{name: opencv}}
shot_detector: {{name: uniform}}
embedding_provider: {{name: mock}}
index_provider: {{name: exact-numpy}}
fusion: {{name: single-stage}}
reranker: {{name: identity}}
benchmark_inputs: {{name: proxy}}
device: cpu
batch_size: {value if field == "batch_size" else 1}
seed: {value if field == "seed" else 0}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=field):
        load_config(path)
