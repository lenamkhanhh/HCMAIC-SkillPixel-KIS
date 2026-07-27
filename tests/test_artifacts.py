"""Index artifact build/load and consistency-gate tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider
from hcmaic.indexing.artifacts import (
    ArtifactError,
    build_index_artifacts,
    load_index_artifacts,
)
from hcmaic.ingestion.catalog import build_catalog


@pytest.fixture()
def artifacts_copy(built_artifacts_dir: Path, tmp_path: Path) -> Path:
    import shutil

    dest = tmp_path / "artifacts"
    shutil.copytree(built_artifacts_dir, dest)
    return dest


def test_build_writes_all_artifacts(built_artifacts_dir: Path):
    for name in (
        "catalog.jsonl",
        "dataset_manifest.json",
        "embeddings.npy",
        "id_map.json",
        "index_manifest.json",
    ):
        assert (built_artifacts_dir / name).is_file(), name


def test_index_manifest_contents(built_artifacts_dir: Path):
    manifest = json.loads(
        (built_artifacts_dir / "index_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "hcmaic-index-v1"
    assert manifest["dimension"] == 64
    assert manifest["n_frames"] == 12
    assert manifest["embedding"]["provider"] == "mock"
    assert manifest["embedding"]["version"] == "mock-palette-v1"
    assert manifest["normalization"] == "l2"
    assert manifest["dataset_manifest_hash"]
    assert manifest["created_at"]
    assert "index_version" in manifest
    assert manifest["config"]["embedding_provider"]["name"] == "mock"
    assert manifest["config_hash"]
    assert manifest["code_version"]


def test_load_round_trip(built_artifacts_dir: Path):
    artifacts = load_index_artifacts(built_artifacts_dir)
    assert artifacts.embeddings.shape == (12, 64)
    assert artifacts.id_map == [r.frame_id for r in artifacts.catalog]
    assert artifacts.catalog[0].embedding_version == "mock-palette-v1"
    norms = np.linalg.norm(artifacts.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_missing_artifact_is_actionable(artifacts_copy: Path):
    (artifacts_copy / "id_map.json").unlink()
    with pytest.raises(ArtifactError, match="Missing artifact id_map.json"):
        load_index_artifacts(artifacts_copy)


def test_row_count_mismatch_rejected(artifacts_copy: Path):
    embeddings = np.load(artifacts_copy / "embeddings.npy")
    np.save(artifacts_copy / "embeddings.npy", embeddings[:-1])
    with pytest.raises(ArtifactError, match="row mismatch"):
        load_index_artifacts(artifacts_copy)


def test_dimension_mismatch_rejected(artifacts_copy: Path):
    manifest_path = artifacts_copy / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dimension"] = 1280  # the upstream bug class
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="Dimension mismatch"):
        load_index_artifacts(artifacts_copy)


def test_dataset_manifest_hash_mismatch_rejected(artifacts_copy: Path):
    manifest_path = artifacts_copy / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="Dataset manifest hash mismatch"):
        load_index_artifacts(artifacts_copy)


def test_manifest_frame_count_mismatch_rejected(artifacts_copy: Path):
    manifest_path = artifacts_copy / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["n_frames"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="Frame count mismatch"):
        load_index_artifacts(artifacts_copy)


def test_non_finite_embeddings_rejected(artifacts_copy: Path):
    embeddings = np.load(artifacts_copy / "embeddings.npy")
    embeddings[0, 0] = np.nan
    np.save(artifacts_copy / "embeddings.npy", embeddings)
    with pytest.raises(ArtifactError, match="non-finite"):
        load_index_artifacts(artifacts_copy)


def test_non_normalized_embeddings_rejected(artifacts_copy: Path):
    embeddings = np.load(artifacts_copy / "embeddings.npy")
    embeddings[0] *= 2.0
    np.save(artifacts_copy / "embeddings.npy", embeddings)
    with pytest.raises(ArtifactError, match="L2-normalized"):
        load_index_artifacts(artifacts_copy)


def test_id_order_mismatch_rejected(artifacts_copy: Path):
    id_map_path = artifacts_copy / "id_map.json"
    ids = json.loads(id_map_path.read_text(encoding="utf-8"))
    ids[0], ids[1] = ids[1], ids[0]
    id_map_path.write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(ArtifactError, match="id_map.json order"):
        load_index_artifacts(artifacts_copy)


def test_empty_catalog_rejected(tmp_path: Path, sample_root: Path):
    with pytest.raises(ArtifactError, match="empty"):
        build_index_artifacts(
            sample_root, [], DeterministicMockEmbeddingProvider(), tmp_path / "out"
        )


def test_rebuild_is_deterministic(sample_root: Path, tmp_path: Path):
    catalog_a = build_catalog(sample_root)
    catalog_b = build_catalog(sample_root)
    out_a = build_index_artifacts(
        sample_root, catalog_a, DeterministicMockEmbeddingProvider(), tmp_path / "a"
    )
    out_b = build_index_artifacts(
        sample_root, catalog_b, DeterministicMockEmbeddingProvider(), tmp_path / "b"
    )
    emb_a = np.load(out_a / "embeddings.npy")
    emb_b = np.load(out_b / "embeddings.npy")
    np.testing.assert_array_equal(emb_a, emb_b)
    assert (
        (out_a / "id_map.json").read_text(encoding="utf-8")
        == (out_b / "id_map.json").read_text(encoding="utf-8")
    )
