"""Safe Kaggle packaging contract tests for SkillPixel KIS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmaic.benchmark.kaggle import (
    KagglePackageConfig,
    KagglePackageError,
    build_kaggle_package,
    validate_kaggle_package,
)


def _config(tmp_path: Path, output: Path) -> KagglePackageConfig:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dataset_manifest.json").write_text(
        json.dumps({"dataset_hash": "raw-hash", "n_videos": 1, "n_frames": 2}),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.csv"
    questions.write_text("query_id,task,text,query_image\nT1,TKIS,hello,\n", encoding="utf-8")
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("video,frame_count\ndemo.mp4,2\n", encoding="utf-8")
    return KagglePackageConfig(
        output_dir=output,
        raw_input=raw,
        questions_path=questions,
        corpus_path=corpus,
        index_dir=None,
    )


def test_build_and_validate_kaggle_package_is_metadata_only(tmp_path: Path):
    package_dir = tmp_path / "package"
    outputs = build_kaggle_package(_config(tmp_path, package_dir))

    report = validate_kaggle_package(package_dir)
    assert report["valid"] is True
    assert outputs["manifest"].is_file()
    assert not list(package_dir.rglob("*.mp4"))
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["contains_raw_videos"] is False
    assert manifest["contains_model_weights"] is False
    assert manifest["contains_tokens"] is False
    assert manifest["contains_generated_index"] is False


def test_kaggle_package_rejects_large_or_forbidden_files(tmp_path: Path):
    package_dir = tmp_path / "package"
    build_kaggle_package(_config(tmp_path, package_dir))
    (package_dir / "weights.safetensors").write_bytes(b"x")

    with pytest.raises(KagglePackageError, match="forbidden"):
        validate_kaggle_package(package_dir)
