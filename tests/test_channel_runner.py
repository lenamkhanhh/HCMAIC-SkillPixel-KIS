import json
from pathlib import Path

import cv2
import numpy as np
from hcmaic.retrieval.channel_runner import (
    build_object_channel,
    build_ocr_channel,
)

from hcmaic.retrieval.object_retrieval import ObjectRecord
from hcmaic.retrieval.ocr_bm25 import load_ocr_artifact
from hcmaic.skillpixel.raw import ingest_raw_videos


def _write_video(path: Path, frame_count: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 24))
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
    writer.release()
    return path


class _OCRProvider:
    name = "test-real-ocr"
    revision = "ocr-revision"

    def infer_image(self, image_path: Path):
        return [(image_path.stem, 0.9, ())]

    def manifest_metadata(self, *, batch_size: int):
        return {
            "model_source_url": "https://example.invalid/ocr",
            "weights_sha256": "ocr-sha",
            "runtime": {"device": "cpu", "batch_size": batch_size},
        }


class _ObjectProvider:
    name = "test-real-object"
    revision = "object-revision"

    def infer_images(self, image_paths, *, batch_size: int):
        return [[("person", 0.8, (1.0, 2.0, 3.0, 4.0))] for _ in image_paths]

    def manifest_metadata(self, *, batch_size: int):
        return {
            "model_source_url": "https://example.invalid/object",
            "weights_sha256": "object-sha",
            "runtime": {"device": "cpu", "batch_size": batch_size},
        }


def test_ocr_runner_maps_raw_catalog_and_builds_bm25_artifact(tmp_path: Path) -> None:
    _write_video(tmp_path / "videos" / "demo.avi")
    raw_root = tmp_path / "raw"
    ingest_raw_videos(tmp_path / "videos", raw_root, stride_frames=2)

    result = build_ocr_channel(raw_root, tmp_path / "channels" / "ocr", _OCRProvider())

    assert result.status == "built"
    assert result.n_input_frames == 3
    assert result.n_records == 3
    artifact = load_ocr_artifact(result.artifact_dir, dataset_manifest_hash=result.dataset_hash)
    assert [record.source_frame_idx for record in artifact.records] == [0, 2, 4]
    assert [record.frame_uid for record in artifact.records] == [
        "demo:000",
        "demo:001",
        "demo:002",
    ]
    assert artifact.manifest["model_source_url"] == "https://example.invalid/ocr"
    assert artifact.manifest["weights_sha256"] == "ocr-sha"


def test_object_runner_deduplicates_labels_and_preserves_source_mapping(tmp_path: Path) -> None:
    _write_video(tmp_path / "videos" / "demo.avi")
    raw_root = tmp_path / "raw"
    ingest_raw_videos(tmp_path / "videos", raw_root, stride_frames=2)

    result = build_object_channel(
        raw_root,
        tmp_path / "channels" / "object",
        _ObjectProvider(),
        batch_size=2,
    )

    assert result.status == "built"
    assert result.n_input_frames == 3
    assert result.n_records == 3
    records_path = result.artifact_dir / "objects.jsonl"
    records = [
        ObjectRecord(**json.loads(line))
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(record.source_frame_idx in {0, 2, 4} for record in records)
    assert all(record.frame_uid.startswith("demo:") for record in records)
