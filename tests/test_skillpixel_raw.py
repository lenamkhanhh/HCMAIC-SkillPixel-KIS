"""Raw-video-first SkillPixel ingestion tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hcmaic.skillpixel.raw import (
    RawIngestError,
    ingest_raw_videos,
    validate_raw_dataset,
)


def _write_video(path: Path, frame_count: int = 8, fps: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48)
    )
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (idx * 20 % 255, idx * 30 % 255, idx * 40 % 255)
        writer.write(frame)
    writer.release()
    return path


def test_raw_ingestion_preserves_source_frame_indices_and_manifest(tmp_path: Path):
    source = _write_video(tmp_path / "raw" / "demo.mp4")
    output = tmp_path / "generated"

    report = ingest_raw_videos(source.parent, output, stride_frames=2)

    assert report.n_videos == 1
    assert report.n_frames == 4
    mapping_path = output / "map-keyframes" / "demo.csv"
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [int(row["n"]) for row in rows] == [0, 1, 2, 3]
    assert [int(row["frame_idx"]) for row in rows] == [0, 2, 4, 6]
    assert [int(row["source_frame_idx"]) for row in rows] == [0, 2, 4, 6]
    assert all(row["sampling_policy"] == "uniform_stride_2_v1" for row in rows)
    assert {row["timestamp_ms"] for row in rows} == {"0", "1000", "2000", "3000"}
    assert all(
        (output / "keyframes" / "demo" / f"{int(row['n']):03d}.jpg").is_file()
        for row in rows
    )

    media = json.loads(
        (output / "media-info" / "demo.json").read_text(encoding="utf-8")
    )
    assert media["frame_count"] == 8
    assert media["video_filename"] == "demo.mp4"

    manifest = json.loads(
        (output / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sampling_policy"] == "uniform_stride_2_v1"
    assert manifest["raw_videos"][0]["sha256"]
    assert manifest["raw_videos"][0]["source_path"] == str(source.resolve())
    assert validate_raw_dataset(output).n_frames == 4


def test_raw_ingestion_rejects_invalid_stride_and_out_of_range_mapping(tmp_path: Path):
    source = _write_video(tmp_path / "raw" / "demo.mp4")
    output = tmp_path / "generated"

    with pytest.raises(RawIngestError, match="stride_frames"):
        ingest_raw_videos(source.parent, output, stride_frames=0)

    ingest_raw_videos(source.parent, output, stride_frames=2)
    mapping = output / "map-keyframes" / "demo.csv"
    content = mapping.read_text(encoding="utf-8")
    mapping.write_text(content.replace(",6,", ",99,"), encoding="utf-8")

    with pytest.raises(RawIngestError, match="source_frame_idx"):
        validate_raw_dataset(output)


def test_raw_ingestion_does_not_use_btc_keyframe_layout(tmp_path: Path):
    source = _write_video(tmp_path / "raw" / "video8328.mp4")
    output = tmp_path / "generated"

    ingest_raw_videos(source.parent, output, stride_frames=4)

    assert not (output / "keyframe_mapping.csv").exists()
    assert (output / "map-keyframes" / "video8328.csv").is_file()
    assert not (output / "clip-features-32").exists()

