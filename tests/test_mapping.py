"""Mapping CSV parsing tests (both supported layouts)."""

from pathlib import Path

import pytest

from hcmaic.ingestion.mapping import (
    MappingError,
    find_keyframe_image,
    load_mapping_rows,
)


def test_single_file_layout(sample_root: Path):
    rows = load_mapping_rows(sample_root)
    assert len(rows) == 12
    assert {r.video_id for r in rows} == {
        "L01_V001", "L01_V002", "L01_V003", "L01_V004", "L01_V005",
    }
    first = next(r for r in rows if r.video_id == "L01_V001" and r.n == 1)
    assert first.pts_time == 1.0
    assert first.fps == 25.0
    assert first.frame_idx == 25


def test_per_video_layout(tmp_path: Path):
    per_video = tmp_path / "map-keyframes"
    per_video.mkdir()
    (per_video / "VID_A.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.5,30.0,15\n2,2.0,30.0,60\n",
        encoding="utf-8",
    )
    (per_video / "VID_B.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,1.0,25.0,25\n",
        encoding="utf-8",
    )
    rows = load_mapping_rows(tmp_path)
    assert [(r.video_id, r.n) for r in rows] == [
        ("VID_A", 1), ("VID_A", 2), ("VID_B", 1),
    ]


def test_missing_columns_is_actionable(tmp_path: Path):
    (tmp_path / "keyframe_mapping.csv").write_text(
        "video_id,n,frame_idx\nV1,1,25\n", encoding="utf-8"
    )
    with pytest.raises(MappingError, match="missing required column"):
        load_mapping_rows(tmp_path)


def test_unparsable_row(tmp_path: Path):
    (tmp_path / "keyframe_mapping.csv").write_text(
        "video_id,n,pts_time,fps,frame_idx\nV1,one,1.0,25.0,25\n",
        encoding="utf-8",
    )
    with pytest.raises(MappingError, match="unparsable row"):
        load_mapping_rows(tmp_path)


def test_no_mapping_at_all(tmp_path: Path):
    with pytest.raises(MappingError, match="No keyframe mapping found"):
        load_mapping_rows(tmp_path)


def test_empty_per_video_dir(tmp_path: Path):
    (tmp_path / "map-keyframes").mkdir()
    with pytest.raises(MappingError, match="no .csv files"):
        load_mapping_rows(tmp_path)


def test_find_keyframe_image_probes_extensions(tmp_path: Path):
    video_dir = tmp_path / "keyframes" / "V1"
    video_dir.mkdir(parents=True)
    (video_dir / "001.png").write_bytes(b"x")
    found = find_keyframe_image(tmp_path, "V1", 1)
    assert found is not None and found.name == "001.png"
    assert find_keyframe_image(tmp_path, "V1", 2) is None


def test_find_keyframe_image_unpadded_name(tmp_path: Path):
    video_dir = tmp_path / "keyframes" / "V1"
    video_dir.mkdir(parents=True)
    (video_dir / "7.jpg").write_bytes(b"x")
    found = find_keyframe_image(tmp_path, "V1", 7)
    assert found is not None and found.name == "7.jpg"
