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


def test_extended_shot_mapping_columns_are_preserved(tmp_path: Path):
    (tmp_path / "keyframe_mapping.csv").write_text(
        (
            "video_id,n,pts_time,fps,frame_idx,shot_id,frame_id,shot_start,"
            "shot_end,width,height,timestamp_source,ingestion_provider,"
            "sampling_policy\n"
            "V1,1,2.5,25,62,V1:shot-001,V1:001,2.0,4.0,1920,1080,"
            "exact_pts,ffmpeg,uniform-v1\n"
        ),
        encoding="utf-8",
    )
    row = load_mapping_rows(tmp_path)[0]
    assert row.shot_id == "V1:shot-001"
    assert row.frame_id == "V1:001"
    assert row.shot_start == pytest.approx(2.0)
    assert row.shot_end == pytest.approx(4.0)
    assert row.timestamp_source == "exact_pts"
    assert row.ingestion_provider == "ffmpeg"
    assert row.sampling_policy == "uniform-v1"


def test_legacy_mapping_gets_explicit_derived_defaults(tmp_path: Path):
    (tmp_path / "keyframe_mapping.csv").write_text(
        "video_id,n,pts_time,fps,frame_idx\nV1,1,2.5,25,62\n",
        encoding="utf-8",
    )
    row = load_mapping_rows(tmp_path)[0]
    assert row.shot_id is None
    assert row.frame_id == "V1:001"
    assert row.timestamp_source == "legacy_mapping"


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
