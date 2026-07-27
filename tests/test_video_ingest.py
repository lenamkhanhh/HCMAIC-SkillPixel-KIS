"""Raw video ingestion tests.

Fixture videos are generated on the fly with OpenCV (MJPG/AVI — codec is
bundled in the pip wheel, no system FFmpeg needed). On machines with FFmpeg
on PATH the ffmpeg backend is exercised instead of/alongside OpenCV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from hcmaic.ingestion import video as vid  # noqa: E402
from hcmaic.ingestion.video import (  # noqa: E402
    IngestError,
    collect_video_files,
    ingest_dataset,
    ingest_video,
    probe_video,
    sanitize_video_id,
)

RED = (220, 40, 40)
BLUE = (40, 70, 220)
GREEN = (40, 180, 60)
YELLOW = (230, 220, 50)


def make_video(
    path: Path,
    colors: list[tuple[int, int, int]],
    fps: float = 2.0,
    size: tuple[int, int] = (64, 48),
) -> Path:
    """Write a solid-color-per-frame AVI (MJPG) deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    assert writer.isOpened(), "OpenCV MJPG writer unavailable"
    for r, g, b in colors:
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:] = (b, g, r)  # BGR
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def color_video(tmp_path: Path) -> Path:
    # 12 frames @ 2 fps = 6s: 3x red, 3x blue, 3x green, 3x yellow
    colors = [RED] * 3 + [BLUE] * 3 + [GREEN] * 3 + [YELLOW] * 3
    return make_video(tmp_path / "videos" / "demo_clip.avi", colors)


# -- probing ---------------------------------------------------------------


def test_probe_metadata(color_video: Path):
    info = probe_video(color_video)
    assert info.video_id == "demo_clip"
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(2.0)
    assert info.frame_count == 12
    assert info.duration_s == pytest.approx(6.0)
    assert info.backend in ("opencv", "ffmpeg")


def test_probe_missing_file(tmp_path: Path):
    with pytest.raises(IngestError, match="not found"):
        probe_video(tmp_path / "nope.mp4")


def test_probe_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "movie.wmv"
    bad.write_bytes(b"x")
    with pytest.raises(IngestError, match="unsupported extension"):
        probe_video(bad)


def test_probe_corrupt_video(tmp_path: Path):
    corrupt = tmp_path / "broken.mp4"
    corrupt.write_bytes(b"this is definitely not a video" * 10)
    with pytest.raises(IngestError):
        probe_video(corrupt)


def test_no_backend_is_actionable(monkeypatch, color_video: Path):
    monkeypatch.setattr(vid, "_ffmpeg_binaries", lambda: None)
    monkeypatch.setattr(vid, "_have_opencv", lambda: False)
    with pytest.raises(IngestError, match="FFmpeg.*--extra video"):
        probe_video(color_video)


# -- video id sanitization -------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("L21_V001", "L21_V001"),
        ("My Video (1)", "My_Video__1"),
        ("clip.final", "clip_final"),
    ],
)
def test_sanitize_video_id(raw: str, expected: str):
    assert sanitize_video_id(raw) == expected


def test_sanitize_video_id_rejects_garbage():
    with pytest.raises(IngestError, match="video_id"):
        sanitize_video_id("###")


# -- single-video ingestion ------------------------------------------------


def test_ingest_writes_dataset_layout(color_video: Path, tmp_path: Path):
    out = tmp_path / "dataset"
    result = ingest_video(color_video, out, interval_s=1.0)
    info = result.info

    assert info.video_id == "demo_clip"
    assert result.n_kept == 4  # one per color after dedup
    assert result.n_duplicates > 0

    frames = sorted((out / "keyframes" / "demo_clip").glob("*.jpg"))
    assert [f.name for f in frames] == [
        f"{i:03d}.jpg" for i in range(1, result.n_kept + 1)
    ]

    with open(out / "keyframe_mapping.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == result.n_kept
    assert set(rows[0]) >= {
        "video_id", "n", "pts_time", "fps", "frame_idx", "width", "height",
    }
    assert rows[0]["timestamp_source"] in {"best_effort_pts", "exact_pts"}
    for row in rows:
        assert row["video_id"] == "demo_clip"
        assert 0.0 <= float(row["pts_time"]) <= info.duration_s
        assert int(row["width"]) == 64
        assert int(row["height"]) == 48
    ns = [int(r["n"]) for r in rows]
    assert ns == sorted(ns) == list(range(1, result.n_kept + 1))

    media = json.loads(
        (out / "media-info" / "demo_clip.json").read_text(encoding="utf-8")
    )
    assert media["length"] == pytest.approx(6.0)
    assert media["width"] == 64 and media["height"] == 48
    # privacy: only the file name, never a local absolute path
    assert media["source_file"] == "demo_clip.avi"
    assert "\\" not in json.dumps(media) and ":/" not in json.dumps(media)


def test_ingest_dedups_identical_frames(tmp_path: Path):
    boring = make_video(tmp_path / "static_clip.avi", [RED] * 10)
    result = ingest_video(boring, tmp_path / "dataset", interval_s=1.0)
    assert result.n_kept == 1  # every later frame is a near-duplicate
    assert result.n_duplicates == result.n_candidates - 1


def test_ingest_refuses_overwrite_without_force(color_video: Path, tmp_path: Path):
    out = tmp_path / "dataset"
    ingest_video(color_video, out, interval_s=1.0)
    with pytest.raises(IngestError, match="--force"):
        ingest_video(color_video, out, interval_s=1.0)


def test_ingest_force_replaces_cleanly(color_video: Path, tmp_path: Path):
    out = tmp_path / "dataset"
    ingest_video(color_video, out, interval_s=1.0)
    result = ingest_video(color_video, out, interval_s=2.0, force=True)
    with open(out / "keyframe_mapping.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # no duplicated rows from the first run
    assert len(rows) == result.n_kept
    assert len({(r["video_id"], r["n"]) for r in rows}) == len(rows)


def test_failed_force_preserves_previous_dataset(
    monkeypatch: pytest.MonkeyPatch, color_video: Path, tmp_path: Path
):
    out = tmp_path / "dataset"
    first = ingest_video(color_video, out, interval_s=1.0)
    mapping_before = (out / "keyframe_mapping.csv").read_bytes()
    frames_before = {
        p.name: p.read_bytes()
        for p in (out / "keyframes" / first.info.video_id).glob("*.jpg")
    }

    def fail_extraction(*args, **kwargs):
        raise IngestError("simulated decoder failure")

    monkeypatch.setattr(vid, "_iter_candidates_opencv", fail_extraction)
    monkeypatch.setattr(vid, "_iter_candidates_ffmpeg", fail_extraction)

    with pytest.raises(IngestError, match="simulated decoder failure"):
        ingest_video(color_video, out, interval_s=2.0, force=True)

    assert (out / "keyframe_mapping.csv").read_bytes() == mapping_before
    assert {
        p.name: p.read_bytes()
        for p in (out / "keyframes" / first.info.video_id).glob("*.jpg")
    } == frames_before


def test_failed_force_commit_restores_previous_dataset(
    monkeypatch: pytest.MonkeyPatch, color_video: Path, tmp_path: Path
):
    out = tmp_path / "dataset"
    first = ingest_video(color_video, out, interval_s=1.0)
    mapping_before = (out / "keyframe_mapping.csv").read_bytes()
    media_before = (out / "media-info" / f"{first.info.video_id}.json").read_bytes()
    frames_before = {
        p.name: p.read_bytes()
        for p in (out / "keyframes" / first.info.video_id).glob("*.jpg")
    }

    original_replace = vid.os.replace

    def fail_on_mapping_replace(src, dst):
        if (
            Path(dst).name == "keyframe_mapping.csv"
            and Path(src).parent.name == "_commit"
        ):
            raise OSError("simulated mapping replace failure")
        return original_replace(src, dst)

    monkeypatch.setattr(vid.os, "replace", fail_on_mapping_replace)

    with pytest.raises(IngestError, match="atomic replacement failed"):
        ingest_video(color_video, out, interval_s=2.0, force=True)

    assert (out / "keyframe_mapping.csv").read_bytes() == mapping_before
    assert (out / "media-info" / f"{first.info.video_id}.json").read_bytes() == media_before
    assert {
        p.name: p.read_bytes()
        for p in (out / "keyframes" / first.info.video_id).glob("*.jpg")
    } == frames_before


def test_parse_ffmpeg_showinfo_preserves_non_zero_pts():
    stderr = """
[Parsed_showinfo_1 @ 000001] n:   0 pts: 22500 pts_time:2.5 pos:0
[Parsed_showinfo_1 @ 000001] n:   1 pts: 40500 pts_time:4.5 pos:123
"""
    assert vid._parse_ffmpeg_showinfo(stderr) == [2.5, 4.5]


def test_parse_ffmpeg_showinfo_rejects_missing_or_invalid_pts():
    with pytest.raises(IngestError, match="timestamp"):
        vid._parse_ffmpeg_showinfo("[Parsed_showinfo_1] n:0 pts:NOPTS")
    with pytest.raises(IngestError, match="negative"):
        vid._parse_ffmpeg_showinfo("[Parsed_showinfo_1] n:0 pts_time:-0.25")


def test_ingest_respects_max_frames(color_video: Path, tmp_path: Path):
    result = ingest_video(
        color_video, tmp_path / "dataset", interval_s=0.5, max_frames=3
    )
    assert result.n_candidates <= 3
    assert result.n_kept <= 3


def test_ingest_bad_parameters(color_video: Path, tmp_path: Path):
    with pytest.raises(IngestError, match="interval_s"):
        ingest_video(color_video, tmp_path / "d", interval_s=0)
    with pytest.raises(IngestError, match="max_frames"):
        ingest_video(color_video, tmp_path / "d", max_frames=0)


@pytest.mark.parametrize("evil", ["../evil", "a/b", "..", "x:y", "dot.dot"])
def test_explicit_video_id_traversal_rejected(color_video: Path, tmp_path: Path, evil: str):
    with pytest.raises(IngestError, match="invalid"):
        ingest_video(color_video, tmp_path / "dataset", video_id=evil)
    # nothing may have been written outside/inside the dataset
    assert not (tmp_path / "dataset" / "keyframes").exists()
    assert not (tmp_path / "evil").exists()


def test_ingest_custom_video_id(color_video: Path, tmp_path: Path):
    result = ingest_video(
        color_video, tmp_path / "dataset", video_id="L99_V042", interval_s=1.0
    )
    assert result.info.video_id == "L99_V042"
    assert (tmp_path / "dataset" / "keyframes" / "L99_V042" / "001.jpg").is_file()


# -- batch ingestion -------------------------------------------------------


def test_collect_video_files(tmp_path: Path, color_video: Path):
    videos_dir = color_video.parent
    make_video(videos_dir / "second.avi", [GREEN] * 4)
    (videos_dir / "notes.txt").write_text("not a video", encoding="utf-8")
    files = collect_video_files(videos_dir)
    assert [f.name for f in files] == ["demo_clip.avi", "second.avi"]
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(IngestError, match="No supported video"):
        collect_video_files(empty_dir)
    with pytest.raises(IngestError, match="does not exist"):
        collect_video_files(tmp_path / "missing_dir")


def test_ingest_dataset_continues_past_failures(tmp_path: Path):
    videos = tmp_path / "videos"
    make_video(videos / "good_one.avi", [RED] * 2 + [BLUE] * 2)
    (videos / "broken.mp4").write_bytes(b"garbage bytes not video")
    out = tmp_path / "dataset"

    results, failures = ingest_dataset(videos, out, interval_s=1.0)

    assert [r.info.video_id for r in results] == ["good_one"]
    assert len(failures) == 1 and failures[0]["file"] == "broken.mp4"
    report = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))
    assert report["n_videos_ok"] == 1
    assert report["n_videos_failed"] == 1
    assert report["failures"][0]["file"] == "broken.mp4"


def test_ingest_dataset_video_id_requires_single_file(tmp_path: Path):
    videos = tmp_path / "videos"
    make_video(videos / "a.avi", [RED] * 2)
    make_video(videos / "b.avi", [BLUE] * 2)
    with pytest.raises(IngestError, match="single file"):
        ingest_dataset(videos, tmp_path / "out", video_id="X")
