"""Dataset validator tests: every mandated failure mode."""

import json
from pathlib import Path

from hcmaic.ingestion.validator import validate_dataset, write_validation_report


def _codes(issues):
    return [i.code for i in issues]


def test_sample_dataset_is_valid(sample_root: Path):
    report = validate_dataset(sample_root)
    assert report.ok, [e.message for e in report.errors]
    assert report.n_videos == 5
    assert report.n_frames == 12
    # missing optional metadata is a warning, never an error
    assert "missing-metadata" in _codes(report.warnings)


def test_missing_image(dataset_copy: Path):
    (dataset_copy / "keyframes" / "L01_V001" / "001.jpg").unlink()
    report = validate_dataset(dataset_copy)
    assert not report.ok
    assert "missing-image" in _codes(report.errors)
    msg = next(e for e in report.errors if e.code == "missing-image").message
    assert "L01_V001" in msg and "001" in msg


def test_duplicate_frame_id(dataset_copy: Path):
    mapping = dataset_copy / "keyframe_mapping.csv"
    lines = mapping.read_text(encoding="utf-8").rstrip().splitlines()
    lines.append(lines[1])  # duplicate the first data row
    mapping.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_dataset(dataset_copy)
    assert "duplicate-frame" in _codes(report.errors)


def test_invalid_video_id_and_path_escape(dataset_copy: Path):
    mapping = dataset_copy / "keyframe_mapping.csv"
    content = mapping.read_text(encoding="utf-8")
    content += "../evil,1,1.0,25.0,25\n"
    mapping.write_text(content, encoding="utf-8")
    report = validate_dataset(dataset_copy)
    issue = next(e for e in report.errors if e.code == "invalid-video-id")
    assert "../evil" in issue.message


def test_negative_timestamp(dataset_copy: Path):
    mapping = dataset_copy / "keyframe_mapping.csv"
    content = mapping.read_text(encoding="utf-8")
    content += "L01_V001,9,-2.0,25.0,-50\n"
    mapping.write_text(content, encoding="utf-8")
    report = validate_dataset(dataset_copy)
    assert "negative-timestamp" in _codes(report.errors)


def test_non_finite_timing_values(dataset_copy: Path):
    mapping = dataset_copy / "keyframe_mapping.csv"
    content = mapping.read_text(encoding="utf-8")
    content += "L01_V001,9,nan,25.0,225\n"
    content += "L01_V001,10,2.0,inf,250\n"
    mapping.write_text(content, encoding="utf-8")
    report = validate_dataset(dataset_copy)
    assert "non-finite-timing" in _codes(report.errors)


def test_timestamp_beyond_duration(dataset_copy: Path):
    # L01_V001 declares length=60 in media-info; add a frame at 100s with a
    # real image so only the duration check fires.
    src = dataset_copy / "keyframes" / "L01_V001" / "001.jpg"
    (dataset_copy / "keyframes" / "L01_V001" / "009.jpg").write_bytes(src.read_bytes())
    mapping = dataset_copy / "keyframe_mapping.csv"
    content = mapping.read_text(encoding="utf-8")
    content += "L01_V001,9,100.0,25.0,2500\n"
    mapping.write_text(content, encoding="utf-8")
    report = validate_dataset(dataset_copy)
    issue = next(e for e in report.errors if e.code == "timestamp-out-of-range")
    assert "100.0" in issue.message and "60" in issue.message


def test_unreadable_image(dataset_copy: Path):
    target = dataset_copy / "keyframes" / "L01_V002" / "001.jpg"
    target.write_bytes(b"this is not a jpeg")
    report = validate_dataset(dataset_copy)
    issue = next(e for e in report.errors if e.code == "unreadable-image")
    assert "L01_V002" in issue.message


def test_orphan_image_warning(dataset_copy: Path):
    src = dataset_copy / "keyframes" / "L01_V001" / "001.jpg"
    (dataset_copy / "keyframes" / "L01_V001" / "099.jpg").write_bytes(src.read_bytes())
    report = validate_dataset(dataset_copy)
    assert report.ok  # orphan is a warning, not an error
    assert "orphan-image" in _codes(report.warnings)


def test_broken_mapping_is_single_actionable_error(tmp_path: Path):
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert _codes(report.errors) == ["mapping-columns"]


def test_skip_image_check(dataset_copy: Path):
    target = dataset_copy / "keyframes" / "L01_V002" / "001.jpg"
    target.write_bytes(b"junk")
    report = validate_dataset(dataset_copy, check_images=False)
    assert "unreadable-image" not in _codes(report.errors)


def test_write_validation_report(dataset_copy: Path, tmp_path: Path):
    report = validate_dataset(dataset_copy)
    out = tmp_path / "out" / "validation_report.json"
    write_validation_report(report, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_frames"] == 12
    assert data["errors"] == []
