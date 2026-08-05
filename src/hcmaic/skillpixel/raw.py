"""Deterministic raw-video ingestion for the SkillPixel submission slice.

This module deliberately owns its generated dataset contract.  The source
video is decoded in order and every saved image keeps the original decoded
frame number in ``source_frame_idx``.  BTC keyframes, feature files and
mapping files are never read here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hcmaic.ingestion.video import SUPPORTED_EXTENSIONS

RAW_MANIFEST_NAME = "dataset_manifest.json"
RAW_SCHEMA_VERSION = "skillpixel-raw-v1"
MAPPING_COLUMNS = (
    "n",
    "pts_time",
    "fps",
    "frame_idx",
    "source_frame_idx",
    "timestamp_ms",
    "sampling_policy",
    "timestamp_source",
    "video_filename",
    "frame_count",
    "width",
    "height",
)
DEFAULT_STRIDE_FRAMES = 10
_JPEG_QUALITY = 92


class RawIngestError(RuntimeError):
    """Raised when raw ingestion or generated-data validation fails."""


@dataclass(frozen=True)
class RawVideoInfo:
    video_id: str
    video_filename: str
    source_path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    sha256: str
    timestamp_source: str


@dataclass(frozen=True)
class RawIngestReport:
    videos: tuple[RawVideoInfo, ...]
    sampling_policy: str
    frame_count: int

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    @property
    def n_frames(self) -> int:
        return self.frame_count


@dataclass(frozen=True)
class RawDatasetStats:
    n_videos: int
    n_frames: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_files(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise RawIngestError(f"Raw video input does not exist: {input_path}")
    if not files:
        raise RawIngestError(
            f"No raw videos with extensions {sorted(SUPPORTED_EXTENSIONS)} found in {input_path}"
        )
    return files


def _video_id(path: Path) -> str:
    raw = path.stem
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in raw)
    cleaned = cleaned.strip("_-")
    if not cleaned:
        raise RawIngestError(f"Cannot derive a video_id from raw video {path.name!r}")
    return cleaned


def _probe(path: Path) -> tuple[int, int, float, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - project extra is installed in tests
        raise RawIngestError(
            "Raw video ingestion requires OpenCV. Install with: uv sync --extra video"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RawIngestError(f"{path.name}: OpenCV cannot open the raw video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RawIngestError(
            f"{path.name}: invalid OpenCV metadata width={width}, height={height}, fps={fps}"
        )
    return width, height, fps, max(frame_count, 0)


def _write_mapping(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _extract_one(source: Path, output_root: Path, stride_frames: int) -> RawVideoInfo:
    import cv2

    width, height, fps, probed_frame_count = _probe(source)
    video_id = _video_id(source)
    frames_dir = output_root / "keyframes" / video_id
    mapping_path = output_root / "map-keyframes" / f"{video_id}.csv"
    media_path = output_root / "media-info" / f"{video_id}.json"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    rows: list[dict[str, Any]] = []
    decoded_count = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            source_frame_idx = decoded_count
            if source_frame_idx % stride_frames == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                n = len(rows)
                pts_time = source_frame_idx / fps
                image_path = frames_dir / f"{n:03d}.jpg"
                Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).save(
                    image_path, quality=_JPEG_QUALITY
                )
                rows.append(
                    {
                        "n": n,
                        "pts_time": round(pts_time, 6),
                        "fps": round(fps, 6),
                        "frame_idx": source_frame_idx,
                        "source_frame_idx": source_frame_idx,
                        "timestamp_ms": round(pts_time * 1000),
                        "sampling_policy": f"uniform_stride_{stride_frames}_v1",
                        "timestamp_source": "cfr_frame_index",
                        "video_filename": source.name,
                        "frame_count": probed_frame_count,
                        "width": width,
                        "height": height,
                    }
                )
            decoded_count += 1
    finally:
        capture.release()

    if decoded_count == 0 or not rows:
        raise RawIngestError(f"{source.name}: no decodable frames were produced")
    frame_count = decoded_count
    if probed_frame_count and probed_frame_count != decoded_count:
        frame_count = decoded_count
        for row in rows:
            row["frame_count"] = frame_count

    _write_mapping(mapping_path, rows)
    info = RawVideoInfo(
        video_id=video_id,
        video_filename=source.name,
        source_path=source.resolve(),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps,
        sha256=_sha256(source),
        timestamp_source="cfr_frame_index",
    )
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_text(
        json.dumps(
            {
                "video_id": info.video_id,
                "video_filename": info.video_filename,
                "source_file": info.video_filename,
                "source_video_path": str(info.source_path),
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frame_count": info.frame_count,
                "length": round(info.duration_s, 6),
                "duration_seconds": round(info.duration_s, 6),
                "timestamp_source": info.timestamp_source,
                "sampling_policy": f"uniform_stride_{stride_frames}_v1",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return info


def _generated_file_hashes(root: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != RAW_MANIFEST_NAME
        and path.suffix.lower() in {".csv", ".json", ".jpg", ".jpeg", ".png"}
    )
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def _write_manifest(root: Path, videos: list[RawVideoInfo], stride_frames: int) -> None:
    policy = f"uniform_stride_{stride_frames}_v1"
    raw_videos = [
        {
            "video_id": item.video_id,
            "video_filename": item.video_filename,
            "source_path": str(item.source_path),
            "sha256": item.sha256,
            "frame_count": item.frame_count,
            "fps": item.fps,
            "width": item.width,
            "height": item.height,
            "duration_seconds": item.duration_s,
            "timestamp_source": item.timestamp_source,
        }
        for item in videos
    ]
    payload: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "sampling_policy": policy,
        "stride_frames": stride_frames,
        "raw_videos": raw_videos,
        "n_videos": len(videos),
        "n_frames": sum(
            len(list((root / "map-keyframes" / f"{item.video_id}.csv").open())) - 1
            for item in videos
        ),
        "files": _generated_file_hashes(root),
    }
    payload["dataset_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    (root / RAW_MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ingest_raw_videos(
    input_path: Path,
    output_root: Path,
    *,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    force: bool = False,
) -> RawIngestReport:
    """Decode raw videos in source order and write a versioned raw dataset."""
    if stride_frames < 1:
        raise RawIngestError(f"stride_frames must be >= 1, got {stride_frames}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    infos: list[RawVideoInfo] = []
    for source in _video_files(Path(input_path)):
        video_id = _video_id(source)
        frames_dir = output_root / "keyframes" / video_id
        if frames_dir.exists() and any(frames_dir.iterdir()):
            if not force:
                raise RawIngestError(
                    f"{video_id}: generated frames already exist in {output_root}; "
                    "use force=True/--force to replace them"
                )
            shutil.rmtree(frames_dir)
            (output_root / "map-keyframes" / f"{video_id}.csv").unlink(missing_ok=True)
            (output_root / "media-info" / f"{video_id}.json").unlink(missing_ok=True)
        infos.append(_extract_one(source, output_root, stride_frames))
    _write_manifest(output_root, infos, stride_frames)
    return RawIngestReport(
        tuple(infos),
        f"uniform_stride_{stride_frames}_v1",
        sum(
            len(list((output_root / "map-keyframes" / f"{item.video_id}.csv").open())) - 1
            for item in infos
        ),
    )


def validate_raw_dataset(root: Path) -> RawDatasetStats:
    """Fail closed on source-frame, image, count, or manifest mismatches."""
    root = Path(root)
    manifest_path = root / RAW_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RawIngestError(f"Missing {RAW_MANIFEST_NAME} in {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RawIngestError(f"Invalid {manifest_path}: {exc}") from exc
    raw_videos = manifest.get("raw_videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise RawIngestError("dataset_manifest.json has no raw_videos")

    total_frames = 0
    for video in raw_videos:
        video_id = str(video.get("video_id", ""))
        frame_count = int(video.get("frame_count", -1))
        if not video_id or frame_count < 1:
            raise RawIngestError(f"Invalid frame_count/video_id for {video_id!r}")
        mapping_path = root / "map-keyframes" / f"{video_id}.csv"
        if not mapping_path.is_file():
            raise RawIngestError(f"Missing mapping for {video_id}")
        with mapping_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing = [column for column in MAPPING_COLUMNS[:7] if column not in fields]
            if missing:
                raise RawIngestError(f"{mapping_path.name}: missing required columns {missing}")
            seen_source: set[int] = set()
            rows = list(reader)
        for row in rows:
            try:
                n = int(row["n"])
                source_idx = int(row["source_frame_idx"])
                frame_idx = int(row["frame_idx"])
                timestamp_ms = int(row["timestamp_ms"])
                pts_time = float(row["pts_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RawIngestError(f"{mapping_path.name}: invalid mapping row {row}") from exc
            if source_idx in seen_source:
                raise RawIngestError(f"duplicate source_frame_idx={source_idx} for {video_id}")
            seen_source.add(source_idx)
            if source_idx < 0 or source_idx >= frame_count:
                raise RawIngestError(
                    f"source_frame_idx={source_idx} out of range for {video_id} [0, {frame_count})"
                )
            if frame_idx != source_idx:
                raise RawIngestError(
                    f"frame_idx/source_frame_idx mismatch for {video_id} n={n}: "
                    f"{frame_idx} != {source_idx}"
                )
            if timestamp_ms != round(pts_time * 1000):
                raise RawIngestError(f"timestamp mismatch for {video_id} n={n}")
            image = root / "keyframes" / video_id / f"{n:03d}.jpg"
            if not image.is_file():
                raise RawIngestError(f"missing image for {video_id} n={n}: {image}")
            try:
                with Image.open(image) as handle:
                    handle.verify()
            except (OSError, SyntaxError) as exc:
                raise RawIngestError(f"unreadable image for {video_id} n={n}") from exc
        if not rows:
            raise RawIngestError(f"empty mapping for {video_id}")
        total_frames += len(rows)

    expected = int(manifest.get("n_frames", -1))
    if expected != total_frames:
        raise RawIngestError(
            f"dataset manifest frame count {expected} != mapping count {total_frames}"
        )
    return RawDatasetStats(n_videos=len(raw_videos), n_frames=total_frames)
