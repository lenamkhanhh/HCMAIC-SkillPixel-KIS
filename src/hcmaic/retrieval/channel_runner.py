"""Build versioned raw-derived OCR, object and optional ASR artifacts."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmaic.contracts.models import FrameRecord
from hcmaic.ingestion.catalog import load_catalog
from hcmaic.retrieval.asr import ASRRecord, write_asr_artifact
from hcmaic.retrieval.object_retrieval import ObjectRecord, write_object_artifact
from hcmaic.retrieval.ocr_bm25 import (
    OCRRecord,
    load_ocr_artifact,
    write_ocr_artifact,
)
from hcmaic.retrieval.real_channels import (
    ASRObservation,
    ObjectObservation,
    OCRObservation,
    asr_records_for_video,
    object_records_for_frame,
    ocr_record_for_frame,
    probe_audio_stream,
    sha256_path,
)


@dataclass(frozen=True)
class ChannelRunResult:
    """Stable CLI-facing result for a channel build attempt."""

    channel: str
    status: str
    artifact_dir: Path | None
    manifest_path: Path
    dataset_hash: str
    n_input_frames: int
    n_records: int
    details: dict[str, Any]


def _read_dataset_hash(raw_root: Path) -> str:
    manifest_path = Path(raw_root) / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = str(manifest.get("dataset_hash", "")).strip()
    if not dataset_hash:
        raise ValueError(f"raw dataset manifest has no dataset_hash: {manifest_path}")
    return dataset_hash


def load_raw_catalog_frames(raw_root: Path) -> list[tuple[FrameRecord, Path]]:
    """Load catalog rows and assert every provider input is a generated raw image."""
    raw_root = Path(raw_root).resolve()
    catalog_path = raw_root / "catalog.jsonl"
    frames = load_catalog(catalog_path)
    frames.sort(
        key=lambda frame: (
            frame.video_id,
            frame.source_frame_idx if frame.source_frame_idx is not None else frame.frame_idx,
            frame.frame_id,
        )
    )
    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("raw catalog has duplicate frame_id values")
    resolved: list[tuple[FrameRecord, Path]] = []
    for frame in frames:
        image_path = (raw_root / frame.image_path).resolve()
        try:
            image_path.relative_to(raw_root)
        except ValueError as exc:
            raise ValueError(f"raw catalog image escapes raw root: {frame.image_path}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        resolved.append((frame, image_path))
    if not resolved:
        raise ValueError(f"raw catalog is empty: {catalog_path}")
    return resolved


def _provider_metadata(provider: Any, *, batch_size: int) -> dict[str, Any]:
    metadata_method = getattr(provider, "manifest_metadata", None)
    if not callable(metadata_method):
        raise ValueError("real channel provider must expose manifest_metadata(batch_size=...)")
    metadata = metadata_method(batch_size=batch_size)
    if not isinstance(metadata, dict):
        raise ValueError("provider manifest_metadata must return a mapping")
    reserved = {
        "format",
        "records",
        "records_sha256",
        "n_records",
        "provider",
        "revision",
        "dataset_manifest_hash",
        "raw_video_source",
        "btc_artifacts_used",
        "provider_execution",
        "evidence_level",
        "quality_status",
    }
    overlap = sorted(reserved.intersection(metadata))
    if overlap:
        raise ValueError(f"provider metadata cannot override reserved fields: {overlap}")
    return dict(metadata)


def _provider_identity(provider: Any) -> tuple[str, str]:
    name = str(getattr(provider, "name", "")).strip()
    revision = str(getattr(provider, "revision", "")).strip()
    if not name or not revision or "mock" in name.casefold():
        raise ValueError("channel provider must expose non-mock name and revision")
    return name, revision


def _base_manifest_extra(
    *,
    raw_root: Path,
    channel: str,
    n_input_frames: int,
    dataset_hash: str,
    provider: Any,
    batch_size: int,
) -> dict[str, Any]:
    metadata = _provider_metadata(provider, batch_size=batch_size)
    metadata.update(
        {
            "channel": channel,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "raw_catalog_sha256": sha256_path(Path(raw_root) / "catalog.jsonl"),
            "n_input_frames": n_input_frames,
            "mapping_schema": (
                "raw_catalog.frame_uid -> video_id -> source_frame_idx -> timestamp_ms"
            ),
            "raw_root": str(Path(raw_root).resolve()),
        }
    )
    return metadata


def _stage_manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / "channel_stage_manifest.json"


def _write_stage_manifest(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _stage_manifest_path(output_dir)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _ocr_checkpoint_paths(output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    return (
        output_dir.with_name(f"{output_dir.name}.checkpoint.jsonl"),
        output_dir.with_name(f"{output_dir.name}.checkpoint.json"),
    )


def _write_ocr_checkpoint_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_ocr_checkpoint(
    output_dir: Path,
    *,
    dataset_hash: str,
    raw_catalog_sha256: str,
    n_input_frames: int,
    provider: str,
    revision: str,
    valid_frame_uids: set[str],
) -> tuple[set[str], list[OCRRecord]]:
    records_path, manifest_path = _ocr_checkpoint_paths(output_dir)
    if not records_path.exists() and not manifest_path.exists():
        return set(), []
    if not records_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            "OCR checkpoint is incomplete; both checkpoint JSONL and manifest are required"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read OCR checkpoint manifest: {manifest_path}") from exc
    expected = {
        "dataset_hash": dataset_hash,
        "raw_catalog_sha256": raw_catalog_sha256,
        "n_input_frames": n_input_frames,
        "provider": provider,
        "revision": revision,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"OCR checkpoint identity mismatch: {mismatches}")

    processed_frame_uids: set[str] = set()
    records: list[OCRRecord] = []
    try:
        lines = records_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read OCR checkpoint records: {records_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            # A hard process stop can leave only the final JSONL line partial.
            if line_number == len(lines):
                break
            raise ValueError(
                f"invalid OCR checkpoint record at line {line_number}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"OCR checkpoint record {line_number} is not an object")
        frame_uid = str(payload.get("frame_uid", "")).strip()
        if not frame_uid or frame_uid not in valid_frame_uids:
            raise ValueError(f"OCR checkpoint has unknown frame_uid: {frame_uid!r}")
        if frame_uid in processed_frame_uids:
            raise ValueError(f"OCR checkpoint has duplicate frame_uid: {frame_uid}")
        processed_frame_uids.add(frame_uid)
        record_payload = payload.get("record")
        if record_payload is None:
            continue
        if not isinstance(record_payload, dict):
            raise ValueError(f"OCR checkpoint record payload is invalid: {frame_uid}")
        record = OCRRecord(**record_payload)
        if record.frame_uid != frame_uid:
            raise ValueError(f"OCR checkpoint frame identity mismatch: {frame_uid}")
        if record.provider != provider or record.revision != revision:
            raise ValueError(f"OCR checkpoint provider identity mismatch: {frame_uid}")
        records.append(record)
    if len(processed_frame_uids) > n_input_frames:
        raise ValueError("OCR checkpoint contains more frames than the raw catalog")
    return processed_frame_uids, records


def _append_ocr_checkpoint(
    handle: Any,
    *,
    frame_uid: str,
    record: OCRRecord | None,
) -> None:
    payload = {
        "frame_uid": frame_uid,
        "record": record.to_dict() if record is not None else None,
    }
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _remove_ocr_checkpoint(output_dir: Path) -> None:
    records_path, manifest_path = _ocr_checkpoint_paths(output_dir)
    for path in (records_path, manifest_path):
        if path.exists():
            path.unlink()


def _resume_result(
    output_dir: Path,
    *,
    channel: str,
    dataset_hash: str,
    n_input_frames: int,
    artifact_name: str,
    manifest_name: str,
    load_artifact: Any,
) -> ChannelRunResult | None:
    output_dir = Path(output_dir)
    if not (output_dir / artifact_name).is_file() or not (output_dir / manifest_name).is_file():
        return None
    artifact = load_artifact(output_dir, dataset_manifest_hash=dataset_hash)
    records = artifact.records
    return ChannelRunResult(
        channel=channel,
        status="resumed",
        artifact_dir=output_dir,
        manifest_path=_stage_manifest_path(output_dir),
        dataset_hash=dataset_hash,
        n_input_frames=n_input_frames,
        n_records=len(records),
        details={"provider": artifact.manifest.get("provider")},
    )


def _assert_new_or_resumable(output_dir: Path) -> None:
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise ValueError(
            f"channel output is not empty and is not a complete resumable artifact: {output_dir}"
        )


def _coerce_ocr_observation(value: Any) -> OCRObservation:
    if isinstance(value, OCRObservation):
        return value
    if isinstance(value, (list, tuple)) and len(value) >= 1:
        text = str(value[0])
        confidence = float(value[1]) if len(value) > 1 and value[1] is not None else None
        boxes = tuple(value[2]) if len(value) > 2 and value[2] else ()
        return OCRObservation(text, confidence, boxes)
    raise ValueError(f"unsupported OCR observation type: {type(value).__name__}")


def _coerce_object_observation(value: Any) -> ObjectObservation:
    if isinstance(value, ObjectObservation):
        return value
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        bbox = value[2] if len(value) > 2 else None
        return ObjectObservation(str(value[0]), float(value[1]), bbox)
    raise ValueError(f"unsupported object observation type: {type(value).__name__}")


def _coerce_asr_observation(value: Any) -> ASRObservation:
    if isinstance(value, ASRObservation):
        return value
    raise ValueError(f"unsupported ASR observation type: {type(value).__name__}")


def build_ocr_channel(
    raw_root: Path,
    output_dir: Path,
    provider: Any,
    *,
    batch_size: int = 1,
) -> ChannelRunResult:
    """Run real OCR on every generated raw frame and persist a BM25-ready artifact."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    dataset_hash = _read_dataset_hash(raw_root)
    pairs = load_raw_catalog_frames(raw_root)
    resumed = _resume_result(
        output_dir,
        channel="ocr",
        dataset_hash=dataset_hash,
        n_input_frames=len(pairs),
        artifact_name="ocr.jsonl",
        manifest_name="ocr_manifest.json",
        load_artifact=load_ocr_artifact,
    )
    if resumed is not None:
        return resumed
    provider_name, revision = _provider_identity(provider)
    raw_catalog_sha256 = sha256_path(Path(raw_root) / "catalog.jsonl")
    valid_frame_uids = {frame.frame_id for frame, _image_path in pairs}
    processed_frame_uids, records = _load_ocr_checkpoint(
        output_dir,
        dataset_hash=dataset_hash,
        raw_catalog_sha256=raw_catalog_sha256,
        n_input_frames=len(pairs),
        provider=provider_name,
        revision=revision,
        valid_frame_uids=valid_frame_uids,
    )
    _assert_new_or_resumable(output_dir)
    checkpoint_records_path, checkpoint_manifest_path = _ocr_checkpoint_paths(output_dir)
    if not checkpoint_manifest_path.exists():
        _write_ocr_checkpoint_manifest(
            checkpoint_manifest_path,
            {
                "format": "hcmaic-raw-ocr-checkpoint-v1",
                "dataset_hash": dataset_hash,
                "raw_catalog_sha256": raw_catalog_sha256,
                "n_input_frames": len(pairs),
                "provider": provider_name,
                "revision": revision,
            },
        )
    with checkpoint_records_path.open("a", encoding="utf-8") as checkpoint_handle:
        for frame, image_path in pairs:
            if frame.frame_id in processed_frame_uids:
                continue
            observations = [
                _coerce_ocr_observation(item) for item in provider.infer_image(image_path)
            ]
            record = ocr_record_for_frame(
                frame,
                observations,
                provider=provider_name,
                revision=revision,
            )
            if record is not None:
                records.append(record)
            _append_ocr_checkpoint(
                checkpoint_handle,
                frame_uid=frame.frame_id,
                record=record,
            )
            processed_frame_uids.add(frame.frame_id)
    if not records:
        stage_path = _write_stage_manifest(
            output_dir,
            {
                "channel": "ocr",
                "status": "no_text_detected",
                "dataset_hash": dataset_hash,
                "n_input_frames": len(pairs),
                "n_records": 0,
            },
        )
        _remove_ocr_checkpoint(output_dir)
        return ChannelRunResult(
            "ocr", "no_text_detected", None, stage_path, dataset_hash, len(pairs), 0, {}
        )
    artifact = write_ocr_artifact(
        records,
        output_dir,
        dataset_manifest_hash=dataset_hash,
        manifest_extra=_base_manifest_extra(
            raw_root=raw_root,
            channel="ocr",
            n_input_frames=len(pairs),
            dataset_hash=dataset_hash,
            provider=provider,
            batch_size=batch_size,
        ),
    )
    _remove_ocr_checkpoint(output_dir)
    stage_path = _write_stage_manifest(
        output_dir,
        {
            "channel": "ocr",
            "status": "built",
            "dataset_hash": dataset_hash,
            "n_input_frames": len(pairs),
            "n_records": len(artifact.records),
            "provider": provider_name,
            "revision": revision,
        },
    )
    return ChannelRunResult(
        "ocr",
        "built",
        Path(output_dir),
        stage_path,
        dataset_hash,
        len(pairs),
        len(artifact.records),
        {},
    )


def build_object_channel(
    raw_root: Path,
    output_dir: Path,
    provider: Any,
    *,
    batch_size: int = 8,
) -> ChannelRunResult:
    """Run a real detector over raw frames and persist mapped object records."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    dataset_hash = _read_dataset_hash(raw_root)
    pairs = load_raw_catalog_frames(raw_root)
    from hcmaic.retrieval.object_retrieval import load_object_artifact

    resumed = _resume_result(
        output_dir,
        channel="object",
        dataset_hash=dataset_hash,
        n_input_frames=len(pairs),
        artifact_name="objects.jsonl",
        manifest_name="object_manifest.json",
        load_artifact=load_object_artifact,
    )
    if resumed is not None:
        return resumed
    _assert_new_or_resumable(output_dir)
    provider_name, revision = _provider_identity(provider)
    observations = provider.infer_images([path for _frame, path in pairs], batch_size=batch_size)
    if len(observations) != len(pairs):
        raise ValueError(
            f"object provider returned {len(observations)} rows for {len(pairs)} frames"
        )
    records: list[ObjectRecord] = []
    for (frame, _image_path), frame_observations in zip(pairs, observations, strict=True):
        records.extend(
            object_records_for_frame(
                frame,
                [_coerce_object_observation(item) for item in frame_observations],
                provider=provider_name,
                revision=revision,
            )
        )
    if not records:
        stage_path = _write_stage_manifest(
            output_dir,
            {
                "channel": "object",
                "status": "no_detections",
                "dataset_hash": dataset_hash,
                "n_input_frames": len(pairs),
                "n_records": 0,
            },
        )
        return ChannelRunResult(
            "object", "no_detections", None, stage_path, dataset_hash, len(pairs), 0, {}
        )
    artifact = write_object_artifact(
        records,
        output_dir,
        dataset_manifest_hash=dataset_hash,
        manifest_extra=_base_manifest_extra(
            raw_root=raw_root,
            channel="object",
            n_input_frames=len(pairs),
            dataset_hash=dataset_hash,
            provider=provider,
            batch_size=batch_size,
        ),
    )
    stage_path = _write_stage_manifest(
        output_dir,
        {
            "channel": "object",
            "status": "built",
            "dataset_hash": dataset_hash,
            "n_input_frames": len(pairs),
            "n_records": len(artifact.records),
            "provider": provider_name,
            "revision": revision,
        },
    )
    return ChannelRunResult(
        "object",
        "built",
        Path(output_dir),
        stage_path,
        dataset_hash,
        len(pairs),
        len(artifact.records),
        {},
    )


def _raw_video_paths(raw_root: Path, video_root: Path | None) -> list[tuple[str, str, Path]]:
    manifest = json.loads((Path(raw_root) / "dataset_manifest.json").read_text(encoding="utf-8"))
    paths: list[tuple[str, str, Path]] = []
    for item in manifest.get("raw_videos", []):
        video_id = str(item.get("video_id", ""))
        filename = str(item.get("video_filename", ""))
        configured = Path(str(item.get("source_path", "")))
        candidates = [configured]
        if video_root is not None:
            candidates.insert(0, Path(video_root) / filename)
        video_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if video_path is not None:
            paths.append((video_id, filename, video_path))
    return paths


def build_asr_channel(
    raw_root: Path,
    output_dir: Path,
    provider: Any,
    *,
    video_root: Path | None = None,
) -> ChannelRunResult:
    """Run optional ASR with audio probing; absence never creates fake records."""
    dataset_hash = _read_dataset_hash(raw_root)
    pairs = load_raw_catalog_frames(raw_root)
    from hcmaic.retrieval.asr import load_asr_artifact

    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for frame, _path in pairs:
        grouped[frame.video_id].append(frame)
    resumed = _resume_result(
        output_dir,
        channel="asr",
        dataset_hash=dataset_hash,
        n_input_frames=len(pairs),
        artifact_name="asr.jsonl",
        manifest_name="asr_manifest.json",
        load_artifact=load_asr_artifact,
    )
    if resumed is not None:
        return resumed
    _assert_new_or_resumable(output_dir)
    provider_name, revision = _provider_identity(provider)
    all_records: list[ASRRecord] = []
    probe_status: dict[str, str] = {}
    video_paths = _raw_video_paths(raw_root, video_root)
    for video_id, _filename, video_path in video_paths:
        has_audio = probe_audio_stream(video_path)
        if has_audio is True:
            probe_status[video_id] = "audio_present"
        elif has_audio is False:
            probe_status[video_id] = "no_audio"
        else:
            probe_status[video_id] = "probe_unavailable"
        if has_audio is False:
            continue
        observations = [
            _coerce_asr_observation(item) for item in provider.infer_video(video_path)
        ]
        all_records.extend(
            asr_records_for_video(
                grouped.get(video_id, []),
                observations,
                provider=provider_name,
                revision=revision,
            )
        )
    if not all_records:
        stage_path = _write_stage_manifest(
            output_dir,
            {
                "channel": "asr",
                "status": "skipped_no_audio_or_transcript",
                "dataset_hash": dataset_hash,
                "n_input_frames": len(pairs),
                "n_records": 0,
                "audio_probe": probe_status,
            },
        )
        return ChannelRunResult(
            "asr",
            "skipped_no_audio_or_transcript",
            None,
            stage_path,
            dataset_hash,
            len(pairs),
            0,
            {"audio_probe": probe_status},
        )
    artifact = write_asr_artifact(
        all_records,
        output_dir,
        dataset_manifest_hash=dataset_hash,
        manifest_extra={
            **_base_manifest_extra(
                raw_root=raw_root,
                channel="asr",
                n_input_frames=len(pairs),
                dataset_hash=dataset_hash,
                provider=provider,
                batch_size=1,
            ),
            "audio_probe": probe_status,
            "n_videos_scanned": len(video_paths),
        },
    )
    stage_path = _write_stage_manifest(
        output_dir,
        {
            "channel": "asr",
            "status": "built",
            "dataset_hash": dataset_hash,
            "n_input_frames": len(pairs),
            "n_records": len(artifact.records),
            "provider": provider_name,
            "revision": revision,
            "audio_probe": probe_status,
        },
    )
    return ChannelRunResult(
        "asr",
        "built",
        Path(output_dir),
        stage_path,
        dataset_hash,
        len(pairs),
        len(artifact.records),
        {"audio_probe": probe_status},
    )
