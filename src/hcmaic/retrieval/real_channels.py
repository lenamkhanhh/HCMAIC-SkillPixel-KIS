"""Raw-frame adapters for optional real SkillPixel retrieval channels.

The module deliberately keeps heavyweight dependencies optional and lazy.  A
channel is either constructed with a real provider and emits provenance-rich
observations, or raises :class:`RealChannelUnavailable`; callers must record
that state instead of silently substituting another model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmaic.contracts.models import FrameRecord
from hcmaic.retrieval.asr import ASRRecord
from hcmaic.retrieval.object_retrieval import (
    ObjectRecord,
    normalize_object_label,
)
from hcmaic.retrieval.ocr_bm25 import OCRRecord

PP_OCRV6_SOURCE_URL = (
    "https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html"
)
ULTRALYTICS_PREDICT_SOURCE_URL = "https://docs.ultralytics.com/modes/predict/"
FASTER_WHISPER_SOURCE_URL = "https://github.com/SYSTRAN/faster-whisper"


class RealChannelUnavailable(RuntimeError):
    """Raised when a requested real provider cannot be executed safely."""


@dataclass(frozen=True)
class OCRObservation:
    """One raw-image OCR detection before it is mapped to a frame record."""

    text: str
    confidence: float | None = None
    boxes: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True)
class ObjectObservation:
    """One raw-image object detection before it is mapped to a frame record."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ASRObservation:
    """One timestamped transcript segment before frame anchoring."""

    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None


def _frame_identity(frame: FrameRecord) -> tuple[str, str, str, int, int]:
    source_frame_idx = frame.source_frame_idx
    if source_frame_idx is None:
        source_frame_idx = frame.frame_idx
    video_filename = frame.video_filename or f"{frame.video_id}.mp4"
    return (
        frame.frame_id,
        frame.video_id,
        video_filename,
        source_frame_idx,
        frame.timestamp_ms,
    )


def _checked_confidence(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return confidence


def _checked_bbox(
    bbox: Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    values = tuple(float(value) for value in bbox)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("bbox coordinates must be finite and non-negative")
    return values  # type: ignore[return-value]


def ocr_record_for_frame(
    frame: FrameRecord,
    observations: Iterable[OCRObservation],
    *,
    provider: str,
    revision: str,
) -> OCRRecord | None:
    """Aggregate non-empty OCR detections while preserving source-frame identity."""
    texts: list[str] = []
    boxes: list[list[float]] = []
    confidences: list[float] = []
    for observation in observations:
        text = observation.text.strip()
        if not text:
            continue
        texts.append(text)
        confidence = _checked_confidence(observation.confidence, "OCR confidence")
        if confidence is not None:
            confidences.append(confidence)
        for box in observation.boxes:
            checked = _checked_bbox(box)
            if checked is not None:
                boxes.append(list(checked))
    if not texts:
        return None
    frame_uid, video_id, video_filename, source_frame_idx, timestamp_ms = _frame_identity(frame)
    metadata: dict[str, Any] = {}
    if boxes:
        metadata["boxes"] = boxes
    return OCRRecord(
        frame_uid=frame_uid,
        video_id=video_id,
        video_filename=video_filename,
        source_frame_idx=source_frame_idx,
        timestamp_ms=timestamp_ms,
        text=" ".join(texts),
        provider=provider,
        revision=revision,
        confidence=(sum(confidences) / len(confidences) if confidences else None),
        metadata=metadata,
    )


def object_records_for_frame(
    frame: FrameRecord,
    observations: Iterable[ObjectObservation],
    *,
    provider: str,
    revision: str,
) -> list[ObjectRecord]:
    """Map detections to a frame and keep the highest-confidence label once."""
    selected: dict[str, ObjectObservation] = {}
    order: list[str] = []
    for observation in observations:
        label = observation.label.strip()
        normalized = normalize_object_label(label)
        if not normalized:
            continue
        confidence = _checked_confidence(observation.confidence, "object confidence")
        assert confidence is not None
        checked = ObjectObservation(
            label=label,
            confidence=confidence,
            bbox=_checked_bbox(observation.bbox),
        )
        if normalized not in selected:
            order.append(normalized)
            selected[normalized] = checked
        elif checked.confidence > selected[normalized].confidence:
            selected[normalized] = checked
    frame_uid, video_id, video_filename, source_frame_idx, timestamp_ms = _frame_identity(frame)
    return [
        ObjectRecord(
            frame_uid=frame_uid,
            video_id=video_id,
            video_filename=video_filename,
            source_frame_idx=source_frame_idx,
            timestamp_ms=timestamp_ms,
            label=selected[normalized].label,
            confidence=selected[normalized].confidence,
            provider=provider,
            revision=revision,
            bbox=selected[normalized].bbox,
        )
        for normalized in order
    ]


def asr_records_for_video(
    frames: Sequence[FrameRecord],
    segments: Iterable[ASRObservation],
    *,
    provider: str,
    revision: str,
) -> list[ASRRecord]:
    """Anchor each transcript midpoint to the nearest sampled frame."""
    ordered_frames = sorted(
        frames,
        key=lambda frame: (
            frame.timestamp_ms,
            frame.source_frame_idx if frame.source_frame_idx is not None else frame.frame_idx,
            frame.frame_id,
        ),
    )
    if not ordered_frames:
        return []
    records: list[ASRRecord] = []
    seen_segment_ids: set[str] = set()
    for segment in segments:
        if not segment.segment_id.strip():
            raise ValueError("ASR segment_id must not be blank")
        if segment.segment_id in seen_segment_ids:
            raise ValueError(f"duplicate ASR segment_id: {segment.segment_id}")
        seen_segment_ids.add(segment.segment_id)
        if segment.start_ms < 0 or segment.end_ms < segment.start_ms:
            raise ValueError("ASR segment timestamps are invalid")
        if not segment.text.strip():
            continue
        confidence = _checked_confidence(segment.confidence, "ASR confidence")
        midpoint = (segment.start_ms + segment.end_ms) / 2.0
        frame = min(
            ordered_frames,
            key=lambda item: (
                abs(item.timestamp_ms - midpoint),
                item.timestamp_ms,
                item.source_frame_idx if item.source_frame_idx is not None else item.frame_idx,
                item.frame_id,
            ),
        )
        frame_uid, video_id, video_filename, source_frame_idx, timestamp_ms = _frame_identity(
            frame
        )
        records.append(
            ASRRecord(
                segment_id=segment.segment_id,
                frame_uid=frame_uid,
                video_id=video_id,
                video_filename=video_filename,
                source_frame_idx=source_frame_idx,
                timestamp_ms=timestamp_ms,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=segment.text.strip(),
                provider=provider,
                revision=revision,
                confidence=confidence,
            )
        )
    return sorted(records, key=lambda record: (record.start_ms, record.segment_id))


def sha256_path(path: Path) -> str:
    """Hash a file or a directory deterministically for provider provenance."""
    path = Path(path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    raise FileNotFoundError(path)


def package_version(package: str) -> str | None:
    """Return an installed package version without importing its heavy runtime."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_json_value(value: Any) -> Any:
    """Convert numpy/tensor-like provider outputs into JSON-compatible values."""
    if hasattr(value, "tolist"):
        return _safe_json_value(value.tolist())
    if isinstance(value, tuple):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    return value


def _model_checksum(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return sha256_path(path)
    except FileNotFoundError:
        return None


class PaddleOCRFrameProvider:
    """Real PaddleOCR adapter with explicit PP-OCRv6/v5 fallback reporting."""

    def __init__(
        self,
        *,
        model_version: str = "PP-OCRv6",
        model_path: Path | None = None,
        device: str = "cpu",
        allow_model_download: bool = False,
    ) -> None:
        self.requested_model_version = model_version
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.device = device
        self.allow_model_download = allow_model_download
        self.actual_model_version = model_version
        self.fallback_from: str | None = None
        self._engine = self._build_engine()

    @property
    def name(self) -> str:
        return "paddleocr"

    @property
    def revision(self) -> str:
        return self.actual_model_version

    def _build_engine(self) -> Any:
        if self.model_path is not None and not self.model_path.exists():
            raise RealChannelUnavailable(f"OCR model path does not exist: {self.model_path}")
        if self.model_path is None and not self.allow_model_download:
            raise RealChannelUnavailable(
                "PP-OCR model is not cached; pass an explicit model path or "
                "--allow-model-download"
            )
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            raise RealChannelUnavailable(
                "paddleocr is not installed; install the OCR extra in the execution environment"
            ) from exc

        try:
            return self._construct(PaddleOCR, self.requested_model_version)
        except Exception as first_error:
            if self.requested_model_version != "PP-OCRv6":
                raise RealChannelUnavailable(
                    f"cannot initialize {self.requested_model_version}: {first_error}"
                ) from first_error
            try:
                engine = self._construct(PaddleOCR, "PP-OCRv5")
            except Exception as fallback_error:
                raise RealChannelUnavailable(
                    f"PP-OCRv6 failed ({first_error}); PP-OCRv5 fallback failed "
                    f"({fallback_error})"
                ) from fallback_error
            self.actual_model_version = "PP-OCRv5"
            self.fallback_from = "PP-OCRv6"
            return engine

    def _construct(self, constructor: Any, version: str) -> Any:
        signature = inspect.signature(constructor)
        parameters = signature.parameters
        kwargs: dict[str, Any] = {
            "ocr_version": version,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": self.device,
        }
        if self.model_path is not None:
            for key in (
                "model_dir",
                "text_detection_model_dir",
                "text_recognition_model_dir",
            ):
                if key in parameters:
                    kwargs[key] = str(self.model_path)
                    if key != "model_dir":
                        break
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "enable_mkldnn" in parameters or accepts_kwargs:
            # PaddleOCR 3.x routes this flag through **kwargs.  Disabling
            # oneDNN avoids a known Windows CPU PIR runtime failure while
            # keeping the provider explicit and real.
            kwargs["enable_mkldnn"] = False
        supported = {key: value for key, value in kwargs.items() if key in parameters}
        if accepts_kwargs:
            supported["enable_mkldnn"] = False
        return constructor(**supported)

    def infer_image(self, image_path: Path) -> list[OCRObservation]:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        try:
            if hasattr(self._engine, "predict"):
                raw = self._engine.predict(str(image_path))
                raw_results = list(raw) if not isinstance(raw, dict) else [raw]
            else:
                raw_results = self._engine.ocr(str(image_path), cls=False)
        except Exception as exc:
            raise RealChannelUnavailable(f"OCR inference failed for {image_path}: {exc}") from exc
        return _parse_ocr_results(raw_results)

    def manifest_metadata(self, *, batch_size: int = 1) -> dict[str, Any]:
        return {
            "model_source_url": PP_OCRV6_SOURCE_URL,
            "model_id": self.actual_model_version,
            "requested_model_id": self.requested_model_version,
            "model_version": self.actual_model_version,
            "weights_path": str(self.model_path) if self.model_path is not None else None,
            "weights_sha256": _model_checksum(self.model_path),
            "provider_package": "paddleocr",
            "provider_package_version": package_version("paddleocr"),
            "runtime": {
                "device": self.device,
                "batch_size": batch_size,
                "enable_mkldnn": False,
            },
            "fallback_from": self.fallback_from,
            "provider_evidence": "REAL_PROVIDER_ARTIFACT",
        }


def _parse_ocr_results(raw_results: Any) -> list[OCRObservation]:
    """Parse PaddleOCR 3.x result objects and the legacy nested output."""
    results = raw_results if isinstance(raw_results, list) else [raw_results]
    observations: list[OCRObservation] = []
    for result in results:
        payload = result
        if hasattr(payload, "json"):
            json_value = payload.json
            payload = json_value() if callable(json_value) else json_value
        if isinstance(payload, str):
            try:
                import json

                payload = json.loads(payload)
            except ValueError:
                payload = None
        if isinstance(payload, dict):
            texts = _as_sequence(payload.get("rec_texts", payload.get("texts", [])))
            scores = _as_sequence(payload.get("rec_scores", payload.get("scores", [])))
            polygons = _as_sequence(
                payload.get("dt_polys", payload.get("rec_polys", payload.get("boxes", [])))
            )
            for index, text in enumerate(texts):
                box = _polygon_to_bbox(polygons[index]) if index < len(polygons) else None
                score = float(scores[index]) if index < len(scores) else None
                observations.append(OCRObservation(str(text), score, (box,) if box else ()))
            continue
        observations.extend(_parse_legacy_ocr(payload))
    return observations


def _parse_legacy_ocr(payload: Any) -> list[OCRObservation]:
    if not isinstance(payload, list):
        return []
    detections = payload
    if detections and isinstance(detections[0], list):
        first = detections[0]
        if (
            not (
            len(first) == 2
            and isinstance(first[1], (list, tuple))
            and first[1]
            and isinstance(first[1][0], str)
            )
            and len(detections) == 1
            and isinstance(first, list)
        ):
            detections = first
    observations: list[OCRObservation] = []
    for detection in detections:
        if not isinstance(detection, (list, tuple)) or len(detection) != 2:
            continue
        box, text_score = detection
        if not isinstance(text_score, (list, tuple)) or not text_score:
            continue
        text = str(text_score[0])
        score = float(text_score[1]) if len(text_score) > 1 else None
        bbox = _polygon_to_bbox(box)
        observations.append(OCRObservation(text, score, (bbox,) if bbox else ()))
    return observations


def _as_sequence(value: Any) -> list[Any]:
    value = _safe_json_value(value)
    return value if isinstance(value, list) else []


def _polygon_to_bbox(value: Any) -> tuple[float, float, float, float] | None:
    values = _safe_json_value(value)
    if not isinstance(values, list) or not values:
        return None
    if len(values) == 4 and all(isinstance(item, (int, float)) for item in values):
        return _checked_bbox(values)
    points = [point for point in values if isinstance(point, list) and len(point) >= 2]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return _checked_bbox((min(xs), min(ys), max(xs), max(ys)))


class UltralyticsObjectProvider:
    """Real Ultralytics detector adapter operating only on raw frame images."""

    def __init__(
        self,
        *,
        model: str = "yolo11n.pt",
        model_path: Path | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        allow_model_download: bool = False,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.model_id = model
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.allow_model_download = allow_model_download
        self._model = self._build_model()

    @property
    def name(self) -> str:
        return f"ultralytics:{self.model_id}"

    @property
    def revision(self) -> str:
        return _model_checksum(self.model_path) or self.model_id

    def _build_model(self) -> Any:
        if self.model_path is not None and not self.model_path.is_file():
            raise RealChannelUnavailable(f"object model path does not exist: {self.model_path}")
        if self.model_path is None and not self.allow_model_download:
            raise RealChannelUnavailable(
                "Ultralytics weights are not cached; pass --object-model-path or "
                "--allow-model-download"
            )
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RealChannelUnavailable(
                "ultralytics is not installed; install the object extra in the "
                "execution environment"
            ) from exc
        source = str(self.model_path) if self.model_path is not None else self.model_id
        try:
            return YOLO(source)
        except Exception as exc:
            raise RealChannelUnavailable(f"cannot initialize Ultralytics {source}: {exc}") from exc

    def infer_images(
        self,
        image_paths: Sequence[Path],
        *,
        batch_size: int = 8,
    ) -> list[list[ObjectObservation]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        paths = [Path(path) for path in image_paths]
        if any(not path.is_file() for path in paths):
            missing = next(path for path in paths if not path.is_file())
            raise FileNotFoundError(missing)
        all_observations: list[list[ObjectObservation]] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            try:
                results = self._model.predict(
                    source=[str(path) for path in chunk],
                    batch=batch_size,
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False,
                )
                results = list(results)
            except Exception as exc:
                raise RealChannelUnavailable(
                    f"object inference failed for batch starting at {start}: {exc}"
                ) from exc
            if len(results) != len(chunk):
                raise RealChannelUnavailable(
                    f"object provider returned {len(results)} results for {len(chunk)} images"
                )
            for result in results:
                all_observations.append(_parse_ultralytics_result(result, self._model))
        return all_observations

    def manifest_metadata(self, *, batch_size: int = 8) -> dict[str, Any]:
        return {
            "model_source_url": ULTRALYTICS_PREDICT_SOURCE_URL,
            "model_id": self.model_id,
            "weights_path": str(self.model_path) if self.model_path is not None else None,
            "weights_sha256": _model_checksum(self.model_path),
            "provider_package": "ultralytics",
            "provider_package_version": package_version("ultralytics"),
            "runtime": {
                "device": self.device,
                "batch_size": batch_size,
                "confidence_threshold": self.confidence_threshold,
            },
            "provider_evidence": "REAL_PROVIDER_ARTIFACT",
        }


def _parse_ultralytics_result(result: Any, model: Any) -> list[ObjectObservation]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _as_sequence(getattr(boxes, "xyxy", []))
    confidences = _as_sequence(getattr(boxes, "conf", []))
    classes = _as_sequence(getattr(boxes, "cls", []))
    names = getattr(result, "names", getattr(model, "names", {}))
    observations: list[ObjectObservation] = []
    for index, coordinates in enumerate(xyxy):
        if index >= len(confidences) or index >= len(classes):
            continue
        class_index = int(classes[index])
        if isinstance(names, dict):
            label = str(names.get(class_index, class_index))
        elif isinstance(names, list) and class_index < len(names):
            label = str(names[class_index])
        else:
            label = str(class_index)
        bbox = _checked_bbox(coordinates)
        observations.append(
            ObjectObservation(
                label=label,
                confidence=float(confidences[index]),
                bbox=bbox,
            )
        )
    return observations


class FasterWhisperProvider:
    """Real faster-whisper adapter returning timestamped transcript segments."""

    def __init__(
        self,
        *,
        model: str = "small",
        model_path: Path | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        allow_model_download: bool = False,
    ) -> None:
        self.model_id = model
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.device = device
        self.compute_type = compute_type
        self.allow_model_download = allow_model_download
        self._model = self._build_model()

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def revision(self) -> str:
        return _model_checksum(self.model_path) or self.model_id

    def _build_model(self) -> Any:
        if self.model_path is not None and not self.model_path.exists():
            raise RealChannelUnavailable(f"ASR model path does not exist: {self.model_path}")
        if self.model_path is None and not self.allow_model_download:
            raise RealChannelUnavailable(
                "Whisper weights are not cached; pass --asr-model-path or "
                "--allow-model-download"
            )
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RealChannelUnavailable(
                "faster-whisper is not installed; install the ASR extra in the "
                "execution environment"
            ) from exc
        source = str(self.model_path) if self.model_path is not None else self.model_id
        kwargs = {"device": self.device, "compute_type": self.compute_type}
        if self.model_path is None and not self.allow_model_download:
            raise RealChannelUnavailable("ASR model download is disabled")
        try:
            return WhisperModel(source, **kwargs)
        except Exception as exc:
            raise RealChannelUnavailable(
                f"cannot initialize faster-whisper {source}: {exc}"
            ) from exc

    def infer_video(self, video_path: Path) -> list[ASRObservation]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        try:
            segments, _info = self._model.transcribe(str(video_path), vad_filter=True)
            materialized = list(segments)
        except Exception as exc:
            raise RealChannelUnavailable(f"ASR inference failed for {video_path}: {exc}") from exc
        observations: list[ASRObservation] = []
        for index, segment in enumerate(materialized):
            start_ms = max(0, round(float(getattr(segment, "start", 0.0)) * 1000))
            end_ms = max(start_ms, round(float(getattr(segment, "end", 0.0)) * 1000))
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            segment_id = f"{video_path.stem}:segment-{index:06d}"
            confidence = _segment_confidence(segment)
            observations.append(ASRObservation(segment_id, start_ms, end_ms, text, confidence))
        return observations

    def manifest_metadata(self, *, batch_size: int = 1) -> dict[str, Any]:
        return {
            "model_source_url": FASTER_WHISPER_SOURCE_URL,
            "model_id": self.model_id,
            "weights_path": str(self.model_path) if self.model_path is not None else None,
            "weights_sha256": _model_checksum(self.model_path),
            "provider_package": "faster-whisper",
            "provider_package_version": package_version("faster-whisper"),
            "runtime": {
                "device": self.device,
                "compute_type": self.compute_type,
                "batch_size": batch_size,
            },
            "provider_evidence": "REAL_PROVIDER_ARTIFACT",
        }


def _segment_confidence(segment: Any) -> float | None:
    """Use confidence only when the backend exposes a bounded scalar."""
    for attribute in ("confidence", "avg_logprob"):
        value = getattr(segment, attribute, None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if attribute == "confidence" and 0.0 <= value <= 1.0 and math.isfinite(value):
            return value
    return None


def probe_audio_stream(video_path: Path) -> bool | None:
    """Probe audio with ffprobe; ``None`` means the probe executable is unavailable."""
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.returncode == 0 and bool(completed.stdout.strip())
