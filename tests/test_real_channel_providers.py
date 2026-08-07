from pathlib import Path

import pytest

from hcmaic.contracts.models import FrameRecord
from hcmaic.retrieval.asr import ASRRecord, write_asr_artifact
from hcmaic.retrieval.object_retrieval import ObjectRecord, write_object_artifact
from hcmaic.retrieval.ocr_bm25 import OCRRecord, write_ocr_artifact
from hcmaic.retrieval.real_channels import (
    ASRObservation,
    ObjectObservation,
    OCRObservation,
    PaddleOCRFrameProvider,
    asr_records_for_video,
    object_records_for_frame,
    ocr_record_for_frame,
)


def test_paddleocr_constructor_disables_windows_mkldnn_runtime_path() -> None:
    provider = PaddleOCRFrameProvider.__new__(PaddleOCRFrameProvider)
    provider.device = "cpu"
    provider.model_path = None
    captured: dict[str, object] = {}

    def fake_constructor(ocr_version: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    provider._construct(fake_constructor, "PP-OCRv6")

    assert captured["enable_mkldnn"] is False


def test_paddleocr_legacy_constructor_requests_cuda_without_silent_cpu_fallback() -> None:
    provider = PaddleOCRFrameProvider.__new__(PaddleOCRFrameProvider)
    provider.device = "cuda"
    provider.model_path = None
    captured: dict[str, object] = {}

    class LegacyPaddleOCR:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    provider._construct(LegacyPaddleOCR, "PP-OCRv6")

    assert captured["use_gpu"] is True
    assert captured["use_angle_cls"] is False
    assert captured["enable_mkldnn"] is False
    assert provider.actual_model_version == "PaddleOCR-legacy"


def _frame(
    *,
    frame_id: str = "video1:000",
    timestamp_ms: int = 800,
    source_frame_idx: int = 20,
) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        video_id="video1",
        keyframe_id=frame_id.rsplit(":", 1)[1],
        frame_idx=source_frame_idx,
        source_frame_idx=source_frame_idx,
        timestamp_ms=timestamp_ms,
        image_path="keyframes/video1/000.jpg",
        video_filename="video1.mp4",
        frame_count=100,
    )


def test_ocr_mapping_preserves_source_identity_and_boxes() -> None:
    record = ocr_record_for_frame(
        _frame(),
        [
            OCRObservation("Xe xanh", 0.91, ((1.0, 2.0, 30.0, 40.0),)),
            OCRObservation("123", 0.81, ((5.0, 6.0, 20.0, 25.0),)),
        ],
        provider="paddleocr",
        revision="PP-OCRv6",
    )

    assert record is not None
    assert isinstance(record, OCRRecord)
    assert record.frame_uid == "video1:000"
    assert record.source_frame_idx == 20
    assert record.timestamp_ms == 800
    assert record.text == "Xe xanh 123"
    assert record.metadata == {
        "boxes": [[1.0, 2.0, 30.0, 40.0], [5.0, 6.0, 20.0, 25.0]
        ],
    }


def test_ocr_empty_observations_do_not_create_fake_text() -> None:
    assert (
        ocr_record_for_frame(
            _frame(),
            [],
            provider="paddleocr",
            revision="PP-OCRv6",
        )
        is None
    )


def test_object_mapping_deduplicates_label_by_highest_confidence() -> None:
    records = object_records_for_frame(
        _frame(),
        [
            ObjectObservation("person", 0.42, (1.0, 2.0, 3.0, 4.0)),
            ObjectObservation("person", 0.88, (5.0, 6.0, 7.0, 8.0)),
            ObjectObservation("car", 0.73, None),
        ],
        provider="ultralytics:yolo11n",
        revision="weights-sha256",
    )

    assert [record.label for record in records] == ["person", "car"]
    person = next(record for record in records if record.label == "person")
    assert person.confidence == pytest.approx(0.88)
    assert person.bbox == (5.0, 6.0, 7.0, 8.0)
    assert all(record.source_frame_idx == 20 for record in records)


def test_asr_segments_anchor_to_nearest_sampled_source_frame() -> None:
    frames = [
        _frame(frame_id="video1:000", timestamp_ms=800, source_frame_idx=20),
        _frame(frame_id="video1:001", timestamp_ms=1600, source_frame_idx=40),
    ]
    records = asr_records_for_video(
        frames,
        [
            ASRObservation(
                segment_id="video1:segment-0",
                start_ms=1200,
                end_ms=1500,
                text="xin chao",
                confidence=None,
            )
        ],
        provider="faster-whisper",
        revision="small-sha256",
    )

    assert records == [
        ASRRecord(
            segment_id="video1:segment-0",
            frame_uid="video1:001",
            video_id="video1",
            video_filename="video1.mp4",
            source_frame_idx=40,
            timestamp_ms=1600,
            start_ms=1200,
            end_ms=1500,
            text="xin chao",
            provider="faster-whisper",
            revision="small-sha256",
        )
    ]


def test_channel_manifests_keep_weight_source_and_runtime_metadata(tmp_path: Path) -> None:
    frame = _frame()
    ocr_dir = tmp_path / "ocr"
    object_dir = tmp_path / "object"
    asr_dir = tmp_path / "asr"
    metadata = {
        "model_source_url": "https://example.invalid/model",
        "weights_sha256": "abc123",
        "runtime": {"device": "cpu", "batch_size": 4},
    }
    write_ocr_artifact(
        [OCRRecord(
            frame_uid=frame.frame_id,
            video_id=frame.video_id,
            video_filename=frame.video_filename or "video1.mp4",
            source_frame_idx=20,
            timestamp_ms=800,
            text="hello",
            provider="paddleocr",
            revision="PP-OCRv6",
        )],
        ocr_dir,
        dataset_manifest_hash="dataset-hash",
        manifest_extra=metadata,
    )
    write_object_artifact(
        [ObjectRecord(
            frame_uid=frame.frame_id,
            video_id=frame.video_id,
            video_filename=frame.video_filename or "video1.mp4",
            source_frame_idx=20,
            timestamp_ms=800,
            label="person",
            confidence=0.9,
            provider="ultralytics:yolo11n",
            revision="weights-sha256",
        )],
        object_dir,
        dataset_manifest_hash="dataset-hash",
        manifest_extra=metadata,
    )
    write_asr_artifact(
        [ASRRecord(
            segment_id="video1:segment-0",
            frame_uid=frame.frame_id,
            video_id=frame.video_id,
            video_filename=frame.video_filename or "video1.mp4",
            source_frame_idx=20,
            timestamp_ms=800,
            start_ms=700,
            end_ms=900,
            text="hello",
            provider="faster-whisper",
            revision="small-sha256",
        )],
        asr_dir,
        dataset_manifest_hash="dataset-hash",
        manifest_extra=metadata,
    )

    import json

    for path in (
        ocr_dir / "ocr_manifest.json",
        object_dir / "object_manifest.json",
        asr_dir / "asr_manifest.json",
    ):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["model_source_url"] == metadata["model_source_url"]
        assert manifest["weights_sha256"] == metadata["weights_sha256"]
        assert manifest["runtime"] == metadata["runtime"]
