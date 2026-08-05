"""K4 OCR artifact provenance and BM25 channel tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmaic.retrieval.ocr_bm25 import (
    BM25OCRChannel,
    OCRArtifactError,
    OCRRecord,
    OCRUnavailableError,
    compact_ocr_text,
    load_ocr_artifact,
    normalize_ocr_text,
    write_ocr_artifact,
)


def _records() -> list[OCRRecord]:
    return [
        OCRRecord(
            frame_uid="V1:000010",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=10,
            timestamp_ms=500,
            text="go san pham",
            provider="paddleocr",
            revision="paddleocr-local-v1",
            confidence=0.8,
        ),
        OCRRecord(
            frame_uid="V1:000000",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=0,
            timestamp_ms=0,
            text="Sản phẩm gỗ đẹp",
            provider="paddleocr",
            revision="paddleocr-local-v1",
            confidence=0.95,
        ),
        OCRRecord(
            frame_uid="V1:000020",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=20,
            timestamp_ms=1000,
            text="ABC 123",
            provider="paddleocr",
            revision="paddleocr-local-v1",
        ),
    ]


def test_ocr_normalization_removes_diacritics_and_supports_compact_text():
    assert normalize_ocr_text("Sản phẩm GỖ") == "san pham go"
    assert compact_ocr_text("ABC 123") == "abc123"


def test_ocr_artifact_round_trip_and_bm25_mapping(tmp_path: Path):
    artifact = write_ocr_artifact(
        _records(),
        tmp_path / "ocr",
        dataset_manifest_hash="raw-dataset-hash",
    )
    assert [record.source_frame_idx for record in artifact.records] == [0, 10, 20]
    loaded = load_ocr_artifact(
        tmp_path / "ocr", dataset_manifest_hash="raw-dataset-hash"
    )
    channel = BM25OCRChannel(loaded)

    hits = channel.search("sản phẩm gỗ", top_k=2)

    assert hits[0].frame_uid == "V1:000000"
    assert hits[0].video_filename == "V1.mp4"
    assert hits[0].source_frame_idx == 0
    assert hits[0].modality == "ocr"
    assert hits[0].evidence_text == "Sản phẩm gỗ đẹp"
    assert hits[0].evidence["normalized_query"] == "san pham go"

    compact_hits = channel.search("abc123", top_k=1)
    assert compact_hits[0].source_frame_idx == 20


def test_ocr_artifact_rejects_mock_and_tampering(tmp_path: Path):
    with pytest.raises(ValueError, match="real non-mock"):
        OCRRecord(
            frame_uid="V1:0",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=0,
            timestamp_ms=0,
            text="text",
            provider="mock-ocr",
            revision="fixture-v1",
        )

    write_ocr_artifact(_records(), tmp_path / "ocr", dataset_manifest_hash="hash")
    manifest_path = tmp_path / "ocr" / "ocr_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["btc_artifacts_used"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OCRArtifactError, match="BTC"):
        load_ocr_artifact(tmp_path / "ocr")


def test_missing_ocr_artifact_is_unavailable(tmp_path: Path):
    with pytest.raises(OCRUnavailableError, match="unavailable"):
        BM25OCRChannel.from_artifact(tmp_path / "missing")
