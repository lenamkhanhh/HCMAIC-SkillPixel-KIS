"""K6 optional ASR artifact, channel and promotion-gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from hcmaic.retrieval.asr import (
    ASRRecord,
    ASRRetrievalChannel,
    ASRUnavailableError,
    decide_asr_promotion,
    load_asr_artifact,
    write_asr_artifact,
)


def _records() -> list[ASRRecord]:
    return [
        ASRRecord(
            segment_id="V1:seg-1",
            frame_uid="V1:000010",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=10,
            timestamp_ms=500,
            start_ms=250,
            end_ms=900,
            text="the product is ready",
            provider="test-asr-provider",
            revision="test-asr-v1",
            confidence=0.9,
        ),
        ASRRecord(
            segment_id="V1:seg-0",
            frame_uid="V1:000000",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=0,
            timestamp_ms=0,
            start_ms=0,
            end_ms=200,
            text="hello there",
            provider="test-asr-provider",
            revision="test-asr-v1",
            confidence=0.8,
        ),
    ]


def test_asr_promotion_requires_qrels_and_gain():
    disabled_without_qrels = decide_asr_promotion(
        qrels_available=False, baseline_score=None, asr_score=None
    )
    assert not disabled_without_qrels.enabled
    assert disabled_without_qrels.reason == "disabled_without_hcmaic_qrels"

    disabled_without_gain = decide_asr_promotion(
        qrels_available=True, baseline_score=0.7, asr_score=0.705, minimum_gain=0.01
    )
    assert not disabled_without_gain.enabled
    assert disabled_without_gain.gain == pytest.approx(0.005)

    enabled = decide_asr_promotion(
        qrels_available=True, baseline_score=0.7, asr_score=0.72, minimum_gain=0.01
    )
    assert enabled.enabled
    assert enabled.gain == pytest.approx(0.02)


def test_asr_artifact_round_trip_and_source_mapping(tmp_path: Path):
    artifact = write_asr_artifact(
        _records(), tmp_path / "asr", dataset_manifest_hash="raw-hash"
    )
    assert [record.segment_id for record in artifact.records] == ["V1:seg-0", "V1:seg-1"]
    loaded = load_asr_artifact(tmp_path / "asr", dataset_manifest_hash="raw-hash")
    channel = ASRRetrievalChannel(loaded)

    hits = channel.search("product ready", top_k=1)

    assert hits[0].frame_uid == "V1:000010"
    assert hits[0].source_frame_idx == 10
    assert hits[0].timestamp_ms == 500
    assert hits[0].evidence["start_ms"] == 250
    assert hits[0].evidence["end_ms"] == 900


def test_asr_rejects_mock_and_missing_artifact(tmp_path: Path):
    with pytest.raises(ValueError, match="real non-mock"):
        ASRRecord(
            segment_id="seg",
            frame_uid="V1:0",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=0,
            timestamp_ms=0,
            start_ms=0,
            end_ms=1,
            text="text",
            provider="mock-asr",
            revision="fixture-v1",
        )
    with pytest.raises(ASRUnavailableError, match="unavailable"):
        ASRRetrievalChannel.from_artifact(tmp_path / "missing")
