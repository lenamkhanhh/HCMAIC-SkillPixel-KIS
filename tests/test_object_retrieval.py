"""K5 object artifact provenance and posting-list retrieval tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmaic.retrieval.object_retrieval import (
    ObjectArtifactError,
    ObjectRecord,
    ObjectRetrievalChannel,
    ObjectUnavailableError,
    load_object_artifact,
    normalize_object_label,
    write_object_artifact,
)


def _records() -> list[ObjectRecord]:
    return [
        ObjectRecord(
            frame_uid="V1:000020",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=20,
            timestamp_ms=1000,
            label="person",
            confidence=0.65,
            provider="test-object-provider",
            revision="test-object-v1",
        ),
        ObjectRecord(
            frame_uid="V1:000000",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=0,
            timestamp_ms=0,
            label="red car",
            confidence=0.95,
            provider="test-object-provider",
            revision="test-object-v1",
            bbox=(1.0, 2.0, 30.0, 40.0),
        ),
        ObjectRecord(
            frame_uid="V1:000010",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=10,
            timestamp_ms=500,
            label="person",
            confidence=0.9,
            provider="test-object-provider",
            revision="test-object-v1",
        ),
    ]


def test_object_label_normalization():
    assert normalize_object_label("Người, Áo") == "nguoi ao"


def test_object_artifact_round_trip_and_mapping(tmp_path: Path):
    artifact = write_object_artifact(
        _records(), tmp_path / "objects", dataset_manifest_hash="raw-hash"
    )
    assert [record.source_frame_idx for record in artifact.records] == [0, 10, 20]
    loaded = load_object_artifact(
        tmp_path / "objects", dataset_manifest_hash="raw-hash"
    )
    channel = ObjectRetrievalChannel(loaded)

    hits = channel.search("a red car", top_k=2)

    assert hits[0].frame_uid == "V1:000000"
    assert hits[0].source_frame_idx == 0
    assert hits[0].video_filename == "V1.mp4"
    assert hits[0].evidence["confidence"] == pytest.approx(0.95)
    assert hits[0].evidence["bbox"] == [1.0, 2.0, 30.0, 40.0]


def test_object_artifact_rejects_mock_and_btc_tamper(tmp_path: Path):
    with pytest.raises(ValueError, match="real non-mock"):
        ObjectRecord(
            frame_uid="V1:0",
            video_id="V1",
            video_filename="V1.mp4",
            source_frame_idx=0,
            timestamp_ms=0,
            label="person",
            confidence=0.5,
            provider="mock-object",
            revision="fixture-v1",
        )

    write_object_artifact(_records(), tmp_path / "objects", dataset_manifest_hash="hash")
    manifest_path = tmp_path / "objects" / "object_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["btc_artifacts_used"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ObjectArtifactError, match="BTC"):
        load_object_artifact(tmp_path / "objects")


def test_missing_object_artifact_is_unavailable(tmp_path: Path):
    with pytest.raises(ObjectUnavailableError, match="unavailable"):
        ObjectRetrievalChannel.from_artifact(tmp_path / "missing")
