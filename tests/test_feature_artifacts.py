from __future__ import annotations

from pathlib import Path

import pytest

from hcmaic.features.artifacts import (
    FeatureArtifactError,
    load_feature_records,
    write_feature_records,
)
from hcmaic.features.base import FeatureContext
from hcmaic.features.mock import MockOCRProvider


def test_feature_artifact_round_trip_and_hash_validation(tmp_path: Path):
    records = MockOCRProvider().extract(
        FeatureContext("V1", "V1:001", 0, 1000, text_hint="hello")
    )
    path = tmp_path / "features.jsonl"
    manifest = write_feature_records(records, path)
    loaded = load_feature_records(path, expected_content_hash=manifest["content_hash"])
    assert loaded == records
    assert manifest["modality"] == "ocr"


def test_feature_artifact_fails_closed_on_tampering(tmp_path: Path):
    records = MockOCRProvider().extract(
        FeatureContext("V1", "V1:001", 0, 1000, text_hint="hello")
    )
    path = tmp_path / "features.jsonl"
    write_feature_records(records, path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FeatureArtifactError, match="hash"):
        load_feature_records(path, expected_content_hash="wrong")
