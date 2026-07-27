from __future__ import annotations

import pytest

from hcmaic.ingestion.shots import (
    NoShotDetector,
    Shot,
    ShotDetectorUnavailable,
    sample_shot_times,
)


def test_no_shot_detector_returns_one_full_video_shot():
    shots = NoShotDetector().detect(video_id="V1", duration_s=7.5)
    assert shots == [Shot("V1:shot-000", "V1", 0.0, 7.5)]


def test_sample_shot_times_keeps_one_representative_per_shot_and_is_deterministic():
    shots = [
        Shot("V1:shot-000", "V1", 0.0, 1.0),
        Shot("V1:shot-001", "V1", 1.0, 7.0),
    ]
    expected = [0.0, 1.0, 3.0, 5.0]
    assert sample_shot_times(shots, interval_s=2.0, max_frames=10) == expected
    assert sample_shot_times(shots, interval_s=2.0, max_frames=10) == expected


def test_sampling_caps_deterministically_without_dropping_shot_representatives():
    shots = [
        Shot("V1:shot-000", "V1", 0.0, 1.0),
        Shot("V1:shot-001", "V1", 1.0, 2.0),
        Shot("V1:shot-002", "V1", 2.0, 3.0),
    ]
    assert sample_shot_times(shots, interval_s=0.1, max_frames=2) == [0.0, 1.0]


def test_heavy_shot_detector_slots_are_explicitly_unavailable():
    with pytest.raises(ShotDetectorUnavailable, match="PySceneDetect"):
        from hcmaic.ingestion.shots import PySceneDetectShotDetector

        PySceneDetectShotDetector().detect("V1", 1.0)
    with pytest.raises(ShotDetectorUnavailable, match="TransNetV2"):
        from hcmaic.ingestion.shots import TransNetV2ShotDetector

        TransNetV2ShotDetector().detect("V1", 1.0)
