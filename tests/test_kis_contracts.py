"""K0 contracts for the full HCMAIC KIS runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from hcmaic.contracts.kis import (
    Evidence,
    KISChannelConfig,
    KISPipelineConfig,
    KISQuery,
    KISResult,
)


def _evidence() -> Evidence:
    return Evidence(
        channel="visual",
        frame_uid="video-a:000",
        video_id="video-a",
        video_filename="video-a.mp4",
        source_frame_idx=20,
        timestamp_ms=800,
        score=0.91,
        rank=1,
        evidence_level="REAL_PROVIDER",
    )


def test_kis_query_preserves_raw_text_and_routes_task() -> None:
    query = KISQuery(query_id="Q1", task="tkis", text="  biển báo đỏ  ")

    assert query.task == "TKIS"
    assert query.text == "biển báo đỏ"
    assert query.raw_text == "  biển báo đỏ  "
    assert query.image_path is None


def test_kis_query_requires_task_input() -> None:
    with pytest.raises(ValueError, match="TKIS text"):
        KISQuery(query_id="Q1", task="TKIS")
    with pytest.raises(ValueError, match="VKIS image_path"):
        KISQuery(query_id="Q2", task="VKIS")


def test_kis_result_keeps_mapping_and_evidence_envelope() -> None:
    result = KISResult(
        query_id="Q1",
        task="TKIS",
        rank=1,
        frame_uid="video-a:000",
        video_id="video-a",
        video_filename="video-a.mp4",
        source_frame_idx=20,
        timestamp_ms=800,
        channel_scores={"visual": 0.91},
        fused_score=0.91,
        rerank_score=0.91,
        evidence=(_evidence(),),
        executed_channels=("visual",),
    )

    payload = result.to_dict()

    assert result.answer_cell == "video-a.mp4,20"
    assert payload["frame_uid"] == "video-a:000"
    assert payload["source_frame_idx"] == 20
    assert payload["evidence"][0]["channel"] == "visual"
    assert payload["quality_status"] == "UNVALIDATED_ON_HCMAIC"


def test_kis_result_order_is_deterministic() -> None:
    base = dict(
        query_id="Q1",
        task="TKIS",
        frame_uid="video-a:000",
        video_id="video-a",
        video_filename="video-a.mp4",
        source_frame_idx=20,
        timestamp_ms=800,
        channel_scores={"visual": 0.9},
        fused_score=0.9,
    )
    first = KISResult(rank=2, **base)
    second = KISResult(rank=1, **{**base, "frame_uid": "video-b:000", "video_id": "video-b"})

    assert sorted([first, second], key=lambda item: item.sort_key()) == [first, second]


def test_pipeline_config_rejects_mock_production_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mock"):
        KISPipelineConfig(dataset_root=tmp_path, visual_provider="mock")


def test_pipeline_config_serializes_channels_and_quality_status(tmp_path: Path) -> None:
    config = KISPipelineConfig(
        dataset_root=tmp_path,
        visual_provider="clip",
        channels=(
            KISChannelConfig(name="visual", provider="clip"),
            KISChannelConfig(name="ocr", provider="paddleocr", enabled=False),
        ),
    )

    payload = config.to_dict()

    assert payload["quality_status"] == "UNVALIDATED_ON_HCMAIC"
    assert payload["dataset_root"] == str(tmp_path)
    assert payload["channels"][1]["enabled"] is False
    assert config.enabled_channels == ("visual",)
