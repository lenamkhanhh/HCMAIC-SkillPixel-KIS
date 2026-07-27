from __future__ import annotations

import pytest

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.fusion import reciprocal_rank_fusion, weighted_late_fusion


def _channels():
    return {
        "visual": [
            ChannelHit("A", "V1", 1000, "visual", 0.9, 1, "clip"),
            ChannelHit("B", "V1", 2000, "visual", 0.8, 2, "clip"),
        ],
        "ocr": [
            ChannelHit("B", "V1", 2000, "ocr", 0.7, 1, "mock-ocr", "BUS"),
            ChannelHit("C", "V2", 3000, "ocr", 0.6, 2, "mock-ocr", "STOP"),
        ],
    }


def test_rrf_fuses_channels_and_explains_contributions():
    results = reciprocal_rank_fusion(_channels(), rank_constant=60, top_k=3)
    assert results[0].entity_id == "B"
    assert set(results[0].signal_scores) == {"visual", "ocr"}
    assert results[0].contributing_providers == ["clip", "mock-ocr"]


def test_weighted_late_fusion_respects_modality_weights():
    results = weighted_late_fusion(
        _channels(), weights={"visual": 0.2, "ocr": 1.0}, top_k=3
    )
    assert results[0].entity_id == "B"
    assert results[0].final_score > results[1].final_score


def test_weighted_fusion_rejects_unknown_or_negative_weights():
    with pytest.raises(ValueError, match="weight"):
        weighted_late_fusion(_channels(), weights={"visual": -1.0}, top_k=3)
