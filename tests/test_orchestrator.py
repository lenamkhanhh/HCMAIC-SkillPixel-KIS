from __future__ import annotations

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.orchestrator import (
    ChannelUnavailableError,
    RetrievalOrchestrator,
)


class FakeChannel:
    def __init__(self, hits=None, unavailable=False):
        self.hits = hits or []
        self.unavailable = unavailable

    def search(self, text: str, top_k: int):
        assert text
        if self.unavailable:
            raise ChannelUnavailableError("model absent")
        return self.hits[:top_k]


def test_orchestrator_keeps_visual_results_when_optional_channel_is_unavailable():
    visual = ChannelHit("A", "V1", 1000, "visual", 0.9, 1, "clip")
    orchestrator = RetrievalOrchestrator(
        {
            "visual": FakeChannel([visual]),
            "ocr": FakeChannel(unavailable=True),
        }
    )
    output = orchestrator.search("red bus", top_k=5)
    assert [item.entity_id for item in output.candidates] == ["A"]
    assert output.unavailable_channels == {"ocr": "model absent"}


def test_orchestrator_supports_weighted_fusion():
    visual = ChannelHit("A", "V1", 1000, "visual", 0.9, 1, "clip")
    ocr = ChannelHit("B", "V1", 2000, "ocr", 0.8, 1, "mock-ocr")
    orchestrator = RetrievalOrchestrator(
        {"visual": FakeChannel([visual]), "ocr": FakeChannel([ocr])},
        fusion_method="weighted",
        weights={"visual": 0.1, "ocr": 1.0},
    )
    assert orchestrator.search("bus", top_k=2).candidates[0].entity_id == "B"
