"""Canonical per-channel and fused retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelHit:
    entity_id: str
    video_id: str
    timestamp_ms: int
    modality: str
    score: float
    rank: int
    provider: str
    evidence_text: str | None = None


@dataclass
class FusedCandidate:
    entity_id: str
    video_id: str
    timestamp_ms: int
    final_score: float
    signal_scores: dict[str, float] = field(default_factory=dict)
    normalized_scores: dict[str, float] = field(default_factory=dict)
    evidence_texts: dict[str, str] = field(default_factory=dict)
    contributing_providers: list[str] = field(default_factory=list)
    explanation: dict[str, str | float] = field(default_factory=dict)
