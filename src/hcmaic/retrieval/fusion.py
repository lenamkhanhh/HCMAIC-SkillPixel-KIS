"""Deterministic Reciprocal Rank Fusion and weighted late fusion."""

from __future__ import annotations

from hcmaic.retrieval.candidates import ChannelHit, FusedCandidate


def _candidate_for(hit: ChannelHit) -> FusedCandidate:
    return FusedCandidate(
        entity_id=hit.entity_id,
        video_id=hit.video_id,
        timestamp_ms=hit.timestamp_ms,
        final_score=0.0,
    )


def _record_hit(
    candidate: FusedCandidate, modality: str, hit: ChannelHit, normalized: float
) -> None:
    candidate.signal_scores[modality] = hit.score
    candidate.normalized_scores[modality] = normalized
    if hit.evidence_text:
        candidate.evidence_texts[modality] = hit.evidence_text
    if hit.provider not in candidate.contributing_providers:
        candidate.contributing_providers.append(hit.provider)


def _rank(candidates: dict[str, FusedCandidate], top_k: int) -> list[FusedCandidate]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    return sorted(
        candidates.values(), key=lambda item: (-item.final_score, item.entity_id)
    )[:top_k]


def reciprocal_rank_fusion(
    channels: dict[str, list[ChannelHit]], *, rank_constant: int = 60, top_k: int = 100
) -> list[FusedCandidate]:
    if rank_constant < 1:
        raise ValueError("rank_constant must be >= 1")
    candidates: dict[str, FusedCandidate] = {}
    for modality, hits in channels.items():
        for hit in hits:
            if hit.rank < 1:
                raise ValueError("hit rank must be >= 1")
            candidate = candidates.setdefault(hit.entity_id, _candidate_for(hit))
            contribution = 1.0 / (rank_constant + hit.rank)
            candidate.final_score += contribution
            _record_hit(candidate, modality, hit, contribution)
    for candidate in candidates.values():
        candidate.explanation = {
            "method": "rrf",
            "rank_constant": float(rank_constant),
        }
    return _rank(candidates, top_k)


def weighted_late_fusion(
    channels: dict[str, list[ChannelHit]],
    *,
    weights: dict[str, float],
    top_k: int = 100,
) -> list[FusedCandidate]:
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("fusion weight must be >= 0")
    candidates: dict[str, FusedCandidate] = {}
    for modality, hits in channels.items():
        weight = weights.get(modality, 0.0)
        if not hits or weight == 0:
            continue
        values = [hit.score for hit in hits]
        low, high = min(values), max(values)
        for hit in hits:
            normalized = 1.0 if high == low else (hit.score - low) / (high - low)
            candidate = candidates.setdefault(hit.entity_id, _candidate_for(hit))
            candidate.final_score += weight * normalized
            _record_hit(candidate, modality, hit, normalized)
    for candidate in candidates.values():
        candidate.explanation = {"method": "weighted-late-fusion"}
    return _rank(candidates, top_k)
