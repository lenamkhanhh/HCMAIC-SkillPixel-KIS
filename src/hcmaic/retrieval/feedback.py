"""Session feedback contract and a deterministic non-learned reranking path."""

from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, Field

from hcmaic.retrieval.candidates import FusedCandidate


class FeedbackEvent(BaseModel):
    session_id: str = Field(min_length=1)
    query_revision: int = Field(ge=1)
    positive_ids: list[str] = Field(default_factory=list)
    negative_ids: list[str] = Field(default_factory=list)
    prior_result_ids: list[str] = Field(default_factory=list)


def apply_feedback(
    candidates: list[FusedCandidate],
    event: FeedbackEvent,
    *,
    positive_boost: float = 1.0,
    negative_penalty: float = 1.0,
) -> list[FusedCandidate]:
    positives = set(event.positive_ids)
    negatives = set(event.negative_ids)
    adjusted = deepcopy(candidates)
    for candidate in adjusted:
        if candidate.entity_id in positives:
            candidate.final_score += positive_boost
            candidate.explanation["feedback"] = "positive"
        elif candidate.entity_id in negatives:
            candidate.final_score -= negative_penalty
            candidate.explanation["feedback"] = "negative"
    return sorted(adjusted, key=lambda item: (-item.final_score, item.entity_id))
