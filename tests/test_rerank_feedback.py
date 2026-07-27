from __future__ import annotations

from hcmaic.retrieval.candidates import FusedCandidate
from hcmaic.retrieval.feedback import FeedbackEvent, apply_feedback
from hcmaic.retrieval.rerank import PassthroughReranker


def _candidate(entity_id: str, score: float) -> FusedCandidate:
    return FusedCandidate(
        entity_id=entity_id,
        video_id="V1",
        timestamp_ms=1000,
        final_score=score,
    )


def test_passthrough_reranker_is_stable_and_bounded():
    candidates = [_candidate("A", 0.9), _candidate("B", 0.8)]
    assert PassthroughReranker().rerank(candidates, top_k=1) == candidates[:1]


def test_feedback_deterministically_promotes_positive_and_demotes_negative():
    candidates = [_candidate("A", 0.9), _candidate("B", 0.8)]
    event = FeedbackEvent(
        session_id="s1",
        query_revision=2,
        positive_ids=["B"],
        negative_ids=["A"],
        prior_result_ids=["A", "B"],
    )
    reranked = apply_feedback(candidates, event)
    assert [item.entity_id for item in reranked] == ["B", "A"]
    assert reranked[0].explanation["feedback"] == "positive"
