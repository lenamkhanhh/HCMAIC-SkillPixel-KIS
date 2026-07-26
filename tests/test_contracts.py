"""Contract validation tests."""

import pytest
from pydantic import ValidationError

from hcmaic.contracts.models import (
    CanonicalSubmission,
    FrameRecord,
    SearchRequest,
    make_frame_id,
)


def test_make_frame_id():
    assert make_frame_id("L01_V001", "001") == "L01_V001:001"


def test_frame_record_rejects_negative_frame_idx():
    with pytest.raises(ValidationError):
        FrameRecord(
            frame_id="v:001",
            video_id="v",
            keyframe_id="001",
            frame_idx=-1,
            timestamp_ms=0,
            image_path="keyframes/v/001.jpg",
        )


def test_frame_record_rejects_negative_timestamp():
    with pytest.raises(ValidationError):
        FrameRecord(
            frame_id="v:001",
            video_id="v",
            keyframe_id="001",
            frame_idx=0,
            timestamp_ms=-5,
            image_path="keyframes/v/001.jpg",
        )


def test_search_request_defaults():
    req = SearchRequest(query_id="q1", text="red")
    assert req.top_k == 10
    assert req.task_type == "kis"
    assert req.filters == {}


@pytest.mark.parametrize("text", ["", "   "])
def test_search_request_rejects_blank_text(text):
    with pytest.raises(ValidationError):
        SearchRequest(query_id="q1", text=text)


@pytest.mark.parametrize("top_k", [0, -3, 501])
def test_search_request_top_k_bounds(top_k):
    with pytest.raises(ValidationError):
        SearchRequest(query_id="q1", text="red", top_k=top_k)


def test_submission_confidence_bounds():
    with pytest.raises(ValidationError):
        CanonicalSubmission(
            query_id="q",
            task_type="kis",
            video_id="v",
            frame_id="v:001",
            timestamp_ms=0,
            confidence=1.5,
        )
