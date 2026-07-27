from hcmaic.contracts.models import FrameRecord
from hcmaic.retrieval.temporal import expand_temporal


def _frame(frame_id: str, timestamp_ms: int, shot_id: str) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        video_id="V1",
        keyframe_id=frame_id.split(":")[-1],
        frame_idx=timestamp_ms // 40,
        timestamp_ms=timestamp_ms,
        shot_id=shot_id,
        image_path=f"{frame_id}.jpg",
    )


def test_temporal_expansion_includes_window_and_same_shot_with_stable_dedup():
    frames = [
        _frame("V1:001", 1000, "S1"),
        _frame("V1:002", 2000, "S1"),
        _frame("V1:003", 5000, "S2"),
    ]
    expanded = expand_temporal(
        ["V1:002"], frames, window_ms=500, include_same_shot=True
    )
    assert expanded == ["V1:002", "V1:001"]
