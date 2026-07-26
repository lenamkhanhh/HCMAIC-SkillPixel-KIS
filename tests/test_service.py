"""RetrievalService tests: search, filters, timeline, neighbors, preview."""

from pathlib import Path

import pytest

from hcmaic.contracts.models import SearchRequest
from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider
from hcmaic.indexing.artifacts import load_index_artifacts
from hcmaic.retrieval.service import (
    RetrievalService,
    UnknownFrameError,
    UnknownVideoError,
)


def _req(text: str, **kw) -> SearchRequest:
    return SearchRequest(query_id="t", text=text, **kw)


def test_search_result_shape(service: RetrievalService):
    results = service.search(_req("a solid red keyframe", top_k=5))
    assert len(results) == 5
    top = results[0]
    assert top.frame_id == "L01_V001:001"  # the red keyframe
    assert top.rank == 1
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert top.video_id == "L01_V001"
    assert top.timestamp_ms == 1000
    assert top.image_url == "/frames/L01_V001:001/image"
    assert top.signal_scores == {"visual": top.final_score}
    assert top.index_version == service.index_version
    assert top.evidence["keyframe_id"] == "001"
    # scores are non-increasing
    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_is_deterministic(service: RetrievalService):
    first = service.search(_req("yellow frame", top_k=10))
    second = service.search(_req("yellow frame", top_k=10))
    assert [r.frame_id for r in first] == [r.frame_id for r in second]


def test_top_k_larger_than_corpus(service: RetrievalService):
    results = service.search(_req("blue", top_k=500))
    assert len(results) == 12  # whole fixture, no padding, no crash


def test_video_filter_list(service: RetrievalService):
    results = service.search(
        _req("blue", filters={"video_ids": ["L01_V004"]}, top_k=10)
    )
    assert results, "filtered search should still find frames"
    assert {r.video_id for r in results} == {"L01_V004"}


def test_video_filter_comma_string(service: RetrievalService):
    results = service.search(
        _req("blue", filters={"video_ids": "L01_V001, L01_V002"}, top_k=20)
    )
    assert {r.video_id for r in results} == {"L01_V001", "L01_V002"}


def test_video_filter_unknown_video_empty(service: RetrievalService):
    results = service.search(_req("blue", filters={"video_ids": ["NOPE"]}))
    assert results == []


def test_video_filter_bad_type(service: RetrievalService):
    with pytest.raises(ValueError, match="video_ids"):
        service.search(_req("blue", filters={"video_ids": 42}))


def test_timeline_ordering(service: RetrievalService):
    frames = service.timeline("L01_V001")
    assert [f.frame_id for f in frames] == [
        "L01_V001:001", "L01_V001:002", "L01_V001:003",
    ]
    timestamps = [f.timestamp_ms for f in frames]
    assert timestamps == sorted(timestamps)


def test_timeline_unknown_video(service: RetrievalService):
    with pytest.raises(UnknownVideoError):
        service.timeline("NOPE")


def test_neighbors_window(service: RetrievalService):
    neighbors = service.neighbors("L01_V001:002", window=1)
    assert [n.frame_id for n in neighbors] == [
        "L01_V001:001", "L01_V001:002", "L01_V001:003",
    ]
    # window larger than video is clamped
    assert len(service.neighbors("L01_V003:001", window=50)) == 2


def test_get_frame_unknown(service: RetrievalService):
    with pytest.raises(UnknownFrameError):
        service.get_frame("L01_V001:999")


def test_frame_image_path_inside_root(service: RetrievalService, sample_root: Path):
    path = service.frame_image_path("L01_V001:001")
    assert path.is_file()
    assert str(path).startswith(str(sample_root.resolve()))


def test_submission_preview(service: RetrievalService):
    preview = service.submission_preview(
        query_id="q1", task_type="kis", frame_id="L01_V002:003",
        answer="white wall", confidence=0.8,
    )
    assert preview.video_id == "L01_V002"
    assert preview.timestamp_ms == 9000
    assert preview.answer == "white wall"
    assert preview.evidence["frame_idx"] == 225
    assert preview.evidence["index_version"] == service.index_version


def test_provider_dimension_gate(built_artifacts_dir: Path, sample_root: Path):
    artifacts = load_index_artifacts(built_artifacts_dir)
    wrong = DeterministicMockEmbeddingProvider(dimension=32)
    with pytest.raises(ValueError, match="dimension"):
        RetrievalService(artifacts, text_provider=wrong, dataset_root=sample_root)


def test_provider_version_gate(built_artifacts_dir: Path, sample_root: Path):
    artifacts = load_index_artifacts(built_artifacts_dir)
    wrong = DeterministicMockEmbeddingProvider()
    wrong.version = "mock-palette-incompatible"
    with pytest.raises(ValueError, match="version"):
        RetrievalService(artifacts, text_provider=wrong, dataset_root=sample_root)
