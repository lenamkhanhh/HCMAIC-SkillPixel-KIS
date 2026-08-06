"""K7 hybrid fusion, provenance dedup, diversity and bounded rerank tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hcmaic.contracts.kis import KISQuery
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.retrieval.candidates import ChannelHit, FusedCandidate
from hcmaic.retrieval.kis_orchestrator import (
    KISHybridOrchestrator,
    bounded_rerank,
    deduplicate_source_frames,
    diversify_source_frames,
)
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos
from hcmaic.skillpixel.retrieval import SkillPixelRetriever


class _VisualProvider(EmbeddingProvider):
    name = "test-kis-visual"
    version = "test-kis-visual-v1"

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [
                    [float(int(path.stem) + 1) if path.stem.isdigit() else 2.0, 1.0, 0.0]
                    for path in paths
                ],
                dtype=np.float32,
            )
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(
            np.asarray([[float(len(text)), 1.0, 0.0] for text in texts], dtype=np.float32)
        )


class _TextChannel:
    provider = "test-ocr"
    revision = "test-ocr-v1"

    def __init__(
        self,
        *,
        frame_uid: str,
        video_id: str,
        video_filename: str,
        source_frame_idx: int,
        timestamp_ms: int,
    ) -> None:
        self.frame_uid = frame_uid
        self.video_id = video_id
        self.video_filename = video_filename
        self.source_frame_idx = source_frame_idx
        self.timestamp_ms = timestamp_ms

    def search(self, text: str, top_k: int = 100) -> list[ChannelHit]:
        del text
        return [
            ChannelHit(
                entity_id=self.frame_uid,
                video_id=self.video_id,
                timestamp_ms=self.timestamp_ms,
                modality="ocr",
                score=1.0,
                rank=1,
                provider=self.provider,
                evidence_text="hello",
                frame_uid=self.frame_uid,
                video_filename=self.video_filename,
                source_frame_idx=self.source_frame_idx,
                evidence={"normalized": True},
            )
        ][:top_k]


class _RealReranker:
    name = "cross-encoder:test"

    def rerank(
        self,
        candidates: list[FusedCandidate],
        *,
        query_text: str | None,
        top_k: int,
        candidate_limit: int,
        timeout_ms: int,
    ) -> list[FusedCandidate]:
        assert query_text == "hello"
        assert candidate_limit >= top_k
        assert timeout_ms > 0
        for candidate in candidates[:candidate_limit]:
            candidate.rerank_score = candidate.final_score + 1.0
        return sorted(
            candidates[:candidate_limit],
            key=lambda candidate: -(candidate.rerank_score or candidate.final_score),
        )[:top_k]


def _make_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
    assert writer.isOpened()
    for idx in range(8):
        writer.write(np.full((48, 64, 3), idx * 20, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def visual_retriever(tmp_path: Path) -> SkillPixelRetriever:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    raw_root = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw_root, stride_frames=2)
    artifact_dir = tmp_path / "index"
    provider = _VisualProvider()
    build_skillpixel_index(raw_root, artifact_dir, provider)
    return SkillPixelRetriever.from_artifacts(artifact_dir, provider=provider)


def _candidate(
    entity_id: str,
    video_id: str,
    source_frame_idx: int,
    score: float,
) -> FusedCandidate:
    return FusedCandidate(
        entity_id=entity_id,
        video_id=video_id,
        timestamp_ms=source_frame_idx * 100,
        final_score=score,
        frame_uid=entity_id,
        video_filename=f"{video_id}.mp4",
        source_frame_idx=source_frame_idx,
    )


def test_dedup_merges_same_source_frame_and_preserves_evidence():
    first = _candidate("frame-a", "V1", 10, 0.2)
    first.evidence["visual"] = {"score": 0.2}
    second = _candidate("frame-b", "V1", 10, 0.3)
    second.evidence["ocr"] = {"score": 0.3}

    deduped = deduplicate_source_frames([first, second])

    assert len(deduped) == 1
    assert deduped[0].final_score == pytest.approx(0.5)
    assert set(deduped[0].evidence) == {"visual", "ocr"}


def test_diversity_quota_then_backfill_and_rerank_bound():
    candidates = [_candidate(f"f{i}", "V1" if i < 3 else "V2", i, 3.0 - i) for i in range(5)]

    diversified = diversify_source_frames(candidates, top_k=4, max_per_video=2)
    reranked = bounded_rerank(diversified, top_k=3, candidate_limit=4, timeout_ms=50)

    assert [candidate.video_id for candidate in diversified[:4]] == ["V1", "V1", "V2", "V2"]
    assert len(reranked) == 3
    assert all(candidate.rerank_score is not None for candidate in reranked)


def test_hybrid_search_fuses_visual_and_marks_unavailable_channels(
    visual_retriever: SkillPixelRetriever,
):
    orchestrator = KISHybridOrchestrator(
        visual_retriever,
        optional_channels={
            "ocr": _TextChannel(
                frame_uid=visual_retriever.index.catalog[0].frame_id,
                video_id=visual_retriever.index.catalog[0].video_id,
                video_filename=visual_retriever.index.catalog[0].video_filename or "demo.avi",
                source_frame_idx=visual_retriever.index.catalog[0].source_frame_idx or 0,
                timestamp_ms=visual_retriever.index.catalog[0].timestamp_ms,
            )
        },
        max_per_video=2,
    )
    output = orchestrator.search(KISQuery("Q1", "TKIS", text="hello", top_k=3))

    assert output.results
    assert output.executed_channels == ("visual", "ocr")
    assert output.results[0].query_id == "Q1"
    assert "visual" in output.results[0].channel_scores
    assert any("ocr" in result.channel_scores for result in output.results)
    assert any(result.evidence for result in output.results)
    assert output.results[0].quality_status == "UNVALIDATED_ON_HCMAIC"
    assert "object" in output.unavailable_channels
    assert "asr" in output.unavailable_channels


def test_vkis_skips_text_channels(visual_retriever: SkillPixelRetriever, tmp_path: Path):
    query_image = tmp_path / "query.jpg"
    cv2.imwrite(str(query_image), np.full((48, 64, 3), 40, dtype=np.uint8))
    orchestrator = KISHybridOrchestrator(
        visual_retriever,
        optional_channels={"ocr": None},
    )

    output = orchestrator.search(KISQuery("Q2", "VKIS", image_path=query_image, top_k=2))

    assert output.executed_channels == ("visual",)
    assert output.unavailable_channels["ocr"] == "not_applicable_to_vkis_visual_query"


def test_orchestrator_executes_configured_real_reranker(
    visual_retriever: SkillPixelRetriever,
):
    orchestrator = KISHybridOrchestrator(
        visual_retriever,
        reranker=_RealReranker(),
    )

    output = orchestrator.search(KISQuery("Q3", "TKIS", text="hello", top_k=2))

    assert orchestrator.reranker == "cross-encoder:test"
    assert all(result.rerank_score is not None for result in output.results)
