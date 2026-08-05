"""TKIS text routing and query-id preservation tests."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos
from hcmaic.skillpixel.retrieval import (
    SkillPixelRetriever,
    load_skillpixel_questions,
)


class _TextSpyProvider(EmbeddingProvider):
    name = "test-text"
    version = "test-text-v1"

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.image_calls = 0

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        self.image_calls += 1
        return l2_normalize(
            np.asarray([[float(int(path.stem) + 1), 1.0, 0.0] for path in paths], dtype=np.float32)
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls.append(list(texts))
        return l2_normalize(
            np.asarray([[float(len(text)), 1.0, 0.0] for text in texts], dtype=np.float32)
        )


def _make_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
    assert writer.isOpened()
    for idx in range(8):
        writer.write(np.full((48, 64, 3), idx * 20, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def retriever(tmp_path: Path) -> SkillPixelRetriever:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    raw_root = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw_root, stride_frames=2)
    provider = _TextSpyProvider()
    artifacts = tmp_path / "artifacts"
    build_skillpixel_index(raw_root, artifacts, provider)
    return SkillPixelRetriever.from_artifacts(artifacts, provider=provider)


def test_questions_adapter_preserves_tkis_rows(tmp_path: Path):
    path = tmp_path / "questions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "task", "text", "query_image"])
        writer.writeheader()
        writer.writerow({"query_id": "Q1", "task": "TKIS", "text": "red object"})
        writer.writerow({"query_id": "Q2", "task": "VKIS", "query_image": "frames/Q2.jpg"})

    questions = load_skillpixel_questions(path)

    assert [(item.query_id, item.task, item.text) for item in questions] == [
        ("Q1", "TKIS", "red object"),
        ("Q2", "VKIS", ""),
    ]


def test_tkis_batch_uses_text_encoder_and_maps_source_frame(retriever: SkillPixelRetriever):
    results = retriever.search_text_queries(
        [("Q1", "short"), ("Q2", "a much longer text")], top_k=3
    )

    assert retriever.provider.text_calls == [["short", "a much longer text"]]
    assert retriever.provider.image_calls == 1
    assert list(results) == ["Q1", "Q2"]
    assert [hit.rank for hit in results["Q1"]] == [1, 2, 3]
    assert results["Q1"][0].video_filename == "demo.avi"
    assert results["Q1"][0].source_frame_idx in {0, 2, 4, 6}
    assert results["Q1"][0].frame_uid


def test_tkis_rejects_provider_version_mismatch(retriever: SkillPixelRetriever):
    mismatched = _TextSpyProvider()
    mismatched.version = "other-v1"
    with pytest.raises(ValueError, match="version"):
        SkillPixelRetriever.from_artifacts(
            retriever.index.artifact_dir, provider=mismatched
        )
