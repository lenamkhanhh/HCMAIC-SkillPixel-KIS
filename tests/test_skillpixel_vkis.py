"""VKIS image-tower routing tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos
from hcmaic.skillpixel.retrieval import SkillPixelRetriever


class _ImageSpyProvider(EmbeddingProvider):
    name = "test-image"
    version = "test-image-v1"

    def __init__(self) -> None:
        self.image_batches: list[list[Path]] = []
        self.query_image_calls: list[Path] = []
        self.text_calls = 0

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        self.image_batches.append(list(paths))
        return l2_normalize(
            np.asarray(
                [
                    [float(int(path.stem) + 1) if path.stem.isdigit() else 2.0, 1.0, 0.0]
                    for path in paths
                ],
                dtype=np.float32,
            )
        )

    def embed_query_image(self, path: Path) -> np.ndarray:
        self.query_image_calls.append(path)
        return self.embed_images([path])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        return l2_normalize(np.ones((len(texts), 3), dtype=np.float32))


def _make_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
    assert writer.isOpened()
    for idx in range(8):
        writer.write(np.full((48, 64, 3), idx * 20, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def image_retriever(tmp_path: Path) -> tuple[SkillPixelRetriever, _ImageSpyProvider, Path]:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    raw_root = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw_root, stride_frames=2)
    provider = _ImageSpyProvider()
    artifact_dir = tmp_path / "artifacts"
    build_skillpixel_index(raw_root, artifact_dir, provider)
    query = tmp_path / "query.jpg"
    cv2.imwrite(str(query), np.full((48, 64, 3), 60, dtype=np.uint8))
    provider.image_batches.clear()
    return SkillPixelRetriever.from_artifacts(artifact_dir, provider=provider), provider, query


def test_vkis_single_image_uses_image_query_tower(
    image_retriever: tuple[SkillPixelRetriever, _ImageSpyProvider, Path]
):
    retriever, provider, query = image_retriever

    hits = retriever.search_image("Q_0051", query, top_k=3)

    assert provider.query_image_calls == [query]
    assert provider.text_calls == 0
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert all(hit.task == "VKIS" for hit in hits)
    assert all(hit.source_frame_idx in {0, 2, 4, 6} for hit in hits)


def test_vkis_batch_preserves_query_ids_and_uses_same_visual_index(
    image_retriever: tuple[SkillPixelRetriever, _ImageSpyProvider, Path], tmp_path: Path
):
    retriever, provider, query = image_retriever
    query_2 = tmp_path / "query_2.jpg"
    cv2.imwrite(str(query_2), np.full((48, 64, 3), 100, dtype=np.uint8))

    results = retriever.search_image_queries(
        [("Q_0051", query), ("Q_0052", query_2)], top_k=2
    )

    assert list(results) == ["Q_0051", "Q_0052"]
    assert provider.query_image_calls == []
    assert len(provider.image_batches) == 1
    assert provider.image_batches[0] == [query, query_2]
    assert all(len(hits) == 2 for hits in results.values())
