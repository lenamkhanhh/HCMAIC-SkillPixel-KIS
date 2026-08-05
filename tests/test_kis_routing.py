"""K3 canonical TKIS/VKIS routing tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hcmaic.contracts.kis import KISQuery
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos
from hcmaic.skillpixel.retrieval import SkillPixelRetriever


class _RoutingSpyProvider(EmbeddingProvider):
    name = "test-routing"
    version = "test-routing-v1"

    def __init__(self) -> None:
        self.text_batches: list[list[str]] = []
        self.image_batches: list[list[Path]] = []

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        self.image_batches.append(list(paths))
        return l2_normalize(
            np.asarray(
                [[float(int(path.stem) + 1) if path.stem.isdigit() else 2.0, 1.0, 0.0]
                 for path in paths],
                dtype=np.float32,
            )
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.text_batches.append(list(texts))
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
def routing_fixture(tmp_path: Path) -> tuple[SkillPixelRetriever, _RoutingSpyProvider, Path]:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    raw_root = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw_root, stride_frames=2)
    provider = _RoutingSpyProvider()
    artifact_dir = tmp_path / "index"
    index = build_skillpixel_index(raw_root, artifact_dir, provider)
    provider.text_batches.clear()
    provider.image_batches.clear()
    query_image = raw_root / index.catalog[0].image_path
    return (
        SkillPixelRetriever.from_artifacts(artifact_dir, provider=provider),
        provider,
        query_image,
    )


def test_mixed_kis_queries_use_both_towers_and_preserve_order(
    routing_fixture: tuple[SkillPixelRetriever, _RoutingSpyProvider, Path]
):
    retriever, provider, query_image = routing_fixture
    queries = [
        KISQuery("V1", "VKIS", image_path=query_image, top_k=2),
        KISQuery("T1", "TKIS", text="longer text", raw_text="  longer text  ", top_k=3),
    ]

    results = retriever.search_kis_queries(queries)

    assert list(results) == ["V1", "T1"]
    assert provider.text_batches == [["longer text"]]
    assert provider.image_batches == [[query_image]]
    assert len(results["V1"]) == 2
    assert len(results["T1"]) == 3
    assert all(result.task == "VKIS" for result in results["V1"])
    assert all(result.task == "TKIS" for result in results["T1"])
    assert all(result.executed_channels == ("visual",) for result in results["T1"])
    assert all(set(result.channel_scores) == {"visual"} for result in results["T1"])
    assert all(
        result.answer_cell.endswith(str(result.source_frame_idx)) for result in results["T1"]
    )
    assert results["T1"][0].evidence[0].metadata["faiss_row"] >= 0


def test_single_kis_query_routes_by_task(
    routing_fixture: tuple[SkillPixelRetriever, _RoutingSpyProvider, Path]
):
    retriever, provider, query_image = routing_fixture
    result = retriever.search_kis(KISQuery("V1", "VKIS", image_path=query_image, top_k=1))

    assert result[0].query_id == "V1"
    assert result[0].task == "VKIS"
    assert provider.text_batches == []
    assert provider.image_batches == [[query_image]]
