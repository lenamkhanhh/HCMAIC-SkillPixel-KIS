"""K2 visual benchmark tests with a local deterministic provider."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hcmaic.benchmark.kis import (
    KISBenchmarkError,
    benchmark_visual_retrieval,
    write_visual_benchmark_report,
)
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos
from hcmaic.skillpixel.retrieval import SkillPixelQuestion, SkillPixelRetriever


class _TinyRealShapeProvider(EmbeddingProvider):
    """Deterministic test provider; this is never selected by production config."""

    name = "test-real-shape"
    version = "test-real-shape-v1"

    @property
    def dimension(self) -> int:
        return 4

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        rows = []
        for path in paths:
            value = int(path.stem)
            rows.append([float(value + 1), 1.0, 0.0, 0.0])
        return l2_normalize(np.asarray(rows, dtype=np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [[float(len(text)), 1.0, 0.0, 0.0] for text in texts], dtype=np.float32
            )
        )


def _make_video(path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48)
    )
    assert writer.isOpened()
    for idx in range(8):
        writer.write(np.full((48, 64, 3), idx * 20, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def retriever_and_query_root(tmp_path: Path) -> tuple[SkillPixelRetriever, Path]:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    raw_dir = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw_dir, stride_frames=2)
    artifact_dir = tmp_path / "index"
    provider = _TinyRealShapeProvider()
    index = build_skillpixel_index(raw_dir, artifact_dir, provider)
    image_path = raw_dir / index.catalog[0].image_path
    return SkillPixelRetriever(index, provider), image_path


def test_visual_benchmark_measures_both_tasks_without_qrels(
    retriever_and_query_root: tuple[SkillPixelRetriever, Path], tmp_path: Path
):
    retriever, image_path = retriever_and_query_root
    questions = [
        SkillPixelQuestion("T1", "TKIS", "a short text", ""),
        SkillPixelQuestion("V1", "VKIS", "", str(image_path)),
    ]

    report = benchmark_visual_retrieval(retriever, questions, top_k=4)

    assert report.n_queries == 2
    assert report.n_tkis == 1
    assert report.n_vkis == 1
    assert report.index["index_provider"] == "faiss-flat-ip"
    assert report.index["exact"] is True
    assert report.provider["provider"] == "test-real-shape"
    assert report.latency_ms["total_batch"] is not None
    assert report.metrics["recall@1"] is None
    assert report.metrics["mrr"] is None
    assert report.quality_status == "UNVALIDATED_ON_HCMAIC"

    output = write_visual_benchmark_report(report, tmp_path / "benchmark.json")
    assert output.is_file()
    assert '"qrels_present": false' in output.read_text(encoding="utf-8")


def test_visual_benchmark_requires_explicit_qrels_source(
    retriever_and_query_root: tuple[SkillPixelRetriever, Path]
):
    retriever, image_path = retriever_and_query_root
    question = SkillPixelQuestion("T1", "TKIS", "a short text", "")

    with pytest.raises(KISBenchmarkError, match="qrels_source"):
        benchmark_visual_retrieval(
            retriever,
            [question],
            qrels={"T1": "demo.avi,0"},
        )
