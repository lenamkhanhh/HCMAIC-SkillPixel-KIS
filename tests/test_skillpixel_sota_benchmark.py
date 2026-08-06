"""SkillPixel benchmark matrix and provenance tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from hcmaic.benchmark.skillpixel import (
    SkillPixelBenchmarkConfig,
    benchmark_visual_candidate,
    unavailable_candidate_row,
    write_benchmark_outputs,
)
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos


class _TinyProvider(EmbeddingProvider):
    name = "test-real-shape"
    version = "test-real-shape-v1"

    @property
    def dimension(self) -> int:
        return 4

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [[float(int(path.stem) + 1), 1.0, 0.0, 0.0] for path in paths],
                dtype=np.float32,
            )
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(
            np.asarray([[float(len(text)), 1.0, 0.0, 0.0] for text in texts], dtype=np.float32)
        )


def _write_video(path: Path, frame_count: int = 210) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
    assert writer.isOpened()
    for idx in range(frame_count):
        writer.write(np.full((48, 64, 3), idx % 255, dtype=np.uint8))
    writer.release()
    return path


def _config(tmp_path: Path, raw: Path, index: Path) -> SkillPixelBenchmarkConfig:
    questions = tmp_path / "questions.csv"
    questions.write_text(
        "query_id,task,text,query_image\n"
        f"T1,TKIS,bright,\nV1,VKIS,,{(raw / 'keyframes' / 'demo' / '000.jpg').resolve()}\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("video,frame_count\ndemo.avi,210\n", encoding="utf-8")
    return SkillPixelBenchmarkConfig(
        raw_root=raw,
        index_dir=index,
        questions_path=questions,
        corpus_path=corpus,
        output_dir=tmp_path / "benchmark",
        top_k=100,
    )


def test_visual_candidate_writes_submission_and_provenance(tmp_path: Path):
    source = _write_video(tmp_path / "raw" / "demo.avi")
    raw = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw, stride_frames=2)
    index_dir = tmp_path / "index"
    provider = _TinyProvider()
    build_skillpixel_index(raw, index_dir, provider)

    row = benchmark_visual_candidate(
        _config(tmp_path, raw, index_dir),
        provider,
        requested_provider=provider.name,
        selection={"provider": provider.name, "fallback": None},
    )

    assert row["status"] == "validated-local"
    assert row["mapping_errors"] == 0
    assert row["submission_validation"] == "pass"
    assert row["metrics"]["recall@100"] is None
    assert (tmp_path / "benchmark" / "submission.csv").is_file()
    assert (tmp_path / "benchmark" / "validation.json").is_file()
    evidence_path = tmp_path / "benchmark" / "retrieval_evidence_top100.jsonl"
    evidence_rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert evidence_path.is_file()
    assert (tmp_path / "benchmark" / "retrieval_evidence_top20.jsonl").is_file()
    assert (tmp_path / "benchmark" / "retrieval_evidence_top100.csv").is_file()
    assert (tmp_path / "benchmark" / "query_status.jsonl").is_file()
    assert (tmp_path / "benchmark" / "preflight_report.json").is_file()
    assert (tmp_path / "benchmark" / "model_registry.json").is_file()
    assert (tmp_path / "benchmark" / "checksums.sha256").is_file()
    assert len(evidence_rows) == 2 * 100
    assert {
        "query_id",
        "query_type",
        "query_order",
        "rank",
        "video_id",
        "video_filename",
        "keyframe_id",
        "frame_uid",
        "source_frame_idx",
        "timestamp_ms",
        "preview_path",
        "visual_score",
        "ocr_score",
        "object_score",
        "asr_score",
        "rrf_score",
        "rerank_score",
        "provider",
        "model",
        "revision",
        "faiss_row",
        "feature_row",
    }.issubset(evidence_rows[0])
    status_rows = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "query_status.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["query_id"] for row in status_rows] == ["T1", "V1"]
    assert all(row["status"] == "ok" for row in status_rows)


def test_unavailable_candidate_never_reports_fake_quality():
    row = unavailable_candidate_row(
        variant="V1",
        requested_provider="siglip2",
        error="weights are not cached",
        dataset_hash="dataset-hash",
        query_file_hash="query-hash",
    )

    assert row["status"] == "unavailable"
    assert row["provider_execution"] == "unavailable"
    assert row["metrics"] == {}
    assert row["official_skillpixel_score"] is None
    assert "weights" in row["error"]


def test_benchmark_outputs_keep_matrix_rows_and_quality_status(tmp_path: Path):
    rows = [
        unavailable_candidate_row(
            variant="V1",
            requested_provider="siglip2",
            error="not cached",
            dataset_hash="dataset-hash",
            query_file_hash="query-hash",
        )
    ]
    config = SkillPixelBenchmarkConfig(
        raw_root=tmp_path / "raw",
        index_dir=tmp_path / "index",
        questions_path=tmp_path / "questions.csv",
        corpus_path=tmp_path / "corpus.csv",
        output_dir=tmp_path / "out",
    )

    paths = write_benchmark_outputs(config, rows, promotion_decision="retain V0")

    assert paths["csv"].is_file()
    assert paths["report"].is_file()
    assert paths["manifest"].is_file()
    matrix = list(csv.DictReader(paths["csv"].open(encoding="utf-8", newline="")))
    assert matrix[0]["status"] == "unavailable"
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["quality_status"] == "UNVALIDATED_ON_SKILLPIXEL_QRELS"
    assert "retain V0" in paths["report"].read_text(encoding="utf-8")
