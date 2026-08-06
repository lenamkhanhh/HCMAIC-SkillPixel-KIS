import json
from pathlib import Path

import cv2
import numpy as np

from hcmaic.benchmark.hybrid import (
    _reranker_manifest,
    _write_hybrid_checksums,
    benchmark_runtime_candidate,
)
from hcmaic.benchmark.skillpixel import SkillPixelBenchmarkConfig
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.retrieval.ocr_bm25 import BM25OCRChannel, OCRRecord, write_ocr_artifact
from hcmaic.runtime.kis import KISRuntime
from hcmaic.skillpixel.index import build_skillpixel_index, load_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos, validate_raw_dataset
from hcmaic.skillpixel.submission import validate_submission_csv


class _Provider(EmbeddingProvider):
    name = "test-hybrid-provider"
    version = "test-hybrid-v1"

    @property
    def dimension(self) -> int:
        return 4

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [[float(index + 1), 1.0, 0.0, 0.0] for index, _ in enumerate(paths)],
                dtype=np.float32,
            )
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(
            np.asarray([[float(len(text)), 1.0, 0.0, 0.0] for text in texts], dtype=np.float32)
        )


def _write_video(path: Path, frame_count: int = 101) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 24))
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index % 255, dtype=np.uint8))
    writer.release()


def test_runtime_fusion_exports_channel_scores_and_submission(tmp_path: Path) -> None:
    _write_video(tmp_path / "videos" / "demo.avi")
    raw_root = tmp_path / "raw"
    ingest_raw_videos(tmp_path / "videos", raw_root, stride_frames=1)
    provider = _Provider()
    index = build_skillpixel_index(raw_root, tmp_path / "index", provider)
    ocr_dir = tmp_path / "ocr"
    write_ocr_artifact(
        [
            OCRRecord(
                frame_uid="demo:000",
                video_id="demo",
                video_filename="demo.avi",
                source_frame_idx=0,
                timestamp_ms=0,
                text="bright sign",
                provider="test-real-ocr",
                revision="ocr-v1",
            )
        ],
        ocr_dir,
        dataset_manifest_hash=index.index_manifest["dataset_manifest_hash"],
    )
    runtime = KISRuntime.from_components(
        index,
        provider,
        optional_channels={"ocr": BM25OCRChannel.from_artifact(ocr_dir)},
        max_per_video=None,
    )
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "query_id,task,text,query_image\n"
        "T1,TKIS,bright sign,\n"
        "V1,VKIS,,demo.jpg\n",
        encoding="utf-8",
    )
    query_image = tmp_path / "demo.jpg"
    cv2.imwrite(str(query_image), np.zeros((24, 32, 3), dtype=np.uint8))
    corpus_path = tmp_path / "corpus.csv"
    corpus_path.write_text("video,frame_count\ndemo.avi,101\n", encoding="utf-8")
    config = SkillPixelBenchmarkConfig(
        raw_root=raw_root,
        index_dir=tmp_path / "index",
        questions_path=questions_path,
        corpus_path=corpus_path,
        output_dir=tmp_path / "out",
        top_k=100,
    )

    row = benchmark_runtime_candidate(
        config,
        runtime,
        provider,
        requested_provider=provider.name,
        selection={"provider": provider.name, "fallback": None},
        channel_status={"ocr": "ready", "object": "not_configured", "asr": "disabled"},
    )

    assert row["status"] == "validated-local"
    assert row["submission_validation"] == "pass"
    evidence_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "retrieval_evidence_top100.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(evidence_rows) == 200
    tkis_rows = [item for item in evidence_rows if item["query_id"] == "T1"]
    assert any(item["ocr_score"] is not None for item in tkis_rows)
    assert all(item["rrf_score"] is not None for item in evidence_rows)
    assert (tmp_path / "out" / "submission.csv").is_file()


def test_hybrid_checksums_exclude_manifests_written_after_benchmark(tmp_path: Path) -> None:
    (tmp_path / "stable.json").write_text("stable\n", encoding="utf-8")
    (tmp_path / "inference_manifest.json").write_text("before\n", encoding="utf-8")
    (tmp_path / "validation_final.json").write_text("before\n", encoding="utf-8")

    checksum_path = _write_hybrid_checksums(tmp_path)
    first = checksum_path.read_text(encoding="utf-8")

    (tmp_path / "inference_manifest.json").write_text("after\n", encoding="utf-8")
    (tmp_path / "validation_final.json").write_text("after\n", encoding="utf-8")

    assert checksum_path.read_text(encoding="utf-8") == first
    assert "stable.json" in first
    assert "inference_manifest.json" not in first
    assert "validation_final.json" not in first


def test_full_skillpixel_artifact_round_trip_preserves_source_mapping(tmp_path: Path) -> None:
    _write_video(tmp_path / "videos" / "roundtrip.avi", frame_count=201)
    raw_root = tmp_path / "raw"
    ingest_raw_videos(tmp_path / "videos", raw_root, stride_frames=2)
    assert validate_raw_dataset(raw_root).n_frames == 101

    provider = _Provider()
    index_dir = tmp_path / "index"
    build_skillpixel_index(raw_root, index_dir, provider)
    reloaded = load_skillpixel_index(index_dir)

    assert reloaded.faiss_index.ntotal == 101
    first_id = reloaded.id_map[0]
    first_catalog = reloaded.catalog[0]
    assert first_id["faiss_row"] == 0
    assert first_id["feature_row"] == 0
    assert first_id["frame_uid"] == first_catalog.frame_id
    assert first_id["video_id"] == "roundtrip"
    assert first_id["source_frame_idx"] == 0
    assert first_id["timestamp_ms"] == first_catalog.timestamp_ms

    query_image = tmp_path / "query.jpg"
    cv2.imwrite(str(query_image), np.zeros((24, 32, 3), dtype=np.uint8))
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "query_id,task,text,query_image\n"
        "T1,TKIS,roundtrip,\n"
        "V1,VKIS,,query.jpg\n",
        encoding="utf-8",
    )
    corpus_path = tmp_path / "corpus.csv"
    corpus_path.write_text("video,frame_count\nroundtrip.avi,201\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    config = SkillPixelBenchmarkConfig(
        raw_root=raw_root,
        index_dir=index_dir,
        questions_path=questions_path,
        corpus_path=corpus_path,
        output_dir=output_dir,
        top_k=100,
    )
    runtime = KISRuntime.from_components(reloaded, provider, max_per_video=None)
    benchmark_runtime_candidate(
        config,
        runtime,
        provider,
        requested_provider=provider.name,
        selection={"provider": provider.name, "fallback": None},
    )

    submission_report = validate_submission_csv(
        output_dir / "submission.csv", questions_path, corpus_path
    )
    evidence = [
        json.loads(line)
        for line in (output_dir / "retrieval_evidence_top20.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert submission_report.ok
    assert len(evidence) == 40
    assert all(row["faiss_row"] == row["feature_row"] for row in evidence)
    assert all(row["source_frame_idx"] >= 0 for row in evidence)
    assert all(row["timestamp_ms"] >= 0 for row in evidence)


def test_hybrid_registry_exports_real_reranker_metadata() -> None:
    class FakeReranker:
        name = "cross-encoder"

        def manifest_metadata(self) -> dict[str, object]:
            return {"model_id": "test-cross-encoder", "provider_evidence": "REAL_PROVIDER_ARTIFACT"}

    class FakeOrchestrator:
        _real_reranker = FakeReranker()

    class FakeRuntime:
        orchestrator = FakeOrchestrator()

    assert _reranker_manifest(FakeRuntime()) == {
        "status": "ready",
        "model_id": "test-cross-encoder",
        "provider_evidence": "REAL_PROVIDER_ARTIFACT",
    }
