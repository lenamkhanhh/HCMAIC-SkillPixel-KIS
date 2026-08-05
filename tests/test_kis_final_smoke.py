"""K10 compact raw -> index -> TKIS/VKIS -> CSV final smoke test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from hcmaic.contracts.kis import KISQuery
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.runtime.kis import KISRuntime
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos, validate_raw_dataset
from hcmaic.skillpixel.submission import export_skillpixel_submission, validate_submission_csv


class _FinalSmokeProvider(EmbeddingProvider):
    name = "test-final-smoke-provider"
    version = "test-final-smoke-v1"

    @property
    def dimension(self) -> int:
        return 4

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [
                    [float(int(path.stem) + 1) if path.stem.isdigit() else 2.0, 1.0, 0.0, 0.0]
                    for path in paths
                ],
                dtype=np.float32,
            )
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [[float(len(text)), 1.0, 0.0, 0.0] for text in texts], dtype=np.float32
            )
        )


def _make_video(path: Path, frame_count: int = 101) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
    assert writer.isOpened()
    for idx in range(frame_count):
        writer.write(np.full((48, 64, 3), idx % 255, dtype=np.uint8))
    writer.release()
    return path


def test_final_kis_smoke_raw_to_validated_submission(tmp_path: Path):
    raw_video = _make_video(tmp_path / "videos" / "demo.avi")
    raw_root = tmp_path / "raw-generated"
    ingest_raw_videos(raw_video.parent, raw_root, stride_frames=1)
    raw_stats = validate_raw_dataset(raw_root)
    provider = _FinalSmokeProvider()
    index = build_skillpixel_index(raw_root, tmp_path / "index", provider)
    runtime = KISRuntime.from_components(index, provider, max_per_video=None)
    query_image = tmp_path / "query.jpg"
    cv2.imwrite(str(query_image), np.full((48, 64, 3), 30, dtype=np.uint8))

    queries = [
        KISQuery("T1", "TKIS", text="a frame", top_k=100),
        KISQuery("V1", "VKIS", image_path=query_image, top_k=100),
    ]
    outputs = runtime.search_queries(queries)
    assert raw_stats.n_videos == 1
    assert raw_stats.n_frames == 101
    assert all(len(outputs[query.query_id].results) == 100 for query in queries)
    assert all(
        "keyframe_id" not in result.to_dict()
        for output in outputs.values()
        for result in output.results
    )

    questions_path = tmp_path / "questions.csv"
    with questions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "task", "text", "query_image"])
        writer.writeheader()
        writer.writerow({"query_id": "T1", "task": "TKIS", "text": "a frame", "query_image": ""})
        writer.writerow({"query_id": "V1", "task": "VKIS", "text": "", "query_image": "query.jpg"})
    corpus_path = tmp_path / "corpus.csv"
    with corpus_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "frame_count"])
        writer.writeheader()
        writer.writerow({"video": "demo.avi", "frame_count": "101"})
    results_path = tmp_path / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(
                json.dumps(
                    {
                        "query_id": query.query_id,
                        "answers": [result.to_dict() for result in outputs[query.query_id].results],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    submission_path = tmp_path / "submission.csv"
    stats = export_skillpixel_submission(
        questions_path, results_path, corpus_path, submission_path
    )
    validation = validate_submission_csv(submission_path, questions_path, corpus_path)

    assert stats.n_queries == 2
    assert stats.answers_per_query == 100
    assert validation.ok, validation.errors
    assert len(submission_path.read_text(encoding="utf-8").splitlines()) == 3
