"""K9 qrels/evaluator and ablation tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.evaluation.kis import (
    KISEvaluationError,
    KISQrel,
    KISQrelSet,
    evaluate_kis_runtime,
    load_kis_qrels,
    run_kis_ablation,
)
from hcmaic.runtime.kis import KISRuntime
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos
from hcmaic.skillpixel.retrieval import SkillPixelQuestion


class _EvalProvider(EmbeddingProvider):
    name = "test-eval-provider"
    version = "test-eval-v1"

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [
                    [
                        float(int(path.stem) + 1) if path.stem.isdigit() else 2.0,
                        1.0,
                        0.0,
                    ]
                    for path in paths
                ],
                dtype=np.float32,
            )
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(
            np.asarray([[float(len(text)), 1.0, 0.0] for text in texts], dtype=np.float32)
        )


def _make_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
    assert writer.isOpened()
    for idx in range(12):
        writer.write(np.full((48, 64, 3), idx * 20, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def eval_runtime(tmp_path: Path) -> KISRuntime:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    raw_root = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw_root, stride_frames=2)
    provider = _EvalProvider()
    index = build_skillpixel_index(raw_root, tmp_path / "index", provider)
    return KISRuntime.from_components(index, provider)


def test_qrels_adapter_preserves_answer_cells_without_official_claim(tmp_path: Path):
    path = tmp_path / "qrels.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "Q1",
                "relevant_answers": [{"video_filename": "demo.avi", "source_frame_idx": 2}],
                "relevant_frame_uids": ["demo:000001"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qrels = load_kis_qrels(path)
    assert qrels.qrels["Q1"].relevant_answer_cells == frozenset({"demo.avi,2"})
    assert qrels.quality_status == "UNVALIDATED_ON_HCMAIC"


def test_evaluation_with_and_without_qrels(eval_runtime: KISRuntime):
    questions = [SkillPixelQuestion("Q1", "TKIS", "short", "")]
    no_qrels_report, no_qrels_rows = evaluate_kis_runtime(
        eval_runtime, questions, None, top_k=100
    )
    assert no_qrels_report["quality_status"] == "UNVALIDATED_ON_HCMAIC"
    assert no_qrels_report["mrr"] is None
    assert no_qrels_report["recall_at"]["1"] is None
    assert no_qrels_rows[0]["quality_status"] == "UNVALIDATED_ON_HCMAIC"

    qrels = KISQrelSet(
        qrels={
            "Q1": load_kis_qrels_from_values("Q1", "demo:000005", "demo.avi,10")
        },
        source="hcmaic-official",
    )
    report, rows = evaluate_kis_runtime(eval_runtime, questions, qrels, top_k=100)
    assert report["quality_status"] == "VALIDATED_ON_HCMAIC"
    assert report["n_scored"] == 1
    assert report["mrr"] is not None
    assert report["latency_ms"]["p50"] is not None
    assert rows[0]["first_relevant_rank"] is not None


def load_kis_qrels_from_values(query_id: str, frame_uid: str, answer_cell: str):
    return KISQrel(
        query_id=query_id,
        relevant_frame_uids=frozenset({frame_uid}),
        relevant_answer_cells=frozenset({answer_cell}),
    )


def test_ablation_keeps_visual_baseline(eval_runtime: KISRuntime):
    questions = [SkillPixelQuestion("Q1", "TKIS", "short", "")]
    reports = run_kis_ablation(eval_runtime, questions, None, top_k=100)
    assert set(reports) == {
        "visual",
        "visual+ocr",
        "visual+object",
        "visual+asr",
        "all-configured",
    }
    assert reports["visual"]["mrr"] is None
    assert reports["visual"]["quality_status"] == "UNVALIDATED_ON_HCMAIC"


def test_qrels_mismatch_fails_closed(eval_runtime: KISRuntime):
    questions = [SkillPixelQuestion("Q1", "TKIS", "short", "")]
    qrels = KISQrelSet(qrels={}, source="hcmaic-official")
    with pytest.raises(KISEvaluationError, match="mismatch"):
        evaluate_kis_runtime(eval_runtime, questions, qrels, top_k=100)
