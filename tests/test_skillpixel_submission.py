"""SkillPixel top-100 CSV export and validation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hcmaic.skillpixel.submission import (
    SubmissionValidationError,
    export_skillpixel_submission,
    validate_submission_csv,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    questions = tmp_path / "questions.csv"
    with questions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "task", "text", "query_image"])
        writer.writeheader()
        writer.writerow({"query_id": "Q1", "task": "TKIS", "text": "a"})
        writer.writerow({"query_id": "Q2", "task": "VKIS", "query_image": "Q2.jpg"})

    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "path", "duration_seconds", "fps", "frame_count", "width", "height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video": "video_a.mp4",
                "path": "videos/video_a.mp4",
                "duration_seconds": "6",
                "fps": "25",
                "frame_count": "150",
                "width": "64",
                "height": "48",
            }
        )
        writer.writerow(
            {
                "video": "video_b.mp4",
                "path": "videos/video_b.mp4",
                "duration_seconds": "6",
                "fps": "25",
                "frame_count": "150",
                "width": "64",
                "height": "48",
            }
        )

    results = tmp_path / "results.jsonl"
    with results.open("w", encoding="utf-8") as handle:
        for query_id in ("Q1", "Q2"):
            answers = [
                {
                    "video_filename": "video_a.mp4" if rank < 50 else "video_b.mp4",
                    "source_frame_idx": rank,
                }
                for rank in range(100)
            ]
            handle.write(json.dumps({"query_id": query_id, "answers": answers}) + "\n")
    return questions, corpus, results


def test_export_writes_exact_100_quoted_answers_and_round_trips(tmp_path: Path):
    questions, corpus, results = _write_inputs(tmp_path)
    output = tmp_path / "submission.csv"

    stats = export_skillpixel_submission(questions, results, corpus, output)

    assert stats.n_queries == 2
    assert stats.answers_per_query == 100
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, strict=True))
    assert rows[0] == ["query_id"] + [f"answer_{i:03d}" for i in range(1, 101)]
    assert len(rows) == 3
    assert rows[1][0] == "Q1"
    assert rows[1][1] == "video_a.mp4,0"
    assert '"video_a.mp4,0"' in output.read_text(encoding="utf-8")
    report = validate_submission_csv(output, questions, corpus)
    assert report.ok


def test_export_rejects_keyframe_only_and_out_of_range_answers(tmp_path: Path):
    questions, corpus, results = _write_inputs(tmp_path)
    payload = results.read_text(encoding="utf-8").splitlines()
    first = json.loads(payload[0])
    first["answers"][0] = {"video_filename": "video_a.mp4", "keyframe_id": 0}
    first["answers"][1]["source_frame_idx"] = 150
    payload[0] = json.dumps(first)
    results.write_text("\n".join(payload) + "\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="source_frame_idx|out of range"):
        export_skillpixel_submission(questions, results, corpus, tmp_path / "submission.csv")


def test_export_rejects_missing_or_duplicate_query(tmp_path: Path):
    questions, corpus, results = _write_inputs(tmp_path)
    lines = results.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(lines[1])
    duplicate["query_id"] = "Q1"
    results.write_text(lines[0] + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="missing|duplicate"):
        export_skillpixel_submission(questions, results, corpus, tmp_path / "submission.csv")
