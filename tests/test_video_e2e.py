"""Milestone 1 end-to-end: raw video -> keyframes -> validate -> index -> search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from hcmaic.cli.main import main  # noqa: E402
from hcmaic.contracts.models import SearchRequest  # noqa: E402
from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider  # noqa: E402
from hcmaic.indexing.artifacts import build_index_artifacts, load_index_artifacts  # noqa: E402
from hcmaic.ingestion.catalog import build_catalog  # noqa: E402
from hcmaic.ingestion.validator import validate_dataset  # noqa: E402
from hcmaic.retrieval.service import RetrievalService  # noqa: E402
from test_video_ingest import BLUE, GREEN, RED, YELLOW, make_video  # noqa: E402


def test_video_to_search_pipeline(tmp_path: Path):
    videos = tmp_path / "videos"
    make_video(videos / "city_clip.avi", [RED] * 3 + [BLUE] * 3 + [GREEN] * 3)
    make_video(videos / "beach_clip.avi", [YELLOW] * 4)
    dataset = tmp_path / "dataset"
    artifacts = tmp_path / "artifacts"

    # 1. ingest both videos via CLI
    assert (
        main(
            [
                "ingest-video",
                "--input", str(videos),
                "--output", str(dataset),
                "--interval", "1.0",
            ]
        )
        == 0
    )

    # 2. the ingested dataset passes the existing validator with zero errors
    report = validate_dataset(dataset)
    assert report.ok, [e.message for e in report.errors]
    assert report.n_videos == 2

    # 3. catalog + index build on ingested output unchanged
    catalog = build_catalog(dataset)
    assert {r.video_id for r in catalog} == {"city_clip", "beach_clip"}
    sample = catalog[0]
    assert sample.metadata["width"] == 64 and sample.metadata["height"] == 48
    build_index_artifacts(
        dataset, catalog, DeterministicMockEmbeddingProvider(), artifacts
    )

    # 4. search maps back to the right ingested video and timestamp
    service = RetrievalService(
        load_index_artifacts(artifacts), dataset_root=dataset
    )
    hits = service.search(SearchRequest(query_id="e2e", text="a blue frame", top_k=3))
    assert hits[0].video_id == "city_clip"
    frame = service.get_frame(hits[0].frame_id)
    # blue segment spans 1.5s-3.0s of city_clip (3 frames @ 2fps after red)
    assert 1000 <= frame.timestamp_ms <= 3000
    assert (dataset / frame.image_path).is_file()

    yellow_hits = service.search(
        SearchRequest(query_id="e2e2", text="yellow sand", top_k=1)
    )
    assert yellow_hits[0].video_id == "beach_clip"

    # 5. timeline of ingested video is ordered
    timeline = service.timeline("city_clip")
    stamps = [f.timestamp_ms for f in timeline]
    assert stamps == sorted(stamps)


def test_cli_ingest_error_paths(tmp_path: Path):
    # missing input path -> exit 2
    assert (
        main(
            [
                "ingest-video",
                "--input", str(tmp_path / "missing"),
                "--output", str(tmp_path / "out"),
            ]
        )
        == 2
    )
    # directory with only broken videos -> exit 1, report written
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "bad.mp4").write_bytes(b"junk")
    assert (
        main(
            [
                "ingest-video",
                "--input", str(videos),
                "--output", str(tmp_path / "out"),
            ]
        )
        == 1
    )
    report = json.loads(
        (tmp_path / "out" / "ingest_report.json").read_text(encoding="utf-8")
    )
    assert report["n_videos_failed"] == 1
