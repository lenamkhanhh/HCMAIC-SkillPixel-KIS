"""K8 FastAPI/UI runtime surface tests over a raw-derived exact index."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hcmaic.api.kis_app import create_kis_app
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.runtime.kis import KISRuntime
from hcmaic.skillpixel.index import build_skillpixel_index
from hcmaic.skillpixel.raw import ingest_raw_videos


class _APIProvider(EmbeddingProvider):
    name = "test-api-provider"
    version = "test-api-v1"

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(
            np.asarray(
                [
                    [float(int(path.stem) + 1) if path.stem.isdigit() else 2.0, 1.0, 0.0]
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
def api_client(tmp_path: Path) -> tuple[TestClient, Path]:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    raw_root = tmp_path / "generated"
    ingest_raw_videos(source.parent, raw_root, stride_frames=2)
    provider = _APIProvider()
    index = build_skillpixel_index(raw_root, tmp_path / "index", provider)
    query_image = tmp_path / "query.jpg"
    cv2.imwrite(str(query_image), np.full((48, 64, 3), 60, dtype=np.uint8))
    runtime = KISRuntime.from_components(index, provider)
    return TestClient(create_kis_app(runtime)), query_image


def test_kis_api_health_text_image_batch_and_timeline(
    api_client: tuple[TestClient, Path]
):
    client, query_image = api_client

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["kis_runtime"] is True

    text = client.post("/search/text", json={"query_id": "T1", "text": "hello", "top_k": 2})
    assert text.status_code == 200
    assert text.json()["query_id"] == "T1"
    assert text.json()["results"][0]["source_frame_idx"] >= 0
    assert text.json()["quality_status"] == "UNVALIDATED_ON_HCMAIC"

    image = client.post(
        "/search/image",
        json={"query_id": "V1", "image_path": str(query_image), "top_k": 2},
    )
    assert image.status_code == 200
    assert image.json()["task"] == "VKIS"

    batch = client.post(
        "/search/batch",
        json={
            "queries": [
                {"query_id": "V1", "task": "VKIS", "image_path": str(query_image), "top_k": 1},
                {"query_id": "T1", "task": "TKIS", "text": "hello", "top_k": 1},
            ]
        },
    )
    assert batch.status_code == 200
    assert batch.json()["query_order"] == ["V1", "T1"]
    assert list(batch.json()["results"]) == ["V1", "T1"]

    timeline = client.get("/videos/demo/timeline")
    assert timeline.status_code == 200
    assert timeline.json()["n_frames"] == 6
    frame_uid = timeline.json()["frames"][0]["frame_id"]
    image_response = client.get(f"/frames/{frame_uid}/image")
    assert image_response.status_code == 200


def test_kis_api_rejects_invalid_image_payload(api_client: tuple[TestClient, Path]):
    client, _ = api_client
    response = client.post("/search/image", json={"image_path": "a", "image_base64": "b"})
    assert response.status_code == 422
