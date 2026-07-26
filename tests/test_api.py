"""FastAPI endpoint tests: success paths, errors, and safe image serving."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hcmaic.api.app import create_app


@pytest.fixture(scope="module")
def client(built_artifacts_dir: Path, sample_root: Path) -> TestClient:
    app = create_app(built_artifacts_dir, dataset_root=sample_root)
    return TestClient(app)


def test_health(client: TestClient):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["index_size"] == 12
    assert data["n_videos"] == 5
    assert data["embedding_provider"] == "mock"


def test_system_info(client: TestClient):
    data = client.get("/system/info").json()
    assert data["n_frames"] == 12
    assert data["index_manifest"]["embedding"]["version"] == "mock-palette-v1"
    assert len(data["video_ids"]) == 5


def test_search_success(client: TestClient):
    res = client.post(
        "/search", json={"text": "a solid red keyframe", "top_k": 5}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total_found"] == 5
    assert data["query_id"].startswith("q-")
    assert data["latency_ms"] >= 0
    top = data["results"][0]
    assert top["frame_id"] == "L01_V001:001"
    assert top["rank"] == 1
    assert top["index_version"] == data["index_version"]
    assert "visual" in top["signal_scores"]


def test_search_respects_query_id(client: TestClient):
    data = client.post(
        "/search", json={"query_id": "mine", "text": "blue"}
    ).json()
    assert data["query_id"] == "mine"


def test_search_video_filter(client: TestClient):
    data = client.post(
        "/search",
        json={"text": "blue", "filters": {"video_ids": ["L01_V004"]}},
    ).json()
    assert data["total_found"] > 0
    assert {r["video_id"] for r in data["results"]} == {"L01_V004"}


def test_search_empty_result_is_valid(client: TestClient):
    data = client.post(
        "/search", json={"text": "blue", "filters": {"video_ids": ["NOPE"]}}
    ).json()
    assert data["total_found"] == 0
    assert data["results"] == []


def test_search_validation_errors(client: TestClient):
    assert client.post("/search", json={"text": ""}).status_code == 422
    assert client.post("/search", json={}).status_code == 422
    assert client.post("/search", json={"text": "x", "top_k": 0}).status_code == 422
    res = client.post(
        "/search", json={"text": "x", "filters": {"video_ids": 42}}
    )
    assert res.status_code == 422
    assert "video_ids" in str(res.json()["detail"])


def test_get_frame(client: TestClient):
    data = client.get("/frames/L01_V001:002", params={"window": 1}).json()
    assert data["frame"]["frame_id"] == "L01_V001:002"
    neighbor_ids = [n["frame_id"] for n in data["neighbors"]]
    assert neighbor_ids == ["L01_V001:001", "L01_V001:002", "L01_V001:003"]
    current = [n for n in data["neighbors"] if n["is_current"]]
    assert len(current) == 1 and current[0]["frame_id"] == "L01_V001:002"


def test_get_frame_404(client: TestClient):
    res = client.get("/frames/NOPE:001")
    assert res.status_code == 404
    assert "NOPE:001" in res.json()["detail"]


def test_frame_image_served(client: TestClient):
    res = client.get("/frames/L01_V001:001/image")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content[:2] == b"\xff\xd8"  # JPEG magic


def test_frame_image_unknown_404(client: TestClient):
    assert client.get("/frames/NOPE:001/image").status_code == 404


def test_frame_image_traversal_rejected(client: TestClient):
    # Traversal-looking ids are simply unknown frame ids -> 404, never a file read.
    res = client.get("/frames/..%2F..%2Fpyproject.toml/image")
    assert res.status_code == 404


def test_timeline(client: TestClient):
    data = client.get("/videos/L01_V002/timeline").json()
    assert data["n_frames"] == 3
    timestamps = [f["timestamp_ms"] for f in data["frames"]]
    assert timestamps == sorted(timestamps)
    assert all(f["image_url"].endswith("/image") for f in data["frames"])


def test_timeline_404(client: TestClient):
    assert client.get("/videos/NOPE/timeline").status_code == 404


def test_submit_preview(client: TestClient):
    res = client.post(
        "/submit/preview",
        json={"query_id": "q9", "task_type": "kis", "frame_id": "L01_V005:002"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["video_id"] == "L01_V005"
    assert data["frame_id"] == "L01_V005:002"
    assert data["timestamp_ms"] == 5000
    assert data["evidence"]["image_path"] == "keyframes/L01_V005/002.jpg"


def test_submit_preview_404(client: TestClient):
    res = client.post(
        "/submit/preview",
        json={"query_id": "q9", "task_type": "kis", "frame_id": "NOPE:404"},
    )
    assert res.status_code == 404


def test_submit_preview_validation(client: TestClient):
    res = client.post(
        "/submit/preview",
        json={"query_id": "", "task_type": "kis", "frame_id": "L01_V005:002"},
    )
    assert res.status_code == 422


def test_ui_served_at_root(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "HCMAIC keyframe search" in res.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
