"""Versioned exact visual index artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.skillpixel.index import (
    SkillPixelIndexError,
    build_skillpixel_index,
    load_skillpixel_index,
)
from hcmaic.skillpixel.raw import ingest_raw_videos


class _TinyRealShapeProvider(EmbeddingProvider):
    """Deterministic test provider; production CLI never selects this class."""

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
            np.asarray([[float(len(text)), 1.0, 0.0, 0.0] for text in texts], dtype=np.float32)
        )


def _make_video(path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48)
    )
    assert writer.isOpened()
    for idx in range(8):
        frame = np.full((48, 64, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def raw_dataset(tmp_path: Path) -> Path:
    source = _make_video(tmp_path / "raw" / "demo.avi")
    output = tmp_path / "generated"
    ingest_raw_videos(source.parent, output, stride_frames=2)
    return output


def test_build_and_reload_faiss_artifacts_round_trip(raw_dataset: Path, tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    provider = _TinyRealShapeProvider()

    build_skillpixel_index(raw_dataset, artifact_dir, provider)

    for name in (
        "catalog.jsonl",
        "embeddings.npy",
        "id_map.json",
        "index.faiss",
        "dataset_manifest.json",
        "index_manifest.json",
    ):
        assert (artifact_dir / name).is_file(), name

    loaded = load_skillpixel_index(artifact_dir)
    assert loaded.embeddings.shape == (4, 4)
    assert loaded.faiss_index.ntotal == 4
    assert loaded.faiss_index.d == 4
    assert all(np.isclose(np.linalg.norm(row), 1.0) for row in loaded.embeddings)

    id_map = json.loads((artifact_dir / "id_map.json").read_text(encoding="utf-8"))
    assert [item["faiss_row"] for item in id_map] == [0, 1, 2, 3]
    assert [item["feature_row"] for item in id_map] == [0, 1, 2, 3]
    assert [item["source_frame_idx"] for item in id_map] == [0, 2, 4, 6]
    assert all(item["keyframe_id"] != item["source_frame_idx"] for item in id_map[1:])

    query = loaded.embeddings[2]
    oracle = loaded.oracle_search(query, top_k=4)
    production = loaded.search(query, top_k=4)
    assert [hit[0]["frame_uid"] for hit in production] == [
        hit[0]["frame_uid"] for hit in oracle
    ]
    assert [hit[1] for hit in production] == pytest.approx([hit[1] for hit in oracle])


def test_index_load_rejects_source_mapping_tamper(raw_dataset: Path, tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    build_skillpixel_index(raw_dataset, artifact_dir, _TinyRealShapeProvider())
    id_map_path = artifact_dir / "id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map[0]["source_frame_idx"] = 99
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")

    with pytest.raises(SkillPixelIndexError, match="source_frame_idx"):
        load_skillpixel_index(artifact_dir)
