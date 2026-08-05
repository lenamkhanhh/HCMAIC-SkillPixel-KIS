"""SkillPixel question adapter and TKIS text retrieval."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider
from hcmaic.skillpixel.index import SkillPixelIndex, load_skillpixel_index


@dataclass(frozen=True)
class SkillPixelQuestion:
    query_id: str
    task: str
    text: str
    query_image: str


@dataclass(frozen=True)
class SkillPixelHit:
    query_id: str
    task: str
    rank: int
    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    visual_score: float
    image_path: str
    faiss_row: int
    feature_row: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "task": self.task,
            "rank": self.rank,
            "frame_uid": self.frame_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "source_frame_idx": self.source_frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "visual_score": self.visual_score,
            "image_path": self.image_path,
            "faiss_row": self.faiss_row,
            "feature_row": self.feature_row,
        }


def load_skillpixel_questions(path: Path) -> list[SkillPixelQuestion]:
    """Read organizer questions while preserving file order and query IDs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"query_id", "task"} - fields
        if missing:
            raise ValueError(f"questions.csv missing required columns: {sorted(missing)}")
        questions: list[SkillPixelQuestion] = []
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            task = (row.get("task") or "").strip().upper()
            text = (row.get("text") or "").strip()
            query_image = (row.get("query_image") or "").strip()
            if not query_id:
                raise ValueError(f"questions.csv line {line}: query_id is empty")
            if query_id in seen:
                raise ValueError(f"questions.csv has duplicate query_id {query_id!r}")
            if task not in {"TKIS", "VKIS"}:
                raise ValueError(f"questions.csv line {line}: unsupported task {task!r}")
            if task == "TKIS" and not text:
                raise ValueError(f"questions.csv line {line}: TKIS text is empty")
            if task == "VKIS" and not query_image:
                raise ValueError(f"questions.csv line {line}: VKIS query_image is empty")
            seen.add(query_id)
            questions.append(SkillPixelQuestion(query_id, task, text, query_image))
    if not questions:
        raise ValueError(f"questions.csv is empty: {path}")
    return questions


class SkillPixelRetriever:
    """Shared visual index service; P0-D exposes the text tower for TKIS."""

    def __init__(self, index: SkillPixelIndex, provider: EmbeddingProvider) -> None:
        expected = index.provider_info
        if provider.dimension != index.dimension:
            raise ValueError(
                f"provider dimension {provider.dimension} != index dimension {index.dimension}"
            )
        expected_version = str(expected.get("version", ""))
        if expected_version and provider.version != expected_version:
            raise ValueError(
                f"provider version {provider.version!r} != index version {expected_version!r}"
            )
        self.index = index
        self.provider = provider

    @classmethod
    def from_artifacts(
        cls, artifact_dir: Path, *, provider: EmbeddingProvider
    ) -> SkillPixelRetriever:
        return cls(load_skillpixel_index(artifact_dir), provider)

    def _make_hits(
        self,
        query_id: str,
        task: str,
        matches: Iterable[tuple[dict[str, Any], float]],
    ) -> list[SkillPixelHit]:
        return [
            SkillPixelHit(
                query_id=query_id,
                task=task,
                rank=rank,
                frame_uid=str(metadata["frame_uid"]),
                video_id=str(metadata["video_id"]),
                video_filename=str(metadata["video_filename"]),
                source_frame_idx=int(metadata["source_frame_idx"]),
                timestamp_ms=int(metadata["timestamp_ms"]),
                visual_score=float(score),
                image_path=str(metadata["image_path"]),
                faiss_row=int(metadata["faiss_row"]),
                feature_row=int(metadata["feature_row"]),
            )
            for rank, (metadata, score) in enumerate(matches, start=1)
        ]

    def search_text(self, query_id: str, text: str, top_k: int = 100) -> list[SkillPixelHit]:
        return self.search_text_queries([(query_id, text)], top_k=top_k)[query_id]

    def search_text_queries(
        self, queries: list[tuple[str, str]], *, top_k: int = 100
    ) -> dict[str, list[SkillPixelHit]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not queries:
            return {}
        query_ids = [query_id.strip() for query_id, _ in queries]
        if any(not query_id for query_id in query_ids):
            raise ValueError("TKIS query_id must not be empty")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("TKIS query_id values must be unique")
        texts = [text.strip() for _, text in queries]
        if any(not text for text in texts):
            raise ValueError("TKIS text must not be empty")
        vectors = np.asarray(self.provider.embed_texts(texts), dtype=np.float32)
        expected_shape = (len(queries), self.index.dimension)
        if vectors.shape != expected_shape:
            raise ValueError(f"text provider returned {vectors.shape}; expected {expected_shape}")
        results: dict[str, list[SkillPixelHit]] = {}
        for query_id, vector in zip(query_ids, vectors, strict=True):
            results[query_id] = self._make_hits(
                query_id, "TKIS", self.index.search(vector, top_k)
            )
        return results

    def search_tkis_questions(
        self, questions_path: Path, *, top_k: int = 100
    ) -> dict[str, list[SkillPixelHit]]:
        questions = load_skillpixel_questions(questions_path)
        tkis = [(item.query_id, item.text) for item in questions if item.task == "TKIS"]
        return self.search_text_queries(tkis, top_k=top_k)
