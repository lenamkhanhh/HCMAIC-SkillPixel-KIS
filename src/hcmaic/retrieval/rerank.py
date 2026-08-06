"""Bounded and optional real text rerankers for KIS candidates."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Protocol

from hcmaic.retrieval.candidates import FusedCandidate
from hcmaic.retrieval.real_channels import package_version, sha256_path


class RealRerankerUnavailable(RuntimeError):
    """Raised when an explicitly requested reranker cannot run safely."""


class Reranker(Protocol):
    def rerank(
        self, candidates: list[FusedCandidate], *, top_k: int
    ) -> list[FusedCandidate]:
        ...


class PassthroughReranker:
    def rerank(
        self, candidates: list[FusedCandidate], *, top_k: int
    ) -> list[FusedCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return candidates[:top_k]


class CrossEncoderReranker:
    """Real sentence-transformers CrossEncoder with explicit download policy."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    SOURCE_URL = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        model_path: Path | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 256,
        allow_model_download: bool = False,
        local_files_only: bool = True,
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("reranker batch_size/max_length must be >= 1")
        self.model_id = model
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.allow_model_download = allow_model_download
        self.local_files_only = local_files_only and not allow_model_download
        if self.model_path is not None and not self.model_path.exists():
            raise RealRerankerUnavailable(f"reranker model path does not exist: {self.model_path}")
        if self.model_path is None and not allow_model_download:
            raise RealRerankerUnavailable(
                "cross-encoder weights are not cached; pass an explicit model path or "
                "--allow-model-download"
            )
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RealRerankerUnavailable(
                "sentence-transformers is not installed; install it explicitly for real reranking"
            ) from exc
        source = str(self.model_path) if self.model_path is not None else self.model_id
        kwargs: dict[str, Any] = {
            "device": self.device,
            "max_length": self.max_length,
        }
        parameters = inspect.signature(CrossEncoder).parameters
        if "local_files_only" in parameters:
            kwargs["local_files_only"] = self.local_files_only
        if "revision" in parameters:
            kwargs["revision"] = None
        try:
            self._model = CrossEncoder(source, **kwargs)
        except Exception as exc:
            mode = "local cache" if self.local_files_only else "configured model source"
            raise RealRerankerUnavailable(
                f"cross-encoder {source!r} is unavailable from {mode}: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        return "cross-encoder"

    @property
    def revision(self) -> str:
        return sha256_path(self.model_path) if self.model_path is not None else self.model_id

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "model_source_url": self.SOURCE_URL,
            "model_id": self.model_id,
            "weights_path": str(self.model_path) if self.model_path is not None else None,
            "weights_sha256": (
                sha256_path(self.model_path) if self.model_path is not None else None
            ),
            "provider_package": "sentence-transformers",
            "provider_package_version": package_version("sentence-transformers"),
            "runtime": {
                "device": self.device,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
            },
            "local_files_only": self.local_files_only,
            "provider_evidence": "REAL_PROVIDER_ARTIFACT",
        }

    def rerank(
        self,
        candidates: list[FusedCandidate],
        *,
        query_text: str | None,
        top_k: int,
        candidate_limit: int,
        timeout_ms: int,
    ) -> list[FusedCandidate]:
        if top_k < 1 or candidate_limit < top_k or timeout_ms < 1:
            raise ValueError("invalid real reranker bounds")
        bounded = candidates[:candidate_limit]
        if not query_text or not query_text.strip():
            for candidate in bounded:
                candidate.rerank_score = candidate.final_score
                candidate.explanation["reranker"] = "cross-encoder-skipped-no-text"
            return bounded[:top_k]
        pairs = [
            (
                query_text,
                " ".join(candidate.evidence_texts.values()).strip()
                or candidate.entity_id,
            )
            for candidate in bounded
        ]
        try:
            scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:
            raise RealRerankerUnavailable(f"cross-encoder inference failed: {exc}") from exc
        values = [float(score) for score in scores]
        if len(values) != len(bounded) or any(not math.isfinite(score) for score in values):
            raise RealRerankerUnavailable("cross-encoder returned invalid score count/values")
        for candidate, score in zip(bounded, values, strict=True):
            candidate.rerank_score = score
            candidate.explanation.update(
                {
                    "reranker": self.name,
                    "reranker_model": self.model_id,
                    "reranker_candidate_limit": float(candidate_limit),
                }
            )
        return sorted(
            bounded,
            key=lambda item: (
                -(item.rerank_score or item.final_score),
                -item.final_score,
                item.video_id,
                str(item.source_frame_idx),
                item.entity_id,
            ),
        )[:top_k]
