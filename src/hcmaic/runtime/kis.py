"""Production KIS runtime composition over versioned raw-derived artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmaic.embedding.base import EmbeddingProvider
from hcmaic.embedding.factory import get_real_visual_provider
from hcmaic.retrieval.asr import ASRArtifactError, ASRRetrievalChannel, load_asr_artifact
from hcmaic.retrieval.kis_orchestrator import KISHybridOrchestrator, KISHybridOutput
from hcmaic.retrieval.object_retrieval import (
    ObjectArtifactError,
    ObjectRetrievalChannel,
    load_object_artifact,
)
from hcmaic.retrieval.ocr_bm25 import (
    BM25OCRChannel,
    OCRArtifactError,
    load_ocr_artifact,
)
from hcmaic.skillpixel.index import SkillPixelIndex, load_skillpixel_index
from hcmaic.skillpixel.retrieval import SkillPixelRetriever


@dataclass
class KISRuntime:
    """Loaded KIS graph and diagnostics used by CLI, API and rehearsal."""

    index: SkillPixelIndex
    provider: EmbeddingProvider
    retriever: SkillPixelRetriever
    orchestrator: KISHybridOrchestrator
    provider_selection: dict[str, Any]
    channel_status: dict[str, str]

    @classmethod
    def from_components(
        cls,
        index: SkillPixelIndex,
        provider: EmbeddingProvider,
        *,
        optional_channels: dict[str, Any] | None = None,
        provider_selection: dict[str, Any] | None = None,
        channel_status: dict[str, str] | None = None,
        asr_enabled: bool = False,
        max_per_video: int | None = 5,
    ) -> KISRuntime:
        retriever = SkillPixelRetriever(index, provider)
        orchestrator = KISHybridOrchestrator(
            retriever,
            optional_channels=optional_channels,
            asr_enabled=asr_enabled,
            max_per_video=max_per_video,
        )
        return cls(
            index=index,
            provider=provider,
            retriever=retriever,
            orchestrator=orchestrator,
            provider_selection=provider_selection or {},
            channel_status=channel_status or {},
        )

    def search(self, query: Any) -> KISHybridOutput:
        return self.orchestrator.search(query)

    def search_queries(self, queries: list[Any]) -> dict[str, KISHybridOutput]:
        return self.orchestrator.search_queries(queries)

    def frame_image_path(self, frame_uid: str) -> Path:
        for record in self.index.catalog:
            if record.frame_id == frame_uid:
                root = self.index.dataset_root.resolve()
                path = (root / record.image_path).resolve()
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise PermissionError("KIS image path escapes dataset root") from exc
                return path
        raise KeyError(frame_uid)

    def timeline(self, video_id: str) -> list[dict[str, Any]]:
        frames = [record for record in self.index.catalog if record.video_id == video_id]
        if not frames:
            raise KeyError(video_id)
        frames.sort(
            key=lambda record: (
                record.source_frame_idx
                if record.source_frame_idx is not None
                else record.frame_idx,
                record.frame_id,
            )
        )
        return [
            {
                **record.model_dump(),
                "source_frame_idx": record.source_frame_idx
                if record.source_frame_idx is not None
                else record.frame_idx,
                "image_url": f"/frames/{record.frame_id}/image",
            }
            for record in frames
        ]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "kis_runtime": True,
            "index_size": self.index.size,
            "index_version": self.index.index_manifest.get("index_version"),
            "embedding_provider": self.provider.name,
            "embedding": self.provider.info(),
            "n_videos": len({record.video_id for record in self.index.catalog}),
            "channels": dict(self.channel_status),
            "quality_status": "UNVALIDATED_ON_HCMAIC",
        }


def load_kis_runtime(
    index_dir: Path,
    *,
    provider: str = "auto",
    device: str | None = None,
    local_files_only: bool = True,
    batch_size: int | None = None,
    ocr_artifact: Path | None = None,
    object_artifact: Path | None = None,
    asr_artifact: Path | None = None,
    asr_enabled: bool = False,
) -> KISRuntime:
    """Load the exact provider/index pair and optional local channel artifacts."""
    index = load_skillpixel_index(index_dir)
    expected_provider = str(index.provider_info.get("provider", ""))
    prefer = expected_provider if provider == "auto" else provider
    if prefer not in {"siglip2", "clip", "jina-clip-v2"}:
        raise ValueError(f"unsupported KIS visual provider {prefer!r}")
    visual_provider, selection = get_real_visual_provider(
        prefer=prefer,
        device=device,
        local_files_only=local_files_only,
        revision=index.provider_info.get("model_revision"),
        batch_size=batch_size,
    )
    if visual_provider.name != expected_provider:
        raise RuntimeError(
            f"loaded provider {visual_provider.name!r} does not match index provider "
            f"{expected_provider!r}; rebuild index with the cached real provider"
        )
    optional_channels: dict[str, Any] = {}
    channel_status = {
        "ocr": "not_configured",
        "object": "not_configured",
        "asr": "disabled_by_policy" if not asr_enabled else "not_configured",
    }
    dataset_hash = str(index.index_manifest.get("dataset_manifest_hash", ""))
    if ocr_artifact is not None:
        try:
            optional_channels["ocr"] = BM25OCRChannel(
                load_ocr_artifact(ocr_artifact, dataset_manifest_hash=dataset_hash)
            )
            channel_status["ocr"] = "ready"
        except OCRArtifactError as exc:
            channel_status["ocr"] = f"unavailable: {type(exc).__name__}: {exc}"
    if object_artifact is not None:
        try:
            optional_channels["object"] = ObjectRetrievalChannel(
                load_object_artifact(object_artifact, dataset_manifest_hash=dataset_hash)
            )
            channel_status["object"] = "ready"
        except ObjectArtifactError as exc:
            channel_status["object"] = f"unavailable: {type(exc).__name__}: {exc}"
    if asr_enabled and asr_artifact is not None:
        try:
            optional_channels["asr"] = ASRRetrievalChannel(
                load_asr_artifact(asr_artifact, dataset_manifest_hash=dataset_hash)
            )
            channel_status["asr"] = "ready"
        except ASRArtifactError as exc:
            channel_status["asr"] = f"unavailable: {type(exc).__name__}: {exc}"

    return KISRuntime.from_components(
        index,
        visual_provider,
        optional_channels=optional_channels,
        provider_selection=selection,
        channel_status=channel_status,
        asr_enabled=asr_enabled,
    )
