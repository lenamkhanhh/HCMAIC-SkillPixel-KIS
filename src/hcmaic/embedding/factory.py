"""Real provider selection with explicit local-cache fallback evidence."""

from __future__ import annotations

from typing import Any

from hcmaic.embedding.base import EmbeddingProvider
from hcmaic.embedding.clip_real import RealClipEmbeddingProvider
from hcmaic.embedding.siglip2 import RealSiglip2EmbeddingProvider


def get_real_visual_provider(
    *,
    prefer: str = "siglip2",
    device: str | None = None,
    local_files_only: bool = True,
    siglip2_model: str = "google/siglip2-base-patch16-224",
    clip_model: str = "openai/clip-vit-base-patch32",
    revision: str | None = None,
    batch_size: int | None = None,
) -> tuple[EmbeddingProvider, dict[str, Any]]:
    """Load SigLIP2 first, then a real CLIP fallback, never a mock provider."""
    if prefer not in {"siglip2", "clip"}:
        raise ValueError("prefer must be 'siglip2' or 'clip'")

    attempts: list[tuple[str, str]] = []
    candidates = ["siglip2", "clip"] if prefer == "siglip2" else ["clip"]
    for name in candidates:
        try:
            provider: EmbeddingProvider
            if name == "siglip2":
                provider = RealSiglip2EmbeddingProvider(
                    model_name=siglip2_model,
                    device=device,
                    local_files_only=local_files_only,
                    revision=revision,
                    batch_size=batch_size,
                )
            else:
                provider = RealClipEmbeddingProvider(
                    model_name=clip_model,
                    device=device,
                    local_files_only=local_files_only,
                    revision=revision,
                    batch_size=batch_size,
                )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            attempts.append((name, str(exc)))
            continue

        fallback = None
        if attempts:
            fallback = {
                "provider": attempts[0][0],
                "error": attempts[0][1],
                "reason": "preferred provider unavailable; real provider fallback used",
            }
        report = {
            "requested_provider": prefer,
            "provider": provider.name,
            "model": provider.info().get("model_name"),
            "revision": provider.info().get("model_revision"),
            "local_files_only": local_files_only,
            "fallback": fallback,
            "attempts": [{"provider": name_, "error": error} for name_, error in attempts],
        }
        return provider, report

    messages = "; ".join(f"{name}: {error}" for name, error in attempts)
    raise RuntimeError(
        "No real visual provider is available. SigLIP2/CLIP must be installed and "
        f"cached locally; attempts: {messages}"
    )
