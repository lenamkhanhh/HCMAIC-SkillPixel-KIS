"""Concrete local-cache-only SigLIP2 image/text provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
CUDA_BATCH = 8
CPU_BATCH = 8


def _feature_tensor(output: Any, torch: Any) -> Any:
    """Extract a 2-D feature tensor across Transformers model output variants."""
    if isinstance(output, torch.Tensor):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        value = getattr(output, name, None)
        if value is not None:
            if value.ndim == 3:
                return value[:, 0]
            return value
    if isinstance(output, (tuple, list)) and output:
        value = output[0]
        if value.ndim == 3:
            return value[:, 0]
        return value
    raise RuntimeError(f"Could not extract embedding tensor from {type(output)!r}")


def _configured_dimension(config: Any) -> int:
    for candidate in (
        config,
        getattr(config, "vision_config", None),
        getattr(config, "text_config", None),
    ):
        if candidate is None:
            continue
        value = getattr(candidate, "projection_dim", None)
        if value is not None:
            return int(value)
    raise RuntimeError("SigLIP2 model config has no projection_dim")


class RealSiglip2EmbeddingProvider(EmbeddingProvider):
    """SigLIP2 provider; construction never downloads when local-only is true."""

    name = "siglip2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        local_files_only: bool = True,
        revision: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP2 requires torch+transformers. Install with: "
                "uv sync --extra clip"
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        default_batch = CUDA_BATCH if self._device.startswith("cuda") else CPU_BATCH
        self._batch = batch_size or default_batch
        if self._batch < 1:
            raise ValueError("batch_size must be >= 1")
        self._model_name = model_name
        self._local_files_only = local_files_only
        load_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            self._model = AutoModel.from_pretrained(model_name, **load_kwargs).to(
                self._device
            ).eval()
            self._processor = AutoProcessor.from_pretrained(model_name, **load_kwargs)
        except OSError as exc:
            mode = "local cache" if local_files_only else "configured model source"
            raise RuntimeError(
                f"SigLIP2 model {model_name!r} is unavailable from {mode}; "
                "cache the model first or pass local_files_only=False explicitly."
            ) from exc
        self._revision = (
            getattr(self._model.config, "_commit_hash", None) or revision or "main"
        )
        self.version = f"siglip2:{model_name}@{self._revision}"
        self._dimension = _configured_dimension(self._model.config)

    @property
    def dimension(self) -> int:
        return self._dimension

    def info(self) -> dict[str, Any]:
        data = super().info()
        data.update(
            {
                "model_name": self._model_name,
                "model_revision": self._revision,
                "revision": self._revision,
                "device": self._device,
                "batch_size": self._batch,
                "dtype": str(next(self._model.parameters()).dtype),
                "local_files_only": self._local_files_only,
                "preprocessing": "SigLIP2 AutoProcessor RGB resize/crop 224px",
                "evidence_level": "REAL_PROVIDER",
            }
        )
        return data

    def _normalize_features(self, features: Any) -> np.ndarray:
        array = _feature_tensor(features, self._torch).detach().cpu().numpy()
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self._dimension:
            raise RuntimeError(
                f"SigLIP2 returned feature shape {array.shape}; expected (*, {self._dimension})"
            )
        normalized = l2_normalize(array)
        if not np.isfinite(normalized).all():
            raise RuntimeError("SigLIP2 returned non-finite embeddings")
        return normalized

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self._batch):
            images = []
            for path in paths[start : start + self._batch]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            with self._torch.inference_mode():
                features = self._model.get_image_features(**inputs)
            chunks.append(self._normalize_features(features))
        if not chunks:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack(chunks).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            with self._torch.inference_mode():
                features = self._model.get_text_features(**inputs)
            chunks.append(self._normalize_features(features))
        if not chunks:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack(chunks).astype(np.float32)
