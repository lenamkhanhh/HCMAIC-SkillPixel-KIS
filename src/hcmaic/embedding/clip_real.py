"""Real CLIP provider (optional; requires `uv sync --extra clip`).

Lightweight ViT-B/32-class model via transformers. CPU path is mandatory;
CUDA is used when available with a batch size safe for 4 GB VRAM.
Never imported by tests; tests must not download weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
CUDA_BATCH = 8  # safe for 4 GB VRAM with ViT-B/32
CPU_BATCH = 16


class RealClipEmbeddingProvider(EmbeddingProvider):
    name = "clip"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Real CLIP provider needs torch+transformers. Install with: "
                "uv sync --extra clip  (tests never require this)."
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._batch = CUDA_BATCH if self._device.startswith("cuda") else CPU_BATCH
        self._model_name = model_name
        self.version = f"clip:{model_name}"
        self._model = CLIPModel.from_pretrained(model_name).to(self._device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._dimension = int(self._model.config.projection_dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def info(self) -> dict[str, Any]:
        data = super().info()
        data.update(
            {
                "model_name": self._model_name,
                "device": self._device,
                "batch_size": self._batch,
            }
        )
        return data

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self._batch):
            batch_paths = paths[start : start + self._batch]
            images = []
            for p in batch_paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                feats = self._model.get_image_features(**inputs)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        if not chunks:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return l2_normalize(np.vstack(chunks))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            with self._torch.no_grad():
                feats = self._model.get_text_features(**inputs)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        if not chunks:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return l2_normalize(np.vstack(chunks))
