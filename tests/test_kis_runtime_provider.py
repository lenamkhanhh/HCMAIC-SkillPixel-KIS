"""Provider selection tests for the production KIS runtime loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hcmaic.runtime.kis as kis_runtime
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize


class _JinaRuntimeProvider(EmbeddingProvider):
    name = "jina-clip-v2"
    version = "jina-runtime-test"

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(np.ones((len(paths), 3), dtype=np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.ones((len(texts), 3), dtype=np.float32))


def test_load_kis_runtime_accepts_jina_clip_v2(monkeypatch):
    index = SimpleNamespace(
        embeddings=np.zeros((1, 3), dtype=np.float32),
        provider_info={"provider": "jina-clip-v2", "version": "jina-runtime-test"},
        index_manifest={"dataset_manifest_hash": "raw-hash"},
        size=1,
        dimension=3,
    )
    calls: list[str] = []

    def fake_loader(*args, **kwargs):
        return index

    def fake_provider(**kwargs):
        calls.append(kwargs["prefer"])
        return _JinaRuntimeProvider(), {"requested_provider": kwargs["prefer"]}

    monkeypatch.setattr(kis_runtime, "load_skillpixel_index", fake_loader)
    monkeypatch.setattr(kis_runtime, "get_real_visual_provider", fake_provider)

    runtime = kis_runtime.load_kis_runtime(Path("index"), provider="jina-clip-v2")

    assert runtime.provider.name == "jina-clip-v2"
    assert calls == ["jina-clip-v2"]
