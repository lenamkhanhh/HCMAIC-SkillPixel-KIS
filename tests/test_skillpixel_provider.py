"""Provider-boundary tests that never fetch model weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.embedding.factory import get_real_visual_provider


class _RecordingProvider(EmbeddingProvider):
    name = "recording"
    version = "recording-v1"

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(np.ones((len(paths), 3), dtype=np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.ones((len(texts), 3), dtype=np.float32))


def test_provider_boundary_normalizes_float32_and_supports_query_image(tmp_path: Path):
    provider = _RecordingProvider()
    image = tmp_path / "query.jpg"
    image.write_bytes(b"not decoded by the recording provider")

    result = provider.embed_query_image(image)

    assert result.shape == (1, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)


def test_real_factory_uses_siglip2_first_and_passes_cache_only(monkeypatch):
    import hcmaic.embedding.factory as factory

    calls: list[tuple[str, bool]] = []

    class _FakeSigLIP(_RecordingProvider):
        name = "siglip2"
        version = "siglip2:test"

        def __init__(self, **kwargs):
            calls.append(("siglip2", kwargs["local_files_only"]))

    monkeypatch.setattr(factory, "RealSiglip2EmbeddingProvider", _FakeSigLIP)
    provider, report = get_real_visual_provider(prefer="siglip2", local_files_only=True)

    assert provider.name == "siglip2"
    assert report["provider"] == "siglip2"
    assert report["fallback"] is None
    assert calls == [("siglip2", True)]


def test_real_factory_falls_back_to_real_clip_with_evidence(monkeypatch):
    import hcmaic.embedding.factory as factory

    class _UnavailableSigLIP:
        def __init__(self, **kwargs):
            raise RuntimeError("siglip2 cache unavailable")

    class _FakeCLIP(_RecordingProvider):
        name = "clip"
        version = "clip:test"

        def __init__(self, **kwargs):
            assert kwargs["local_files_only"] is True

    monkeypatch.setattr(factory, "RealSiglip2EmbeddingProvider", _UnavailableSigLIP)
    monkeypatch.setattr(factory, "RealClipEmbeddingProvider", _FakeCLIP)
    provider, report = get_real_visual_provider(prefer="siglip2", local_files_only=True)

    assert provider.name == "clip"
    assert report["provider"] == "clip"
    assert report["fallback"]["provider"] == "siglip2"
    assert "cache unavailable" in report["fallback"]["error"]
