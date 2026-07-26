"""Deterministic mock embedding provider tests."""

from pathlib import Path

import numpy as np

from hcmaic.embedding.base import get_provider, l2_normalize
from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider


def test_determinism_across_instances(sample_root: Path):
    paths = [sample_root / "keyframes" / "L01_V001" / "001.jpg"]
    a = DeterministicMockEmbeddingProvider()
    b = DeterministicMockEmbeddingProvider()
    np.testing.assert_array_equal(a.embed_images(paths), b.embed_images(paths))
    np.testing.assert_array_equal(
        a.embed_texts(["red bus"]), b.embed_texts(["red bus"])
    )


def test_normalization(sample_root: Path):
    provider = DeterministicMockEmbeddingProvider()
    images = sorted((sample_root / "keyframes").rglob("*.jpg"))
    vecs = provider.embed_images(images)
    assert vecs.dtype == np.float32
    assert vecs.shape == (len(images), provider.dimension)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    texts = provider.embed_texts(["red", "", "blue and yellow"])
    np.testing.assert_allclose(np.linalg.norm(texts, axis=1), 1.0, atol=1e-5)


def test_cross_modal_signal(sample_root: Path):
    """Text naming a color must rank the image of that color first."""
    provider = DeterministicMockEmbeddingProvider()
    image_ids = ["L01_V001/001", "L01_V001/002", "L01_V001/003"]  # red, blue, green
    paths = [sample_root / "keyframes" / f"{i}.jpg" for i in image_ids]
    image_vecs = provider.embed_images(paths)
    for word, expected_row in (("red", 0), ("blue", 1), ("green", 2)):
        text_vec = provider.embed_texts([f"a {word} scene"])[0]
        scores = image_vecs @ text_vec
        assert scores.argmax() == expected_row, (word, scores.tolist())


def test_synonyms_map_to_anchor():
    provider = DeterministicMockEmbeddingProvider()
    crimson = provider.embed_texts(["crimson"])[0]
    red = provider.embed_texts(["red"])[0]
    assert float(crimson @ red) > 0.9


def test_zero_vector_guard():
    normalized = l2_normalize(np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_get_provider_registry():
    assert isinstance(get_provider("mock"), DeterministicMockEmbeddingProvider)
    try:
        get_provider("nope")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "nope" in str(exc)
