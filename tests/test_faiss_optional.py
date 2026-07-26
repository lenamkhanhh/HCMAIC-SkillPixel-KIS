"""Optional FAISS provider tests — skipped automatically when faiss-cpu
is not installed. ExactNumpyIndex remains the verified mandatory path."""

import numpy as np
import pytest

pytest.importorskip("faiss")

from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider  # noqa: E402
from hcmaic.indexing.faiss_index import FaissIndex  # noqa: E402
from hcmaic.indexing.numpy_index import ExactNumpyIndex  # noqa: E402


@pytest.fixture(scope="module")
def indexes(built_artifacts_dir):
    from hcmaic.indexing.artifacts import load_index_artifacts

    artifacts = load_index_artifacts(built_artifacts_dir)
    faiss_index, numpy_index = FaissIndex(), ExactNumpyIndex()
    faiss_index.build(artifacts.embeddings, artifacts.id_map)
    numpy_index.build(artifacts.embeddings, artifacts.id_map)
    return faiss_index, numpy_index, artifacts


@pytest.mark.parametrize(
    "query",
    [
        "a solid red keyframe",
        "blue scene",
        "cyan and magenta pattern",
        "completely unrelated words",
    ],
)
def test_faiss_matches_numpy(indexes, query):
    faiss_index, numpy_index, _ = indexes
    vec = DeterministicMockEmbeddingProvider().embed_texts([query])[0]
    faiss_hits = faiss_index.search(vec, 10)
    numpy_hits = numpy_index.search(vec, 10)
    assert [h[0] for h in faiss_hits] == [h[0] for h in numpy_hits]
    for (_, fa), (_, na) in zip(faiss_hits, numpy_hits, strict=True):
        assert fa == pytest.approx(na, abs=1e-5)


def test_faiss_filtered_matches_numpy(indexes):
    faiss_index, numpy_index, artifacts = indexes
    vec = DeterministicMockEmbeddingProvider().embed_texts(["blue"])[0]
    mask = np.zeros(len(artifacts.id_map), dtype=bool)
    mask[:5] = True
    faiss_hits = faiss_index.search(vec, 5, allowed_rows=mask)
    numpy_hits = numpy_index.search(vec, 5, allowed_rows=mask)
    assert [h[0] for h in faiss_hits] == [h[0] for h in numpy_hits]


def test_faiss_size_and_errors(indexes):
    faiss_index, _, _ = indexes
    assert faiss_index.size == 12
    with pytest.raises(ValueError, match="row/id mismatch"):
        FaissIndex().build(np.eye(3, dtype=np.float32), ["a"])
    with pytest.raises(RuntimeError, match="not built"):
        FaissIndex().search(np.zeros(3, dtype=np.float32), 1)
