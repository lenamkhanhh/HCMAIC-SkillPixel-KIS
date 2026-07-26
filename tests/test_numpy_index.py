"""ExactNumpyIndex tests: mapping correctness, determinism, filtering."""

import numpy as np
import pytest

from hcmaic.indexing.numpy_index import ExactNumpyIndex


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def small_index() -> ExactNumpyIndex:
    # Orthogonal basis vectors -> unambiguous nearest neighbors.
    vectors = np.eye(4, dtype=np.float32)
    index = ExactNumpyIndex()
    index.build(vectors, ["f0", "f1", "f2", "f3"])
    return index


def test_row_to_frame_mapping(small_index: ExactNumpyIndex):
    """A query aligned with row i must return exactly frame i first."""
    for i, expected in enumerate(["f0", "f1", "f2", "f3"]):
        hits = small_index.search(np.eye(4, dtype=np.float32)[i], top_k=1)
        assert hits[0][0] == expected
        assert hits[0][1] == pytest.approx(1.0)


def test_top_k_limits(small_index: ExactNumpyIndex):
    assert len(small_index.search(_unit([1, 1, 0, 0]), top_k=2)) == 2
    assert len(small_index.search(_unit([1, 1, 0, 0]), top_k=10)) == 4
    assert small_index.search(_unit([1, 0, 0, 0]), top_k=0) == []


def test_deterministic_tie_break():
    # Two identical vectors: tie must break by frame_id ascending.
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    index = ExactNumpyIndex()
    index.build(vectors, ["zz", "aa", "bb"])
    hits = index.search(np.array([1, 0], dtype=np.float32), top_k=3)
    assert [h[0] for h in hits] == ["aa", "zz", "bb"]


def test_allowed_rows_filter(small_index: ExactNumpyIndex):
    mask = np.array([False, True, False, True])
    hits = small_index.search(_unit([1, 1, 1, 1]), top_k=4, allowed_rows=mask)
    assert {h[0] for h in hits} == {"f1", "f3"}


def test_empty_mask_returns_empty(small_index: ExactNumpyIndex):
    mask = np.zeros(4, dtype=bool)
    assert small_index.search(_unit([1, 0, 0, 0]), top_k=3, allowed_rows=mask) == []


def test_build_rejects_row_id_mismatch():
    index = ExactNumpyIndex()
    with pytest.raises(ValueError, match="row/id mismatch"):
        index.build(np.eye(3, dtype=np.float32), ["a", "b"])


def test_build_rejects_non_2d():
    index = ExactNumpyIndex()
    with pytest.raises(ValueError, match="2-D"):
        index.build(np.zeros(3, dtype=np.float32), ["a", "b", "c"])


def test_query_dim_checked(small_index: ExactNumpyIndex):
    with pytest.raises(ValueError, match="dimension"):
        small_index.search(np.zeros(3, dtype=np.float32), top_k=1)


def test_mask_length_checked(small_index: ExactNumpyIndex):
    with pytest.raises(ValueError, match="mask length"):
        small_index.search(
            _unit([1, 0, 0, 0]), top_k=1, allowed_rows=np.array([True, False])
        )


def test_unbuilt_index_raises():
    with pytest.raises(RuntimeError, match="not built"):
        ExactNumpyIndex().search(np.zeros(4, dtype=np.float32), top_k=1)


def test_size(small_index: ExactNumpyIndex):
    assert small_index.size == 4
    assert ExactNumpyIndex().size == 0
