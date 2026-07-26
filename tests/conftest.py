"""Shared fixtures: the committed sample dataset and built artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider
from hcmaic.indexing.artifacts import build_index_artifacts, load_index_artifacts
from hcmaic.ingestion.catalog import build_catalog
from hcmaic.retrieval.service import RetrievalService

SYSTEM_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = SYSTEM_ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def sample_root() -> Path:
    assert SAMPLE_DATA.is_dir(), "committed fixture missing; run scripts/make_fixture.py"
    return SAMPLE_DATA


@pytest.fixture(scope="session")
def built_artifacts_dir(tmp_path_factory: pytest.TempPathFactory, sample_root: Path) -> Path:
    """Artifacts built once per test session from the committed fixture."""
    out = tmp_path_factory.mktemp("artifacts")
    catalog = build_catalog(sample_root)
    provider = DeterministicMockEmbeddingProvider()
    build_index_artifacts(sample_root, catalog, provider, out)
    return out


@pytest.fixture(scope="session")
def service(built_artifacts_dir: Path, sample_root: Path) -> RetrievalService:
    artifacts = load_index_artifacts(built_artifacts_dir)
    return RetrievalService(artifacts, dataset_root=sample_root)


@pytest.fixture()
def dataset_copy(tmp_path: Path, sample_root: Path) -> Path:
    """Mutable per-test copy of the sample dataset."""
    dest = tmp_path / "dataset"
    shutil.copytree(sample_root, dest)
    report = dest / "validation_report.json"
    if report.exists():
        report.unlink()
    return dest
