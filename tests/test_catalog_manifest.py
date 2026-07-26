"""Catalog determinism and dataset manifest tests."""

from pathlib import Path

from hcmaic.ingestion.catalog import build_catalog, load_catalog, write_catalog
from hcmaic.ingestion.manifest import build_dataset_manifest, manifest_hash


def test_catalog_deterministic_order(sample_root: Path):
    first = build_catalog(sample_root)
    second = build_catalog(sample_root)
    assert [r.frame_id for r in first] == [r.frame_id for r in second]
    ids = [r.frame_id for r in first]
    assert ids == sorted(ids), "catalog must be sorted by (video_id, n)"
    assert len(ids) == 12


def test_catalog_fields(sample_root: Path):
    catalog = build_catalog(sample_root)
    record = next(r for r in catalog if r.frame_id == "L01_V001:001")
    assert record.video_id == "L01_V001"
    assert record.keyframe_id == "001"
    assert record.pts == 1.0
    assert record.timestamp_ms == 1000
    assert record.frame_idx == 25
    assert record.image_path == "keyframes/L01_V001/001.jpg"
    assert record.metadata["title"].startswith("Fixture video one")
    assert record.metadata["fps"] == 25.0
    # video without media-info still gets a record
    bare = next(r for r in catalog if r.video_id == "L01_V003")
    assert "title" not in bare.metadata


def test_catalog_round_trip(sample_root: Path, tmp_path: Path):
    catalog = build_catalog(sample_root)
    out = tmp_path / "catalog.jsonl"
    write_catalog(catalog, out)
    loaded = load_catalog(out)
    assert loaded == catalog


def test_catalog_skips_missing_image(dataset_copy: Path):
    (dataset_copy / "keyframes" / "L01_V001" / "001.jpg").unlink()
    catalog = build_catalog(dataset_copy)
    assert "L01_V001:001" not in {r.frame_id for r in catalog}
    assert len(catalog) == 11


def test_manifest_stable_and_sensitive(dataset_copy: Path):
    m1 = build_dataset_manifest(dataset_copy)
    m2 = build_dataset_manifest(dataset_copy)
    assert manifest_hash(m1) == manifest_hash(m2)
    assert m1["n_files"] >= 12  # images + csv + media-info

    # touching content changes the hash
    mapping = dataset_copy / "keyframe_mapping.csv"
    mapping.write_text(
        mapping.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    m3 = build_dataset_manifest(dataset_copy)
    assert manifest_hash(m3) != manifest_hash(m1)


def test_manifest_paths_are_relative_posix(sample_root: Path):
    manifest = build_dataset_manifest(sample_root)
    for path in manifest["files"]:
        assert "\\" not in path
        assert not Path(path).is_absolute()
