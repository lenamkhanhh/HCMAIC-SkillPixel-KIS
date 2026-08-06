"""Validate SkillPixel KIS mapping, evidence, index and submission artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcmaic.benchmark.skillpixel import _mapping_validation
from scripts.skillpixel_kis_build import load_skillpixel_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(run_dir: Path) -> list[str]:
    path = run_dir / "checksums.sha256"
    if not path.is_file():
        return ["missing checksums.sha256"]
    errors: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = run_dir / relative
        if not target.is_file():
            errors.append(f"checksum target missing: {relative}")
        elif _sha256(target) != expected:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def _resolve_index_dir(config: dict[str, Any], run_dir: Path) -> Path:
    configured_index = str(config.get("index_dir", "")).strip()
    if configured_index:
        return Path(configured_index)
    return run_dir / "visual" / str(config.get("model_id", config.get("provider", "siglip2")))


def main(argv: list[str] | None = None) -> int:
    from hcmaic.skillpixel.index import load_skillpixel_index
    from hcmaic.skillpixel.raw import validate_raw_dataset
    from hcmaic.skillpixel.submission import validate_submission_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    errors: list[str] = []
    try:
        config = load_skillpixel_config(args.config)
        raw_root = run_dir / "raw"
        index_dir = _resolve_index_dir(config, run_dir)
        raw_stats = validate_raw_dataset(raw_root)
        index = load_skillpixel_index(index_dir)
        mapping = _mapping_validation(index)
        if not mapping["ok"]:
            errors.append(f"mapping validation failed: {mapping['errors']}")
        questions_path = Path(str(config["questions"]))
        corpus_path = Path(str(config["corpus"]))
        variant = config.get("model_id", config.get("provider", "siglip2"))
        submission = run_dir / f"submission_{variant}.csv"
        if not submission.is_file():
            submission = run_dir / "submission.csv"
        submission_report = validate_submission_csv(submission, questions_path, corpus_path)
        errors.extend(submission_report.errors)
        required = (
            "retrieval_evidence_top100.jsonl",
            "retrieval_evidence_top100.csv",
            "retrieval_evidence_top20.jsonl",
            "query_status.jsonl",
            "preflight_report.json",
            "model_registry.json",
            "checksums.sha256",
        )
        errors.extend(
            f"missing artifact: {name}"
            for name in required
            if not (run_dir / name).is_file()
        )
        errors.extend(_verify_checksums(run_dir))
        report: dict[str, Any] = {
            "valid": not errors,
            "errors": errors,
            "raw": {"n_videos": raw_stats.n_videos, "n_frames": raw_stats.n_frames},
            "index": {
                "n_vectors": index.size,
                "dimension": index.dimension,
                "mapping": mapping,
            },
            "submission": {
                "path": str(submission),
                "valid": submission_report.ok,
                "n_queries": submission_report.n_queries,
            },
            "quality_status": "UNVALIDATED_ON_HCMAIC",
            "training_status": "not_run",
            "raw_video_source": True,
            "btc_artifacts_used": False,
        }
        (run_dir / "validation_final.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 2
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
