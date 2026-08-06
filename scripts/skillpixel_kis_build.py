"""Resumable SkillPixel KIS catalog and visual-index runner."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_value(item) for key, item in value.items()}
    return value


def load_skillpixel_config(path: Path) -> dict[str, Any]:
    """Load YAML and expand only environment-backed path/model values."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("SkillPixel config must be a mapping")
    expanded = _expand_value(payload)
    if not isinstance(expanded, dict):  # pragma: no cover - guarded above
        raise ValueError("SkillPixel config must be a mapping")
    return expanded


def provider_model_kwargs(provider: str, model_path: str) -> dict[str, Any]:
    """Return strict provider factory kwargs; no model fallback is implicit."""
    model_key = {
        "siglip2": "siglip2_model",
        "clip": "clip_model",
        "jina-clip-v2": "jina_model",
    }.get(provider)
    if model_key is None:
        raise ValueError(f"unsupported visual provider: {provider}")
    return {model_key: model_path, "allow_fallback": False}


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _code_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _run_root(config: dict[str, Any]) -> Path:
    value = str(config.get("run_root", "")).strip()
    if not value:
        raise ValueError("config.run_root is required")
    return Path(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_preflight(config: dict[str, Any], raw_root: Path) -> Path:
    from hcmaic.skillpixel.raw import validate_raw_dataset
    from hcmaic.skillpixel.retrieval import load_skillpixel_questions

    stats = validate_raw_dataset(raw_root)
    questions_path = Path(str(config["questions"]))
    questions = load_skillpixel_questions(questions_path)
    raw_manifest_path = raw_root / "dataset_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    missing_images = []
    for question in questions:
        if question.task == "VKIS":
            image = Path(question.query_image)
            if not image.is_absolute():
                image = questions_path.parent / image
            if not image.is_file():
                missing_images.append(question.query_id)
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except ImportError:
        torch_info = {"version": None, "cuda_available": False, "error": "torch unavailable"}
    payload = {
        "format": "hcmaic-skillpixel-kis-preflight-v1",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "code_sha": _code_sha(),
        "dataset_id": config.get("dataset_id", "skillpixel-local"),
        "dataset_hash": raw_manifest.get("dataset_hash"),
        "query_hash": None,
        "corpus_hash": None,
        "n_videos": stats.n_videos,
        "n_frames": stats.n_frames,
        "n_queries": len(questions),
        "n_tkis": sum(question.task == "TKIS" for question in questions),
        "n_vkis": sum(question.task == "VKIS" for question in questions),
        "missing_query_images": missing_images,
        "python": sys.version,
        "torch": torch_info,
        "cuda": torch_info.get("cuda_available", False),
        "device": config.get("device", "cpu"),
        "dtype": config.get("dtype", "float32"),
        "batch_size": int(config.get("batch_size", 32)),
        "model_id": config.get("model_id"),
        "model_path": config.get("model_path"),
        "training_status": "not_run",
        "raw_video_source": True,
        "btc_artifacts_used": False,
    }
    output = _run_root(config) / "preflight_report.json"
    _write_json(output, payload)
    return output


def _catalog_stage(config: dict[str, Any]) -> Path:
    from hcmaic.skillpixel.raw import ingest_raw_videos, validate_raw_dataset

    raw_root = _run_root(config) / "raw"
    raw_input = Path(str(config["raw_input"]))
    if raw_root.exists() and any(raw_root.iterdir()):
        validate_raw_dataset(raw_root)
        print(json.dumps({"stage": "catalog", "status": "resumed", "raw_root": str(raw_root)}))
        return raw_root
    ingest_raw_videos(
        raw_input,
        raw_root,
        stride_frames=int(config.get("stride_frames", 10)),
    )
    validate_raw_dataset(raw_root)
    print(json.dumps({"stage": "catalog", "status": "built", "raw_root": str(raw_root)}))
    return raw_root


def _visual_stage(config: dict[str, Any]) -> Path:
    from hcmaic.embedding.factory import get_real_visual_provider
    from hcmaic.skillpixel.index import build_skillpixel_index, load_skillpixel_index
    from hcmaic.skillpixel.raw import validate_raw_dataset

    raw_root = _run_root(config) / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"catalog stage is required first: {raw_root}")
    validate_raw_dataset(raw_root)
    _write_preflight(config, raw_root)
    provider_name = str(config.get("provider", "siglip2"))
    model_path = str(config.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("config.model_path is required for visual stage")
    configured_index = str(config.get("index_dir", "")).strip()
    index_dir = Path(configured_index) if configured_index else (
        _run_root(config) / "visual" / str(config.get("model_id", provider_name))
    )
    if index_dir.exists() and any(index_dir.iterdir()):
        index = load_skillpixel_index(index_dir)
        print(
            json.dumps(
                {
                    "stage": "visual",
                    "status": "resumed",
                    "index_dir": str(index_dir),
                    "n_vectors": index.size,
                }
            )
        )
        return index_dir
    kwargs = provider_model_kwargs(provider_name, model_path)
    provider, selection = get_real_visual_provider(
        prefer=provider_name,
        device=str(config.get("device", "cpu")),
        local_files_only=_as_bool(config.get("local_files_only"), default=True),
        revision=str(config.get("revision", "")).strip() or None,
        batch_size=int(config.get("batch_size", 32)),
        **kwargs,
    )
    index = build_skillpixel_index(raw_root, index_dir, provider)
    _write_json(
        index_dir / "stage_manifest.json",
        {
            "stage": "visual",
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "code_sha": _code_sha(),
            "provider_selection": selection,
            "provider": provider.info(),
            "index_dir": str(index_dir),
            "n_vectors": index.size,
            "training_status": "not_run",
            "raw_video_source": True,
            "btc_artifacts_used": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "visual",
                "status": "built",
                "index_dir": str(index_dir),
                "n_vectors": index.size,
                "dimension": index.dimension,
            }
        )
    )
    return index_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=["catalog", "visual", "ocr", "object", "asr"],
        required=True,
    )
    parser.add_argument("--model-id")
    args = parser.parse_args(argv)
    try:
        config = load_skillpixel_config(args.config)
        if args.model_id:
            config["model_id"] = args.model_id
        if args.stage == "catalog":
            _catalog_stage(config)
        elif args.stage == "visual":
            _visual_stage(config)
        else:
            raise RuntimeError(
                f"{args.stage} stage has no validated real provider in this checkout; "
                "no mock artifact was written"
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
