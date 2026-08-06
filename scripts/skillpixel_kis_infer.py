"""Run full TKIS/VKIS inference and evidence export from a persisted index."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.skillpixel_kis_build import (
    _as_bool,
    load_skillpixel_config,
    provider_model_kwargs,
)


def _resolve_index(config: dict[str, Any], run_dir: Path) -> Path:
    configured = str(config.get("index_dir", "")).strip()
    if configured:
        return Path(configured)
    return run_dir / "visual" / str(config.get("model_id", config.get("provider", "siglip2")))


def main(argv: list[str] | None = None) -> int:
    from hcmaic.benchmark.skillpixel import SkillPixelBenchmarkConfig, benchmark_visual_candidate
    from hcmaic.embedding.factory import get_real_visual_provider
    from hcmaic.skillpixel.index import load_skillpixel_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        config = load_skillpixel_config(args.config)
        run_dir = Path(args.run_dir)
        index_dir = _resolve_index(config, run_dir)
        index = load_skillpixel_index(index_dir)
        provider_name = str(config.get("provider", index.provider_info.get("provider", "siglip2")))
        model_path = str(config.get("model_path", "")).strip()
        kwargs = provider_model_kwargs(provider_name, model_path)
        provider, selection = get_real_visual_provider(
            prefer=provider_name,
            device=str(config.get("device", "cpu")),
            local_files_only=_as_bool(config.get("local_files_only"), default=True),
            revision=str(config.get("revision", "")).strip()
            or str(index.provider_info.get("model_revision", ""))
            or None,
            batch_size=int(config.get("batch_size", 32)),
            **kwargs,
        )
        benchmark_config = SkillPixelBenchmarkConfig(
            raw_root=run_dir / "raw",
            index_dir=index_dir,
            questions_path=Path(str(config["questions"])),
            corpus_path=Path(str(config["corpus"])),
            output_dir=run_dir,
            top_k=max(100, args.top_k),
        )
        row = benchmark_visual_candidate(
            benchmark_config,
            provider,
            requested_provider=provider_name,
            selection=selection,
            variant=str(config.get("model_id", provider_name)),
        )
        variant = str(config.get("model_id", provider_name))
        submission_variant = run_dir / f"submission_{variant}.csv"
        shutil.copyfile(run_dir / "submission.csv", submission_variant)
        manifest = {
            "format": "hcmaic-skillpixel-kis-inference-v1",
            "run_dir": str(run_dir),
            "index_dir": str(index_dir),
            "provider": provider.info(),
            "selection": selection,
            "benchmark_row": row,
            "submission": str(submission_variant),
            "training_status": "not_run",
        }
        (run_dir / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "run_dir": str(run_dir),
                    "submission": str(submission_variant),
                    "n_queries": row["n_queries"],
                }
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
