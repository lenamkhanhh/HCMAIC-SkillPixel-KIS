"""Build and benchmark an explicit real Jina CLIP v2 SkillPixel candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def jina_provider_kwargs(model: str, *, allow_model_download: bool) -> dict[str, Any]:
    """Return explicit Jina kwargs; this function never enables provider fallback."""
    return {
        "jina_model": model,
        "local_files_only": not allow_model_download,
        "allow_fallback": False,
    }


def main(argv: list[str] | None = None) -> int:
    from hcmaic.benchmark.skillpixel import SkillPixelBenchmarkConfig, benchmark_visual_candidate
    from hcmaic.embedding.factory import get_real_visual_provider
    from hcmaic.skillpixel.index import build_skillpixel_index, load_skillpixel_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model", default="jinaai/jina-clip-v2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args(argv)
    try:
        provider, selection = get_real_visual_provider(
            prefer="jina-clip-v2",
            device=args.device,
            revision=args.revision,
            batch_size=args.batch_size,
            **jina_provider_kwargs(
                args.model,
                allow_model_download=args.allow_model_download,
            ),
        )
        if args.index_dir.is_dir() and any(args.index_dir.iterdir()):
            index = load_skillpixel_index(args.index_dir)
        else:
            index = build_skillpixel_index(args.raw_root, args.index_dir, provider)
        if index.provider_info.get("provider") != provider.name:
            raise RuntimeError("Jina provider/index identity mismatch")
        config = SkillPixelBenchmarkConfig(
            raw_root=args.raw_root,
            index_dir=args.index_dir,
            questions_path=args.questions,
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            top_k=max(100, args.top_k),
        )
        row = benchmark_visual_candidate(
            config,
            provider,
            requested_provider="jina-clip-v2",
            selection=selection,
            variant="jina-clip-v2",
        )
        manifest = {
            "format": "hcmaic-skillpixel-jina-benchmark-v1",
            "provider": provider.info(),
            "selection": selection,
            "index_dir": str(args.index_dir),
            "output_dir": str(args.output_dir),
            "benchmark_row": row,
            "allow_model_download": args.allow_model_download,
            "quality_status": "UNVALIDATED_ON_HCMAIC",
            "training_status": "not_run",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "jina_benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "provider": provider.name,
                    "index_dir": str(args.index_dir),
                    "submission": str(args.output_dir / "submission.csv"),
                    "n_queries": row["n_queries"],
                }
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
