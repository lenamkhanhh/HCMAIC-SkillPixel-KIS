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


def _channel_artifact_dir(config: dict[str, Any], run_dir: Path, channel: str) -> Path:
    configured = str(config.get(f"{channel}_artifact", "")).strip()
    if configured:
        return Path(configured)
    version = str(config.get(f"{channel}_version", "V1")).strip() or "V1"
    return run_dir / "channels" / channel / version


def _load_optional_channels(
    config: dict[str, Any], run_dir: Path, dataset_hash: str
) -> tuple[dict[str, Any], dict[str, str]]:
    from hcmaic.retrieval.asr import ASRArtifactError, ASRRetrievalChannel, load_asr_artifact
    from hcmaic.retrieval.object_retrieval import (
        ObjectArtifactError,
        ObjectRetrievalChannel,
        load_object_artifact,
    )
    from hcmaic.retrieval.ocr_bm25 import BM25OCRChannel, OCRArtifactError, load_ocr_artifact

    channels: dict[str, Any] = {}
    status: dict[str, str] = {
        "ocr": "not_configured",
        "object": "not_configured",
        "asr": "disabled_by_policy",
    }
    ocr_dir = _channel_artifact_dir(config, run_dir, "ocr")
    if (ocr_dir / "ocr_manifest.json").is_file():
        try:
            channels["ocr"] = BM25OCRChannel(
                load_ocr_artifact(ocr_dir, dataset_manifest_hash=dataset_hash)
            )
            status["ocr"] = "ready"
        except OCRArtifactError as exc:
            status["ocr"] = f"unavailable: {type(exc).__name__}: {exc}"
    else:
        status["ocr"] = "not_built"
    object_dir = _channel_artifact_dir(config, run_dir, "object")
    if (object_dir / "object_manifest.json").is_file():
        try:
            channels["object"] = ObjectRetrievalChannel(
                load_object_artifact(object_dir, dataset_manifest_hash=dataset_hash)
            )
            status["object"] = "ready"
        except ObjectArtifactError as exc:
            status["object"] = f"unavailable: {type(exc).__name__}: {exc}"
    else:
        status["object"] = "not_built"
    asr_enabled = _as_bool(config.get("enable_asr_runtime"), default=False)
    asr_dir = _channel_artifact_dir(config, run_dir, "asr")
    if asr_enabled:
        if (asr_dir / "asr_manifest.json").is_file():
            try:
                channels["asr"] = ASRRetrievalChannel(
                    load_asr_artifact(asr_dir, dataset_manifest_hash=dataset_hash)
                )
                status["asr"] = "ready"
            except ASRArtifactError as exc:
                status["asr"] = f"unavailable: {type(exc).__name__}: {exc}"
        else:
            status["asr"] = "not_built"
    return channels, status


def _build_reranker(config: dict[str, Any]) -> Any:
    setting = str(config.get("reranker", "bounded-v1")).strip()
    if setting in {"bounded-v1", "none"}:
        return setting
    if setting not in {"cross-encoder", "cross_encoder"}:
        raise ValueError(f"unsupported reranker: {setting}")
    from hcmaic.retrieval.rerank import CrossEncoderReranker

    model_path_value = str(config.get("reranker_model_path", "")).strip()
    return CrossEncoderReranker(
        model=str(
            config.get(
                "reranker_model",
                CrossEncoderReranker.DEFAULT_MODEL,
            )
        ),
        model_path=Path(model_path_value) if model_path_value else None,
        device=str(config.get("device", "cpu")),
        batch_size=int(config.get("reranker_batch_size", 16)),
        max_length=int(config.get("reranker_max_length", 256)),
        allow_model_download=_as_bool(config.get("allow_model_download"), default=False),
        local_files_only=_as_bool(config.get("local_files_only"), default=True),
    )


def main(argv: list[str] | None = None) -> int:
    from hcmaic.benchmark.hybrid import benchmark_runtime_candidate
    from hcmaic.benchmark.skillpixel import SkillPixelBenchmarkConfig
    from hcmaic.embedding.factory import get_real_visual_provider
    from hcmaic.runtime.kis import KISRuntime
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
        dataset_hash = str(index.index_manifest.get("dataset_manifest_hash", ""))
        optional_channels, channel_status = _load_optional_channels(
            config,
            run_dir,
            dataset_hash,
        )
        fusion_weights = config.get("fusion_weights", {})
        if not isinstance(fusion_weights, dict):
            raise ValueError("config.fusion_weights must be a mapping")
        runtime = KISRuntime.from_components(
            index,
            provider,
            optional_channels=optional_channels,
            provider_selection=selection,
            channel_status=channel_status,
            asr_enabled=_as_bool(config.get("enable_asr_runtime"), default=False),
            max_per_video=(
                int(config["max_per_video"])
                if config.get("max_per_video") not in {None, ""}
                else 5
            ),
            fusion_method=str(config.get("fusion_method", "rrf")),
            fusion_weights={str(key): float(value) for key, value in fusion_weights.items()},
            rank_constant=int(config.get("fusion_rank_constant", 60)),
            candidate_multiplier=int(config.get("candidate_multiplier", 5)),
            reranker=_build_reranker(config),
            rerank_timeout_ms=int(config.get("rerank_timeout_ms", 50)),
        )
        benchmark_config = SkillPixelBenchmarkConfig(
            raw_root=run_dir / "raw",
            index_dir=index_dir,
            questions_path=Path(str(config["questions"])),
            corpus_path=Path(str(config["corpus"])),
            output_dir=run_dir,
            top_k=max(100, args.top_k),
        )
        row = benchmark_runtime_candidate(
            benchmark_config,
            runtime,
            provider,
            requested_provider=provider_name,
            selection=selection,
            channel_status=channel_status,
            variant=str(config.get("model_id", provider_name)),
        )
        variant = str(config.get("model_id", provider_name))
        submission_variant = run_dir / f"submission_{variant}.csv"
        shutil.copyfile(run_dir / "submission.csv", submission_variant)
        manifest = {
            "format": "hcmaic-skillpixel-kis-inference-v2",
            "run_dir": str(run_dir),
            "index_dir": str(index_dir),
            "provider": provider.info(),
            "selection": selection,
            "channel_status": channel_status,
            "runtime_health": runtime.health(),
            "benchmark_row": row,
            "submission": str(submission_variant),
            "quality_status": "UNVALIDATED_ON_HCMAIC",
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
