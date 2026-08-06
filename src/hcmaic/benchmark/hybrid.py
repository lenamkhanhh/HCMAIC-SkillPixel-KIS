"""Execution and evidence export for the canonical KIS hybrid runtime."""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hcmaic.benchmark.skillpixel import (
    SkillPixelBenchmarkConfig,
    SkillPixelBenchmarkError,
    _code_sha,
    _dir_size_mb,
    _mapping_validation,
    _model_registry_entry,
    _rss_mb,
    _sha256_file,
    _write_evidence_csv,
    _write_jsonl,
)
from hcmaic.contracts.kis import KISQuery, KISResult
from hcmaic.runtime.kis import KISRuntime
from hcmaic.skillpixel.retrieval import load_skillpixel_questions
from hcmaic.skillpixel.submission import (
    export_skillpixel_submission,
    validate_submission_csv,
)

QUALITY_STATUS = "UNVALIDATED_ON_HCMAIC"


def _write_hybrid_checksums(output_dir: Path) -> Path:
    """Hash only output contracts; raw/index artifacts have their own manifests."""
    checksum_path = Path(output_dir) / "checksums.sha256"
    lines = []
    for path in sorted(Path(output_dir).iterdir()):
        if not path.is_file() or path == checksum_path:
            continue
        lines.append(f"{_sha256_file(path)}  {path.name}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return checksum_path


def _resolve_query_image(query_image: str, query_root: Path) -> Path:
    path = Path(query_image)
    return path if path.is_absolute() else query_root / path


def _make_queries(config: SkillPixelBenchmarkConfig) -> list[KISQuery]:
    questions = load_skillpixel_questions(config.questions_path)
    query_root = Path(config.questions_path).parent
    return [
        KISQuery(
            query_id=question.query_id,
            task=question.task,
            text=question.text or None,
            image_path=(
                _resolve_query_image(question.query_image, query_root)
                if question.task == "VKIS"
                else None
            ),
            top_k=config.top_k,
        )
        for question in questions
    ]


def _write_runtime_results(
    path: Path, queries: list[KISQuery], outputs: Mapping[str, Any]
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for query in queries:
            output = outputs[query.query_id]
            payload = {
                "query_id": query.query_id,
                "task": query.task,
                "answers": [result.to_dict() for result in output.results],
                "executed_channels": list(output.executed_channels),
                "unavailable_channels": dict(output.unavailable_channels),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _catalog_lookup(index: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    id_map_by_frame = {str(row["frame_uid"]): row for row in index.id_map}
    catalog_by_frame = {record.frame_id: record for record in index.catalog}
    return id_map_by_frame, catalog_by_frame


def _channel_scores(result: KISResult) -> dict[str, float | None]:
    return {
        "visual_score": result.channel_scores.get("visual"),
        "ocr_score": result.channel_scores.get("ocr"),
        "object_score": result.channel_scores.get("object"),
        "asr_score": result.channel_scores.get("asr"),
        "rrf_score": result.fused_score,
        "rerank_score": result.rerank_score,
    }


def _evidence_rows(
    queries: list[KISQuery],
    outputs: Mapping[str, Any],
    *,
    config: SkillPixelBenchmarkConfig,
    provider: Any,
    id_map_by_frame: Mapping[str, Any],
    catalog_by_frame: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider_info = provider.info()
    model = str(provider_info.get("model_name", provider.name))
    revision = str(
        provider_info.get("model_revision", provider_info.get("revision", provider.version))
    )
    evidence_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for query_order, query in enumerate(queries):
        output = outputs[query.query_id]
        status_rows.append(
            {
                "query_id": query.query_id,
                "query_type": query.task,
                "query_order": query_order,
                "status": "ok" if len(output.results) >= config.top_k else "short_results",
                "error": (
                    None
                    if len(output.results) >= config.top_k
                    else "fewer than top_k results"
                ),
                "n_results": len(output.results),
                "top_k": config.top_k,
                "channels": list(output.executed_channels),
                "unavailable_channels": dict(output.unavailable_channels),
                "provider": provider.name,
                "model": model,
                "revision": revision,
            }
        )
        for result in output.results[: config.top_k]:
            id_map = id_map_by_frame.get(result.frame_uid)
            record = catalog_by_frame.get(result.frame_uid)
            if id_map is None or record is None:
                raise SkillPixelBenchmarkError(
                    f"hybrid result frame_uid is absent from index mapping: {result.frame_uid}"
                )
            scores = _channel_scores(result)
            evidence_rows.append(
                {
                    "query_id": query.query_id,
                    "query_type": query.task,
                    "query_order": query_order,
                    "rank": result.rank,
                    "video_id": result.video_id,
                    "video_filename": result.video_filename,
                    "keyframe_id": record.keyframe_id,
                    "frame_uid": result.frame_uid,
                    "source_frame_idx": result.source_frame_idx,
                    "timestamp_ms": result.timestamp_ms,
                    "preview_path": str((Path(config.raw_root) / record.image_path).resolve()),
                    "image_path": record.image_path,
                    **scores,
                    "provider": provider.name,
                    "model": model,
                    "revision": revision,
                    "faiss_row": int(id_map["faiss_row"]),
                    "feature_row": int(id_map["feature_row"]),
                }
            )
    return evidence_rows, status_rows


def _channel_manifests(runtime: KISRuntime) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for name, channel in runtime.orchestrator.optional_channels.items():
        artifact = getattr(channel, "artifact", None)
        manifest = getattr(artifact, "manifest", None)
        if isinstance(manifest, dict):
            manifests[name] = dict(manifest)
        else:
            manifests[name] = {"status": "ready", "provider": getattr(channel, "provider", None)}
    return manifests


def benchmark_runtime_candidate(
    config: SkillPixelBenchmarkConfig,
    runtime: KISRuntime,
    provider: Any,
    *,
    requested_provider: str,
    selection: Mapping[str, Any],
    channel_status: Mapping[str, str] | None = None,
    variant: str = "hybrid-V1",
) -> dict[str, Any]:
    """Run mixed TKIS/VKIS through visual + configured channels and export all evidence."""
    queries = _make_queries(config)
    if not queries:
        raise SkillPixelBenchmarkError("questions.csv produced no KIS queries")
    started = time.perf_counter()
    outputs = runtime.search_queries(queries)
    batch_ms = (time.perf_counter() - started) * 1000.0
    if list(outputs) != [query.query_id for query in queries]:
        raise SkillPixelBenchmarkError("hybrid runtime changed query order or IDs")
    if any(len(outputs[query.query_id].results) < config.top_k for query in queries):
        raise SkillPixelBenchmarkError("hybrid runtime returned fewer than top_k results")

    candidate_dir = Path(config.output_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    results_path = candidate_dir / "results.jsonl"
    _write_runtime_results(results_path, queries, outputs)
    submission_path = candidate_dir / "submission.csv"
    export_skillpixel_submission(
        config.questions_path,
        results_path,
        config.corpus_path,
        submission_path,
    )
    validation = validate_submission_csv(
        submission_path,
        config.questions_path,
        config.corpus_path,
    )
    if not validation.ok:
        raise SkillPixelBenchmarkError(
            f"hybrid submission validation failed: {list(validation.errors)}"
        )
    (candidate_dir / "validation.json").write_text(
        json.dumps(
            {"valid": True, "n_queries": validation.n_queries, "errors": []},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    index = runtime.index
    id_map_by_frame, catalog_by_frame = _catalog_lookup(index)
    evidence_rows, status_rows = _evidence_rows(
        queries,
        outputs,
        config=config,
        provider=provider,
        id_map_by_frame=id_map_by_frame,
        catalog_by_frame=catalog_by_frame,
    )
    evidence_top100 = candidate_dir / "retrieval_evidence_top100.jsonl"
    evidence_top20 = candidate_dir / "retrieval_evidence_top20.jsonl"
    evidence_top100_csv = candidate_dir / "retrieval_evidence_top100.csv"
    evidence_top20_csv = candidate_dir / "retrieval_evidence_top20.csv"
    query_status = candidate_dir / "query_status.jsonl"
    _write_jsonl(evidence_top100, evidence_rows)
    top20_rows = [row for row in evidence_rows if int(row["rank"]) <= 20]
    _write_jsonl(evidence_top20, top20_rows)
    _write_evidence_csv(evidence_top100_csv, evidence_rows)
    _write_evidence_csv(evidence_top20_csv, top20_rows)
    _write_jsonl(query_status, status_rows)

    raw_manifest = json.loads(
        (Path(config.raw_root) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    mapping = _mapping_validation(index)
    if not mapping["ok"]:
        raise SkillPixelBenchmarkError(f"index mapping validation failed: {mapping['errors']}")
    visual_registry = _model_registry_entry(
        provider=provider,
        requested_provider=requested_provider,
        selection=selection,
    )
    channel_manifests = _channel_manifests(runtime)
    model_registry = {
        "format": "hcmaic-skillpixel-kis-model-registry-v2",
        "quality_status": QUALITY_STATUS,
        "training_status": "not_run",
        "visual": visual_registry,
        "channels": channel_manifests,
        "channel_status": dict(channel_status or runtime.channel_status),
        "fusion": {
            "method": runtime.orchestrator.fusion_method,
            "rank_constant": runtime.orchestrator.rank_constant,
            "weights": dict(runtime.orchestrator.fusion_weights),
            "reranker": runtime.orchestrator.reranker,
            "rerank_timeout_ms": runtime.orchestrator.rerank_timeout_ms,
        },
    }
    model_registry_path = candidate_dir / "model_registry.json"
    model_registry_path.write_text(
        json.dumps(model_registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight = {
        "format": "hcmaic-skillpixel-kis-hybrid-preflight-v1",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "code_sha": _code_sha(),
        "quality_status": QUALITY_STATUS,
        "training_status": "not_run",
        "raw_video_source": True,
        "btc_artifacts_used": False,
        "dataset_hash": raw_manifest.get("dataset_hash"),
        "sampling_policy": raw_manifest.get("sampling_policy"),
        "n_videos": raw_manifest.get("n_videos"),
        "n_sampled_frames": raw_manifest.get("n_frames"),
        "query_order": [query.query_id for query in queries],
        "query_order_preserved": list(outputs) == [query.query_id for query in queries],
        "index_dir": str(Path(config.index_dir).resolve()),
        "index_type": "IndexFlatIP",
        "n_vectors": index.size,
        "embedding_dimension": index.dimension,
        "mapping_validation": mapping,
        "runtime_health": runtime.health(),
        "channels": channel_manifests,
        "fusion": model_registry["fusion"],
    }
    preflight_path = candidate_dir / "preflight_report.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resources = {
        "format": "hcmaic-skillpixel-resource-v1",
        "device": provider.info().get("device"),
        "provider": provider.name,
        "ram_mb": _rss_mb(),
        "disk_mb": _dir_size_mb(candidate_dir),
        "query_batch_ms": round(batch_ms, 3),
        "query_mean_ms": round(batch_ms / len(queries), 3),
    }
    resource_path = candidate_dir / "resource_report.json"
    resource_path.write_text(
        json.dumps(resources, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums_path = _write_hybrid_checksums(candidate_dir)
    executed_channels = sorted(
        {channel for output in outputs.values() for channel in output.executed_channels}
    )
    return {
        "run_id": candidate_dir.name,
        "variant": variant,
        "kind": "hybrid",
        "status": "validated-local",
        "provider_execution": "validated-local",
        "quality_status": QUALITY_STATUS,
        "channels": "+".join(executed_channels),
        "provider": provider.name,
        "model": visual_registry.get("model_id"),
        "revision": visual_registry.get("revision"),
        "index_type": "IndexFlatIP",
        "embedding_dimension": index.dimension,
        "n_queries": len(queries),
        "n_tkis": sum(query.task == "TKIS" for query in queries),
        "n_vkis": sum(query.task == "VKIS" for query in queries),
        "n_vectors": index.size,
        "mapping_errors": mapping["n_errors"],
        "query_batch_ms": round(batch_ms, 3),
        "query_mean_ms": round(batch_ms / len(queries), 3),
        "ram_mb": resources["ram_mb"],
        "disk_mb": resources["disk_mb"],
        "submission_validation": "pass",
        "official_skillpixel_score": None,
        "metrics": {},
        "evidence_top100": str(evidence_top100),
        "evidence_top20": str(evidence_top20),
        "query_status": str(query_status),
        "preflight_report": str(preflight_path),
        "model_registry": str(model_registry_path),
        "resource_report": str(resource_path),
        "checksums": str(checksums_path),
        "submission": str(submission_path),
        "channel_status": dict(channel_status or runtime.channel_status),
    }
