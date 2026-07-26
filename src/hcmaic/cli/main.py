"""hcmaic CLI.

Commands:
  ingest-video  --input <video file|dir> --output <dataset>
                [--interval 2.0] [--max-frames 500] [--video-id ID] [--force]
  validate-data --input <dataset>
  build-index   --input <dataset> --output <artifacts>
                [--provider mock|clip] [--index exact-numpy|faiss]
  search        --index <artifacts> --query "<text>" [--top-k 10] [--video-id V1,V2]
  serve         --index <artifacts> [--host 127.0.0.1] [--port 8000] [--data-root <path>]
  evaluate      --index <artifacts> --queries <queries.jsonl> --qrels <qrels.jsonl> [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_ingest(args: argparse.Namespace) -> int:
    from hcmaic.ingestion.video import IngestError, ingest_dataset

    try:
        results, failures = ingest_dataset(
            Path(args.input),
            Path(args.output),
            video_id=args.video_id,
            interval_s=args.interval,
            max_frames=args.max_frames,
            force=args.force,
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for result in results:
        info = result.info
        print(
            f"Ingested {info.video_id}: {result.n_kept} keyframe(s) "
            f"({result.n_candidates} candidates, {result.n_duplicates} "
            f"near-duplicates dropped), {info.width}x{info.height} "
            f"@ {info.fps:.2f} fps, {info.duration_s:.1f}s, "
            f"backend={info.backend}."
        )
        for warning in result.warnings:
            print(f"  warn: {warning}")
    for failure in failures:
        print(f"  FAILED {failure['file']}: {failure['error']}", file=sys.stderr)
    print(f"Dataset: {args.output}  (report: {Path(args.output) / 'ingest_report.json'})")
    if results:
        print(
            f"Next: uv run hcmaic validate-data --input {args.output} && "
            f"uv run hcmaic build-index --input {args.output} --output <artifacts>"
        )
    return 0 if not failures else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    from hcmaic.ingestion.validator import validate_dataset, write_validation_report

    root = Path(args.input)
    if not root.is_dir():
        print(f"error: dataset root {root} is not a directory", file=sys.stderr)
        return 2
    report = validate_dataset(root, check_images=not args.skip_image_check)
    out = Path(args.report) if args.report else root / "validation_report.json"
    write_validation_report(report, out)
    print(
        f"Validated {root}: {report.n_videos} videos, {report.n_frames} frames, "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)."
    )
    for issue in report.errors:
        print(f"  ERROR [{issue.code}] {issue.message}")
    for issue in report.warnings[: args.max_warnings]:
        print(f"  warn  [{issue.code}] {issue.message}")
    hidden = len(report.warnings) - args.max_warnings
    if hidden > 0:
        print(f"  ... {hidden} more warning(s) in {out}")
    print(f"Report: {out}")
    return 0 if report.ok else 1


def _cmd_build_index(args: argparse.Namespace) -> int:
    from hcmaic.embedding.base import get_provider
    from hcmaic.indexing.artifacts import build_index_artifacts
    from hcmaic.ingestion.catalog import build_catalog
    from hcmaic.ingestion.validator import validate_dataset

    root = Path(args.input)
    report = validate_dataset(root, check_images=True)
    if not report.ok:
        print(
            f"error: dataset has {len(report.errors)} validation error(s); "
            f"run 'hcmaic validate-data --input {root}' for details.",
            file=sys.stderr,
        )
        return 1
    catalog = build_catalog(root)
    provider = get_provider(args.provider)
    out_dir = build_index_artifacts(
        root, catalog, provider, Path(args.output), index_provider=args.index
    )
    print(
        f"Built index artifacts in {out_dir}: {len(catalog)} frames, "
        f"dim {provider.dimension}, provider {provider.version}, "
        f"index {args.index}."
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from hcmaic.contracts.models import SearchRequest
    from hcmaic.retrieval.service import load_service

    service = load_service(
        Path(args.index),
        dataset_root=Path(args.data_root) if args.data_root else None,
        index_provider=args.index_provider,
    )
    filters = {}
    if args.video_id:
        filters["video_ids"] = args.video_id
    request = SearchRequest(
        query_id=args.query_id, text=args.query, top_k=args.top_k, filters=filters
    )
    results = service.search(request)
    if args.json:
        print(
            json.dumps(
                [r.model_dump() for r in results], indent=2, ensure_ascii=False
            )
        )
    else:
        print(f"Query: {args.query!r}  top_k={args.top_k}  index={service.index_version}")
        if not results:
            print("(no results)")
        for r in results:
            print(
                f"  #{r.rank:<3} {r.frame_id:<20} score={r.final_score:.4f} "
                f"t={r.timestamp_ms}ms idx={r.frame_idx}"
            )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from hcmaic.api.app import create_app

    app = create_app(
        Path(args.index),
        dataset_root=Path(args.data_root) if args.data_root else None,
        index_provider=args.index_provider,
    )
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from hcmaic.evaluation.evaluator import (
        evaluate,
        format_summary,
        load_qrels,
        load_queries,
        write_reports,
    )
    from hcmaic.retrieval.service import load_service

    service = load_service(
        Path(args.index),
        dataset_root=Path(args.data_root) if args.data_root else None,
    )
    queries = load_queries(Path(args.queries))
    qrels = load_qrels(Path(args.qrels))
    report, per_query = evaluate(service, queries, qrels, top_k=args.top_k)
    out_dir = Path(args.out) if args.out else Path(args.index) / "evaluation"
    report_path, per_query_path = write_reports(report, per_query, out_dir)
    print(format_summary(report))
    print(f"Reports: {report_path} , {per_query_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcmaic", description="HCMAIC keyframe-search MVP"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "ingest-video",
        help="Extract keyframes from raw videos (MP4/MKV/AVI/MOV) into a dataset",
    )
    p.add_argument("--input", required=True, help="Video file or directory of videos")
    p.add_argument("--output", required=True, help="Dataset directory to create/extend")
    p.add_argument(
        "--interval", type=float, default=2.0,
        help="Sampling interval in seconds (default: 2.0)",
    )
    p.add_argument(
        "--max-frames", type=int, default=500,
        help="Maximum keyframes per video (default: 500)",
    )
    p.add_argument(
        "--video-id", help="Override the video id (single-file ingest only)"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Replace keyframes/mapping if the video was already ingested",
    )
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("validate-data", help="Validate a dataset directory")
    p.add_argument("--input", required=True)
    p.add_argument("--report", help="Path for validation_report.json")
    p.add_argument("--skip-image-check", action="store_true")
    p.add_argument("--max-warnings", type=int, default=10)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("build-index", help="Validate, embed, and write artifacts")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--provider", choices=["mock", "clip"], default="mock")
    p.add_argument("--index", choices=["exact-numpy", "faiss"], default="exact-numpy")
    p.set_defaults(func=_cmd_build_index)

    p = sub.add_parser("search", help="Search an index from the command line")
    p.add_argument("--index", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--video-id", help="Restrict to comma-separated video ids")
    p.add_argument("--query-id", default="cli")
    p.add_argument("--data-root", help="Override dataset root for image paths")
    p.add_argument("--index-provider", choices=["exact-numpy", "faiss"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("serve", help="Serve the API and operator UI")
    p.add_argument("--index", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-root", help="Override dataset root for image paths")
    p.add_argument("--index-provider", choices=["exact-numpy", "faiss"])
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("evaluate", help="Run the evaluator")
    p.add_argument("--index", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--qrels", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", help="Output directory for reports")
    p.add_argument("--data-root")
    p.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
