"""CLI surface for the reproducible SkillPixel command chain."""

from __future__ import annotations

from hcmaic.cli.main import build_parser


def test_skillpixel_cli_commands_are_available():
    parser = build_parser()

    ingest = parser.parse_args(
        ["ingest-raw", "--input", "raw", "--output", "generated", "--stride-frames", "10"]
    )
    build = parser.parse_args(
        ["build-skillpixel-index", "--input", "generated", "--output", "index"]
    )
    retrieve = parser.parse_args(
        [
            "retrieve-skillpixel",
            "--index",
            "index",
            "--questions",
            "questions.csv",
            "--results",
            "results.jsonl",
        ]
    )
    export = parser.parse_args(
        [
            "export-skillpixel",
            "--queries",
            "questions.csv",
            "--results",
            "results.jsonl",
            "--corpus",
            "corpus.csv",
            "--output",
            "submission.csv",
        ]
    )
    kis_search = parser.parse_args(
        [
            "search-kis",
            "--index",
            "index",
            "--task",
            "TKIS",
            "--query",
            "a person",
        ]
    )
    kis_retrieve = parser.parse_args(
        [
            "retrieve-kis",
            "--index",
            "index",
            "--questions",
            "questions.csv",
            "--results",
            "results.jsonl",
        ]
    )
    kis_export = parser.parse_args(
        [
            "export-kis",
            "--index",
            "index",
            "--questions",
            "questions.csv",
            "--corpus",
            "corpus.csv",
            "--output",
            "submission.csv",
        ]
    )
    kis_serve = parser.parse_args(
        ["serve-kis", "--index", "index", "--port", "8010"]
    )
    kis_benchmark = parser.parse_args(
        [
            "benchmark-kis",
            "--index",
            "index",
            "--questions",
            "questions.csv",
        ]
    )
    sota_benchmark = parser.parse_args(
        [
            "benchmark-skillpixel",
            "--raw",
            "raw",
            "--index",
            "index",
            "--questions",
            "questions.csv",
            "--corpus",
            "corpus.csv",
            "--out",
            "benchmark",
        ]
    )

    assert ingest.command == "ingest-raw"
    assert ingest.stride_frames == 10
    assert build.command == "build-skillpixel-index"
    assert retrieve.command == "retrieve-skillpixel"
    assert export.command == "export-skillpixel"
    assert kis_search.command == "search-kis"
    assert kis_retrieve.command == "retrieve-kis"
    assert kis_export.command == "export-kis"
    assert kis_serve.command == "serve-kis"
    assert kis_benchmark.command == "benchmark-kis"
    assert sota_benchmark.command == "benchmark-skillpixel"
