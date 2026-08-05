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

    assert ingest.command == "ingest-raw"
    assert ingest.stride_frames == 10
    assert build.command == "build-skillpixel-index"
    assert retrieve.command == "retrieve-skillpixel"
    assert export.command == "export-skillpixel"
