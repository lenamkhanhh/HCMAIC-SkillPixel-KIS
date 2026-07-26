"""Full fixture E2E through the CLI: validate -> build -> search -> API -> evaluate."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from hcmaic.api.app import create_app
from hcmaic.cli.main import main


def test_full_pipeline_via_cli(dataset_copy: Path, tmp_path: Path, capsys):
    artifacts = tmp_path / "artifacts"

    # 1. validate
    assert main(["validate-data", "--input", str(dataset_copy)]) == 0
    report = json.loads(
        (dataset_copy / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["errors"] == []

    # 2. build catalog + embeddings + index artifacts
    assert (
        main(
            [
                "build-index",
                "--input", str(dataset_copy),
                "--output", str(artifacts),
            ]
        )
        == 0
    )
    assert (artifacts / "embeddings.npy").is_file()

    # 3. CLI search returns the correctly mapped frame
    capsys.readouterr()  # clear validate/build output before parsing search JSON
    assert (
        main(
            [
                "search",
                "--index", str(artifacts),
                "--query", "a solid red keyframe",
                "--top-k", "3",
                "--data-root", str(dataset_copy),
                "--json",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    results = json.loads(out[out.index("[") :])
    assert results[0]["frame_id"] == "L01_V001:001"
    assert results[0]["timestamp_ms"] == 1000

    # 4. API search over the same artifacts
    app = create_app(artifacts, dataset_root=dataset_copy)
    client = TestClient(app)
    api_results = client.post(
        "/search", json={"text": "a solid red keyframe", "top_k": 3}
    ).json()["results"]
    assert [r["frame_id"] for r in api_results] == [
        r["frame_id"] for r in results
    ], "CLI and API must rank identically"
    # UI, timeline, image, submission preview all reachable
    assert client.get("/").status_code == 200
    assert client.get("/videos/L01_V001/timeline").json()["n_frames"] == 3
    assert client.get("/frames/L01_V001:001/image").status_code == 200
    preview = client.post(
        "/submit/preview",
        json={"query_id": "q1", "task_type": "kis", "frame_id": "L01_V001:001"},
    ).json()
    assert preview["video_id"] == "L01_V001"

    # 5. evaluate
    eval_out = tmp_path / "eval"
    assert (
        main(
            [
                "evaluate",
                "--index", str(artifacts),
                "--queries", str(dataset_copy / "queries.jsonl"),
                "--qrels", str(dataset_copy / "qrels.jsonl"),
                "--out", str(eval_out),
                "--data-root", str(dataset_copy),
            ]
        )
        == 0
    )
    eval_report = json.loads(
        (eval_out / "evaluation_report.json").read_text(encoding="utf-8")
    )
    assert eval_report["mode"] == "deterministic-mock"
    assert eval_report["n_scored"] == 6


def test_validate_exit_code_on_broken_data(dataset_copy: Path):
    (dataset_copy / "keyframes" / "L01_V001" / "001.jpg").unlink()
    assert main(["validate-data", "--input", str(dataset_copy)]) == 1


def test_build_index_refuses_broken_data(dataset_copy: Path, tmp_path: Path):
    (dataset_copy / "keyframes" / "L01_V001" / "001.jpg").unlink()
    assert (
        main(
            [
                "build-index",
                "--input", str(dataset_copy),
                "--output", str(tmp_path / "artifacts"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "artifacts" / "embeddings.npy").exists()


def test_validate_missing_dir_exit_2(tmp_path: Path):
    assert main(["validate-data", "--input", str(tmp_path / "nope")]) == 2


def test_search_missing_artifacts_exit_2(tmp_path: Path):
    assert (
        main(["search", "--index", str(tmp_path / "empty"), "--query", "x"]) == 2
    )
