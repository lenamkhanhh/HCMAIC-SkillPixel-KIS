"""Reproducible local/Kaggle runner contracts for SkillPixel KIS."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.skillpixel_kis_build import (
    _channel_output_dir,
    _channel_stage,
    load_skillpixel_config,
    provider_model_kwargs,
)
from scripts.skillpixel_kis_kaggle_kernel import (
    _device_from_probe,
    _jina_benchmark_command,
    _resolve_input_path,
    _restore_index,
)
from scripts.skillpixel_kis_validate import _resolve_index_dir


def test_runner_config_expands_environment_variables(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "skillpixel_kis.yaml"
    config_path.write_text(
        "raw_input: ${SKILLPIXEL_RAW_INPUT}\n"
        "questions: ${SKILLPIXEL_QUESTIONS}\n"
        "run_root: ${SKILLPIXEL_RUN_ROOT:-runs/default}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SKILLPIXEL_RAW_INPUT", "/kaggle/input/skillpixel/videos")
    monkeypatch.setenv("SKILLPIXEL_QUESTIONS", "/kaggle/input/skillpixel-query/questions.csv")

    config = load_skillpixel_config(config_path)

    assert config["raw_input"] == "/kaggle/input/skillpixel/videos"
    assert config["questions"] == "/kaggle/input/skillpixel-query/questions.csv"
    assert config["run_root"] == "runs/default"


def test_runner_expands_explicit_rerank_timeout(monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "skillpixel_kis.yaml"
    monkeypatch.setenv("SKILLPIXEL_RERANK_TIMEOUT_MS", "5000")

    config = load_skillpixel_config(config_path)

    assert config["rerank_timeout_ms"] == "5000"


def test_runner_model_mapping_is_explicit_and_no_fallback():
    assert provider_model_kwargs("siglip2", "/models/siglip2") == {
        "siglip2_model": "/models/siglip2",
        "allow_fallback": False,
    }


def test_optional_channel_stage_wiring_is_explicit_and_versioned(tmp_path: Path, monkeypatch):
    raw_root = tmp_path / "run" / "raw"
    raw_root.mkdir(parents=True)
    output_dir = tmp_path / "run" / "channels" / "ocr" / "V2"
    captured = {}

    class FakeOCRProvider:
        def __init__(self, **kwargs):
            captured["provider"] = kwargs

    class FakeResult:
        status = "built"
        artifact_dir = output_dir
        manifest_path = output_dir / "channel_stage_manifest.json"
        dataset_hash = "raw-hash"
        n_input_frames = 3
        n_records = 2
        details = {"provider": "fake-real-ocr"}

    def fake_build(raw, output, provider, *, batch_size):
        captured["build"] = {
            "raw": raw,
            "output": output,
            "provider": provider,
            "batch_size": batch_size,
        }
        return FakeResult()

    monkeypatch.setattr(
        "hcmaic.retrieval.real_channels.PaddleOCRFrameProvider", FakeOCRProvider
    )
    monkeypatch.setattr("hcmaic.retrieval.channel_runner.build_ocr_channel", fake_build)

    result = _channel_stage(
        {
            "run_root": str(tmp_path / "run"),
            "ocr_version": "V2",
            "ocr_model_path": str(tmp_path / "ocr-model"),
            "device": "cpu",
            "ocr_batch_size": 4,
        },
        "ocr",
        allow_model_download=False,
    )

    assert _channel_output_dir(
        {"run_root": str(tmp_path / "run"), "ocr_version": "V2"}, "ocr"
    ) == output_dir
    assert captured["provider"] == {
        "model_version": "PP-OCRv6",
        "model_path": tmp_path / "ocr-model",
        "device": "cpu",
        "allow_model_download": False,
    }
    assert captured["build"]["raw"] == raw_root
    assert captured["build"]["output"] == output_dir
    assert captured["build"]["batch_size"] == 4
    assert result.status == "built"
    assert provider_model_kwargs("clip", "/models/clip") == {
        "clip_model": "/models/clip",
        "allow_fallback": False,
    }


def test_kaggle_runner_resolves_mounted_video_parent(tmp_path: Path, monkeypatch):
    mounted = tmp_path / "input" / "kis-skillpixel" / "videos"
    mounted.mkdir(parents=True)
    (mounted / "video7020.mp4").write_bytes(b"fixture")
    monkeypatch.setattr(
        "scripts.skillpixel_kis_kaggle_kernel.KAGGLE_INPUT_ROOT",
        tmp_path / "input",
    )

    assert _resolve_input_path(
        "/kaggle/input/kis-skillpixel/videos", suffix="*.mp4", return_parent=True
    ) == str(mounted)


def test_kaggle_runner_resolves_query_file_not_parent(tmp_path: Path, monkeypatch):
    query_dir = tmp_path / "input" / "datasets" / "khanhss" / "skillpixel-query"
    query_dir.mkdir(parents=True)
    questions = query_dir / "questions.csv"
    questions.write_text("query_id,task\nq1,TKIS\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.skillpixel_kis_kaggle_kernel.KAGGLE_INPUT_ROOT",
        tmp_path / "input",
    )

    assert _resolve_input_path(
        "/kaggle/input/skillpixel-query/questions.csv", suffix="questions.csv"
    ) == str(questions)


def test_kaggle_runner_records_cpu_fallback_for_unsupported_gpu():
    device, details = _device_from_probe(
        cuda_available=True,
        capability=(6, 0),
        device_name="Tesla P100-PCIE-16GB",
    )

    assert device == "cpu"
    assert details["reason"] == "torch_cuda_build_requires_sm70_or_newer"


def test_kaggle_runner_keeps_cuda_for_supported_gpu():
    device, details = _device_from_probe(
        cuda_available=True,
        capability=(7, 5),
        device_name="Tesla T4",
    )

    assert device == "cuda"
    assert details["compute_capability"] == "sm_75"


def test_kaggle_runner_jina_command_is_explicit_and_no_fallback():
    command = _jina_benchmark_command(
        python="python",
        raw_root="/run/raw",
        index_dir="/run/visual/jina-clip-v2",
        output_dir="/run/jina-clip-v2",
        questions="/query/questions.csv",
        corpus="/query/corpus.csv",
        device="cuda",
    )

    assert command[:2] == ["python", "scripts/skillpixel_kis_jina_benchmark.py"]
    assert "--allow-model-download" in command
    assert "--allow-fallback" not in command


def test_validator_resolves_default_index_relative_to_run_dir(tmp_path: Path):
    assert _resolve_index_dir({"model_id": "V1", "index_dir": ""}, tmp_path) == (
        tmp_path / "visual" / "V1"
    )


def test_kaggle_runner_restores_self_generated_index(tmp_path: Path):
    source = tmp_path / "index-source"
    source.mkdir()
    (source / "index.faiss").write_bytes(b"index")
    run_root = tmp_path / "run"

    assert _restore_index(str(run_root), str(source)) is True
    assert (run_root / "visual" / "V1" / "index.faiss").read_bytes() == b"index"
