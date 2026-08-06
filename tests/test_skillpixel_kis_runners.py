"""Reproducible local/Kaggle runner contracts for SkillPixel KIS."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.skillpixel_kis_build import load_skillpixel_config, provider_model_kwargs
from scripts.skillpixel_kis_kaggle_kernel import _resolve_input_path


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


def test_runner_model_mapping_is_explicit_and_no_fallback():
    assert provider_model_kwargs("siglip2", "/models/siglip2") == {
        "siglip2_model": "/models/siglip2",
        "allow_fallback": False,
    }
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

    assert _resolve_input_path("/kaggle/input/kis-skillpixel/videos", suffix="*.mp4") == str(
        mounted
    )
