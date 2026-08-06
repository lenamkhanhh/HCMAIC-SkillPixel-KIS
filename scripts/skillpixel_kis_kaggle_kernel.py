"""Kaggle script entrypoint: unpack the packaged runners and execute KIS."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE_DATASET = "khanhss/hcmaic-skillpixel-kis-source-20260806"
SOURCE_INPUT = "/kaggle/input/hcmaic-skillpixel-kis-source-20260806"
RAW_INPUT = "/kaggle/input/kis-skillpixel/videos"
QUESTIONS = "/kaggle/input/skillpixel-kis-query-input-20260806/questions.csv"
CORPUS = "/kaggle/input/skillpixel-kis-query-input-20260806/corpus.csv"
RUN_ROOT = "/kaggle/working/skillpixel-kis-run-v1"
MODEL_ROOT = "/kaggle/working/models/siglip2-base-patch16-224"
# The Kaggle job rebuilds the visual index from raw videos.  An index input is
# intentionally not mounted so a stale or teammate-produced artifact cannot
# replace the raw-video source of truth.
INDEX_INPUT = ""
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
REQUESTED_DEVICE = "cuda"


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"command": command[:3] + (["..."] if len(command) > 3 else [])}))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _ensure_dependency(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        print(json.dumps({"installing_dependency": package}))
        _run([sys.executable, "-m", "pip", "install", "--quiet", package])


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _run_optional(
    command: list[str], *, cwd: Path, env: dict[str, str], channel: str
) -> bool:
    try:
        _run(command, cwd=cwd, env=env)
    except subprocess.CalledProcessError as exc:
        print(
            json.dumps(
                {
                    "optional_channel": channel,
                    "status": "unavailable",
                    "error": type(exc).__name__,
                    "returncode": exc.returncode,
                }
            )
        )
        return False
    return True


def _ensure_repository() -> Path:
    repository = Path("/kaggle/working/hcmaic")
    if not repository.exists():
        source = Path(SOURCE_INPUT)
        if not source.exists():
            raise FileNotFoundError(f"Packaged source dataset is not mounted: {source}")
        shutil.copytree(source, repository)
    return repository


def _resolve_input_path(
    configured: str, *, suffix: str | None = None, return_parent: bool = False
) -> str:
    """Resolve Kaggle's mounted slug without changing the SkillPixel source contract."""
    candidate = Path(configured)
    if candidate.exists():
        return str(candidate)
    if not KAGGLE_INPUT_ROOT.exists():
        raise FileNotFoundError(f"Kaggle input root is not mounted: {KAGGLE_INPUT_ROOT}")
    matches = [path for path in KAGGLE_INPUT_ROOT.rglob(suffix or "*") if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Kaggle input path is missing and no fallback was found: {configured}"
        )
    if suffix is None or not return_parent:
        return str(sorted(matches)[0])
    parents: dict[Path, int] = {}
    for match in matches:
        parents[match.parent] = parents.get(match.parent, 0) + 1
    return str(max(parents, key=lambda path: (parents[path], str(path))))


def _resolve_optional_index_input() -> str | None:
    if not INDEX_INPUT:
        return None
    try:
        return _resolve_input_path(INDEX_INPUT, suffix="index.faiss", return_parent=True)
    except FileNotFoundError:
        return None


def _restore_index(run_root: str, index_input: str | None) -> bool:
    if not index_input:
        return False
    source = Path(index_input)
    target = Path(run_root) / "visual" / "V1"
    if target.exists() and any(target.iterdir()):
        return True
    if not source.is_dir():
        raise FileNotFoundError(f"Self-generated index input is not a directory: {source}")
    shutil.copytree(source, target)
    return True


def _download_public_model() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id="google/siglip2-base-patch16-224",
                local_dir=MODEL_ROOT,
                token=token or None,
                max_workers=1,
            )
            return
        except Exception as exc:  # pragma: no cover - depends on Kaggle network
            print(
                json.dumps(
                    {
                        "model_download_attempt": attempt,
                        "model_download_error": type(exc).__name__,
                    }
                )
            )
            if attempt == 3:
                raise
            time.sleep(5 * attempt)


def _device_from_probe(
    *, cuda_available: bool, capability: tuple[int, int] | None, device_name: str | None
) -> tuple[str, dict[str, object]]:
    if not cuda_available:
        return "cpu", {"reason": "cuda_unavailable"}
    if capability is None:
        return "cpu", {"reason": "cuda_capability_unavailable"}
    details: dict[str, object] = {
        "device_name": device_name,
        "compute_capability": f"sm_{capability[0]}{capability[1]}",
    }
    if capability[0] < 7:
        return "cpu", {
            **details,
            "reason": "torch_cuda_build_requires_sm70_or_newer",
        }
    return "cuda", details


def _select_execution_device() -> tuple[str, dict[str, object]]:
    """Use CUDA only when the installed Torch build supports the allocated GPU."""
    try:
        import torch

        available = bool(torch.cuda.is_available())
        capability = torch.cuda.get_device_capability(0) if available else None
        device_name = torch.cuda.get_device_name(0) if available else None
        return _device_from_probe(
            cuda_available=available,
            capability=capability,
            device_name=device_name,
        )
    except Exception as exc:  # pragma: no cover - depends on Kaggle hardware
        return "cpu", {"reason": f"cuda_preflight_failed:{type(exc).__name__}"}


def main() -> int:
    _ensure_dependency("yaml", "pyyaml>=6.0,<7")
    _ensure_dependency("transformers", "transformers>=4.44,<5")
    _ensure_dependency("faiss", "faiss-cpu>=1.8")
    _ensure_dependency("cv2", "opencv-python-headless>=4.9")
    _ensure_dependency("huggingface_hub", "huggingface_hub>=0.24")
    run_ocr = _env_flag("SKILLPIXEL_RUN_OCR", default=True)
    run_object = _env_flag("SKILLPIXEL_RUN_OBJECT", default=True)
    run_asr = _env_flag("SKILLPIXEL_RUN_ASR", default=False)
    allow_optional_download = _env_flag("SKILLPIXEL_ALLOW_MODEL_DOWNLOAD", default=True)
    if run_ocr:
        _ensure_dependency("paddle", "paddlepaddle>=3.0")
        _ensure_dependency("paddleocr", "paddleocr>=3.0")
    if run_object:
        _ensure_dependency("ultralytics", "ultralytics>=8.3")
    if run_asr:
        _ensure_dependency("faster_whisper", "faster-whisper>=1.1")
    repository = _ensure_repository()
    env = os.environ.copy()
    source_path = str(repository / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_path, env.get("PYTHONPATH", "")) if item
    )
    _download_public_model()
    execution_device, device_details = _select_execution_device()
    print(
        json.dumps(
            {
                "requested_device": REQUESTED_DEVICE,
                "selected_device": execution_device,
                "device_details": device_details,
            }
        )
    )
    resolved_raw_input = _resolve_input_path(RAW_INPUT, suffix="*.mp4", return_parent=True)
    resolved_questions = _resolve_input_path(QUESTIONS, suffix="questions.csv")
    resolved_corpus = _resolve_input_path(CORPUS, suffix="corpus.csv")
    resolved_index_input = _resolve_optional_index_input()
    env.update(
        {
            "SKILLPIXEL_RAW_INPUT": resolved_raw_input,
            "SKILLPIXEL_QUESTIONS": resolved_questions,
            "SKILLPIXEL_CORPUS": resolved_corpus,
            "SKILLPIXEL_RUN_ROOT": RUN_ROOT,
            "SKILLPIXEL_PROVIDER": "siglip2",
            "SKILLPIXEL_MODEL_PATH": MODEL_ROOT,
            "SKILLPIXEL_DEVICE": execution_device,
            "SKILLPIXEL_LOCAL_FILES_ONLY": "true",
            "SKILLPIXEL_ALLOW_MODEL_DOWNLOAD": "true" if allow_optional_download else "false",
        }
    )
    config = repository / "configs" / "skillpixel_kis.yaml"
    _run(
        [
            sys.executable,
            "scripts/skillpixel_kis_build.py",
            "--config",
            str(config),
            "--stage",
            "catalog",
        ],
        cwd=repository,
        env=env,
    )
    index_reused = _restore_index(RUN_ROOT, resolved_index_input)
    _run(
        [
            sys.executable,
            "scripts/skillpixel_kis_build.py",
            "--config",
            str(config),
            "--stage",
            "visual",
            "--model-id",
            "V1",
        ],
        cwd=repository,
        env=env,
    )
    optional_stage_status: dict[str, bool] = {}
    if run_ocr:
        optional_stage_status["ocr"] = _run_optional(
            [
                sys.executable,
                "scripts/skillpixel_kis_build.py",
                "--config",
                str(config),
                "--stage",
                "ocr",
                *(["--allow-model-download"] if allow_optional_download else []),
            ],
            cwd=repository,
            env=env,
            channel="ocr",
        )
    if run_object:
        optional_stage_status["object"] = _run_optional(
            [
                sys.executable,
                "scripts/skillpixel_kis_build.py",
                "--config",
                str(config),
                "--stage",
                "object",
                *(["--allow-model-download"] if allow_optional_download else []),
            ],
            cwd=repository,
            env=env,
            channel="object",
        )
    if run_asr:
        optional_stage_status["asr"] = _run_optional(
            [
                sys.executable,
                "scripts/skillpixel_kis_build.py",
                "--config",
                str(config),
                "--stage",
                "asr",
                *(["--allow-model-download"] if allow_optional_download else []),
            ],
            cwd=repository,
            env=env,
            channel="asr",
        )
    _run(
        [
            sys.executable,
            "scripts/skillpixel_kis_infer.py",
            "--config",
            str(config),
            "--run-dir",
            RUN_ROOT,
            "--top-k",
            "100",
        ],
        cwd=repository,
        env=env,
    )
    _run(
        [
            sys.executable,
            "scripts/skillpixel_kis_validate.py",
            "--config",
            str(config),
            "--run-dir",
            RUN_ROOT,
        ],
        cwd=repository,
        env=env,
    )
    manifest = {
        "format": "hcmaic-skillpixel-kis-kaggle-job-v1",
        "source_dataset": SOURCE_DATASET,
        "source_dataset_path": SOURCE_INPUT,
        "raw_input": resolved_raw_input,
        "questions": resolved_questions,
        "corpus": resolved_corpus,
        "index_input": resolved_index_input,
        "index_reused": index_reused,
        "run_root": RUN_ROOT,
        "model_id": "google/siglip2-base-patch16-224",
        "requested_device": REQUESTED_DEVICE,
        "selected_device": execution_device,
        "device_details": device_details,
        "optional_channels_requested": {
            "ocr": run_ocr,
            "object": run_object,
            "asr": run_asr,
        },
        "optional_channel_status": optional_stage_status,
        "allow_optional_model_download": allow_optional_download,
        "training_status": "not_run",
        "raw_video_source": True,
        "btc_artifacts_used": False,
    }
    Path(RUN_ROOT, "kaggle_job_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "run_root": RUN_ROOT, "provider": "siglip2"}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
