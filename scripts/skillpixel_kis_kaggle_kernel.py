"""Kaggle script entrypoint: unpack the packaged runners and execute KIS."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_DATASET = "khanhss/hcmaic-skillpixel-kis-source-20260806"
SOURCE_INPUT = "/kaggle/input/hcmaic-skillpixel-kis-source-20260806"
RAW_INPUT = "/kaggle/input/kis-skillpixel/videos"
QUESTIONS = "/kaggle/input/skillpixel-kis-query-input-20260806/questions.csv"
CORPUS = "/kaggle/input/skillpixel-kis-query-input-20260806/corpus.csv"
RUN_ROOT = "/kaggle/working/skillpixel-kis-run-v1"
MODEL_ROOT = "/kaggle/working/models/siglip2-base-patch16-224"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
REQUESTED_DEVICE = "cuda"


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"command": command[:3] + (["..."] if len(command) > 3 else [])}))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _ensure_dependency(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        _run([sys.executable, "-m", "pip", "install", "--quiet", package])


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


def _download_public_model() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    snapshot_download(
        repo_id="google/siglip2-base-patch16-224",
        local_dir=MODEL_ROOT,
        token=token or None,
        local_dir_use_symlinks=False,
    )


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
    env.update(
        {
            "SKILLPIXEL_RAW_INPUT": resolved_raw_input,
            "SKILLPIXEL_QUESTIONS": resolved_questions,
            "SKILLPIXEL_CORPUS": resolved_corpus,
            "SKILLPIXEL_RUN_ROOT": RUN_ROOT,
            "SKILLPIXEL_SIGLIP2_MODEL": MODEL_ROOT,
            "SKILLPIXEL_DEVICE": execution_device,
            "SKILLPIXEL_LOCAL_FILES_ONLY": "true",
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
        "run_root": RUN_ROOT,
        "model_id": "google/siglip2-base-patch16-224",
        "requested_device": REQUESTED_DEVICE,
        "selected_device": execution_device,
        "device_details": device_details,
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
