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


def _download_public_model() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    snapshot_download(
        repo_id="google/siglip2-base-patch16-224",
        local_dir=MODEL_ROOT,
        token=token or None,
        local_dir_use_symlinks=False,
    )


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
    env.update(
        {
            "SKILLPIXEL_RAW_INPUT": RAW_INPUT,
            "SKILLPIXEL_QUESTIONS": QUESTIONS,
            "SKILLPIXEL_CORPUS": CORPUS,
            "SKILLPIXEL_RUN_ROOT": RUN_ROOT,
            "SKILLPIXEL_SIGLIP2_MODEL": MODEL_ROOT,
            "SKILLPIXEL_DEVICE": "cuda",
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
        "raw_input": RAW_INPUT,
        "questions": QUESTIONS,
        "corpus": CORPUS,
        "run_root": RUN_ROOT,
        "model_id": "google/siglip2-base-patch16-224",
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
