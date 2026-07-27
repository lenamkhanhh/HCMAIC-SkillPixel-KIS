# 03 — Setup and runbook (Windows PowerShell)

## Fresh environment

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC\system
uv python install 3.11
uv sync --locked --extra faiss --extra video
```

Add `--extra clip` only when a real CLIP smoke is intended. Optional provider
doctor does not fetch weights:

```powershell
uv run hcmaic provider-doctor --provider siglip2
uv run hcmaic provider-doctor --provider jina-clip-v2
```

## Fixture and standard path

```powershell
uv run python scripts/make_fixture.py
uv run hcmaic validate-data --input data/sample
uv run hcmaic build-index --input data/sample --output artifacts/sample
uv run hcmaic search --index artifacts/sample --query "a red bus" --top-k 10
uv run hcmaic serve --index artifacts/sample --port 8000
```

## Raw video

```powershell
uv run hcmaic ingest-video --input <video-or-dir> --output data/myset --interval 2
uv run hcmaic validate-data --input data/myset
```

`--force` generates and validates staging output before replacing live files.
FFmpeg is preferred when both FFmpeg and ffprobe exist; otherwise OpenCV is
used. This machine currently exercises OpenCV.

## Benchmarks

```powershell
uv run hcmaic benchmark --config configs/competitive_v1.yaml `
  --out artifacts/benchmark/competitive-v1

uv run hcmaic scale-benchmark --vectors 10000 --dimension 512 `
  --queries 100 --top-k 100 --out artifacts/benchmark/scale.json
```

## Full verification

```powershell
uv run pytest
uv run pytest --cov=src/hcmaic
uv run ruff check src tests scripts
uv run mypy src
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv pip check
git diff --check
```

If `uv pip check` reports duplicate `charset-normalizer` metadata, do not delete
the active environment in place during a run. Close processes using `.venv`,
rename it as a backup, then run a fresh locked `uv sync` and re-run all gates.

CPU is enough for mock/fixture verification. CUDA/model-scale claims require an
explicit GPU run and recorded hardware.
