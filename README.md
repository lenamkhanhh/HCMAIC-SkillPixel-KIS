# HCMAIC keyframe-search MVP

Local, reproducible keyframe text-search system for HCMAIC 2026 (Bảng A).
Fork of [SoftSignalsRetrievalSystems-AIC2025](https://github.com/hhlearntocode/SoftSignalsRetrievalSystems-AIC2025)
(MIT, pinned commit in [UPSTREAM.md](UPSTREAM.md)), refactored into a tested,
self-contained package.

```text
keyframes + mapping CSV + optional metadata
  -> validate -> catalog -> embeddings -> index
  -> CLI / FastAPI / operator UI -> submission preview -> evaluator
```

Everything below runs offline on CPU with the committed fixture dataset —
no network, GPU, FFmpeg, model weights, or private data required.

## Bootstrap (Windows, uv)

```powershell
cd system
uv python install 3.11
uv sync                      # core + dev (test) dependencies
```

Optional extras (not needed for any test):

```powershell
uv sync --extra clip         # real CLIP ViT-B/32 provider (torch, transformers)
uv sync --extra faiss        # FAISS index provider (faiss-cpu)
```

## Quick start on the fixture

```powershell
uv run hcmaic validate-data --input data/sample
uv run hcmaic build-index   --input data/sample --output artifacts/sample
uv run hcmaic search        --index artifacts/sample --query "red bus on the street" --top-k 5
uv run hcmaic serve         --index artifacts/sample --port 8000
# then open http://127.0.0.1:8000/
uv run hcmaic evaluate      --index artifacts/sample --queries data/sample/queries.jsonl --qrels data/sample/qrels.jsonl
```

`serve` hosts the operator UI (query box, top-K grid, frame detail, video
timeline, query history, canonical submission preview) plus the JSON API:
`GET /health`, `GET /system/info`, `POST /search`, `GET /frames/{frame_id}`,
`GET /videos/{video_id}/timeline`, `POST /submit/preview`.

## Raw video ingestion (Milestone 1)

Turn raw videos (MP4/MKV/AVI/MOV) into a searchable dataset:

```powershell
uv sync --extra video      # OpenCV fallback backend (skip if FFmpeg is on PATH)
uv run hcmaic ingest-video --input <video-file-or-dir> --output data/myset --interval 2.0
uv run hcmaic validate-data --input data/myset
uv run hcmaic build-index  --input data/myset --output artifacts/myset
uv run hcmaic search       --index artifacts/myset --query "..." --top-k 10
```

Backends: FFmpeg CLI when `ffmpeg`+`ffprobe` are on PATH, otherwise the
pure-pip OpenCV wheel. Uniform time sampling + near-duplicate removal;
per-video results, warnings, and failures land in
`<output>/ingest_report.json`. Re-ingesting an existing video requires
`--force`.

## Real (BTC-style) data

Point `--input` at a directory with the BTC conventions:

```text
<dataset>/
├── keyframes/<video_id>/<nnn>.jpg
├── keyframe_mapping.csv        # video_id,n,pts_time,fps,frame_idx  (or per-video map-keyframes/<video_id>.csv)
└── media-info/<video_id>.json  # optional YouTube-style metadata
```

Build with the real CLIP provider once `--extra clip` is installed:

```powershell
uv run hcmaic build-index --input <dataset> --output artifacts/real --provider clip
```

The first clip run downloads `openai/clip-vit-base-patch32` (~600 MB) from
Hugging Face. CPU works; CUDA is used automatically when available (batch
size is capped for 4 GB VRAM).

## Tests and verification

```powershell
uv run pytest                          # full offline suite
uv run pytest --cov=src/hcmaic         # with coverage
uv run ruff check src tests
uv run mypy src
.\scripts\verify.ps1                   # full verification loop
```

## Documents

- [GOAL.md](GOAL.md) — mission definition of done
- [UPSTREAM.md](UPSTREAM.md) — provenance, reuse, deviations
- [DECISIONS.md](DECISIONS.md) — architecture decisions
- [PROGRESS.md](PROGRESS.md) — command-level log
- [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) — latest gate results
- [FINAL_HANDOFF.md](FINAL_HANDOFF.md) — recovery + next tickets per role

The fixture proves plumbing only. It says nothing about competition
retrieval quality — benchmark on real BTC data before drawing conclusions.
