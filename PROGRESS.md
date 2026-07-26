# Progress log

Newest entries at the bottom. Every entry: what ran, what resulted.

## 2026-07-26 Phase 0 — Bootstrap

- Read required context (`TEAM_PIPELINE_TASKS_5_NGUOI.md`, `TEAM_TASK_BOARD.md`,
  `00_NOTE_TONG_HOP.md`, SkillPixel Buoi 1+2 notes, BTC training texts 1-2,
  `lab1-fastapi-student/`). `Learn/` untouched.
- `system/` confirmed absent before start (`ls system` → exit 2).
- `git clone https://github.com/hhlearntocode/SoftSignalsRetrievalSystems-AIC2025 system`
  → OK; `git checkout e2c52124e691fc2c71d187d8f587fbe1bcddc38b` → HEAD at
  `e2c5212 clean`. LICENSE (MIT) present. History preserved.
- `git checkout -b hcmaic-2026-foundation` → branch created.
- Baseline inspection (reading, not executing — the heavy model download was
  skipped deliberately):
  - `app.py`: hard-coded `D:/...` DB/index/keyframes paths; `EMBEDDING_DIM=1280`;
    `laion/CLIP-ViT-bigG-14` default; reads table `keyframe_embeddings`;
    `googletrans` imported but absent from `requirements.txt`; startup
    swallows exceptions; `StaticFiles(directory="D:/keyframes")` mounted at
    import time → import fails without that directory.
  - `migrate_embeddings.py`: writes table `keyframes`, dim 512, local
    `image_retrieval.db` → serving/migration mismatch confirmed on all three.
  - No tests/CI/lockfile/evaluator anywhere in the tree.
- Tooling: git 2.52.0, uv 0.11.20, global Python 3.14.1 (not used),
  cpython 3.11 available for uv download. Git identity present (`Khanhdz`),
  not modified.
- Created `UPSTREAM.md`, `GOAL.md`, `DECISIONS.md`, `PROGRESS.md`.
- Moved upstream runnables to `upstream_reference/` via `git mv` (history
  kept); removed `requirements.txt` (preserved in git history; replaced by
  `pyproject.toml` + `uv.lock`).
- `uv python install 3.11` → cpython-3.11.15 installed.
- `uv sync --python 3.11` → .venv created, core+dev deps locked and synced.
- Commit `03b560c` = Phase 0 checkpoint. Phase 0 exit gate: PASS
  (pinned upstream + license + provenance intact, baseline documented from
  inspection, checkpoint restorable via git, `Learn/` untouched).

## 2026-07-26 Phases 1–4 — Contracts, fixture, providers, index, API, CLI

- Implemented `src/hcmaic/`: contracts (pydantic v2), mapping parser
  (single-file + per-video BTC layouts), validator (10 failure modes),
  catalog.jsonl builder, dataset manifest (sha256), mock + real-CLIP
  embedding providers, ExactNumpyIndex + optional FaissIndex, versioned
  index artifacts with load-time consistency gates, RetrievalService,
  FastAPI app, argparse CLI.
- `uv run python scripts/make_fixture.py` → fixture: 5 videos, 12 keyframes,
  6 queries + qrels committed under `data/sample/`.
- `uv run hcmaic validate-data --input data/sample` → 0 errors, 2 expected
  warnings (missing optional metadata).
- `uv run hcmaic build-index --input data/sample --output artifacts/sample`
  → 12 frames, dim 64, mock-palette-v1, exact-numpy.
- `uv run hcmaic search --index artifacts/sample --query "a solid red keyframe"`
  → rank 1 = `L01_V001:001` (the red keyframe) score 0.9947; video filter
  (`--video-id L01_V004`) correctly restricts results.

## 2026-07-26 Phase 5 — Operator UI (verified in a real browser)

- Plain HTML/CSS/JS UI served by FastAPI at `/`.
- `uv run hcmaic serve --index artifacts/sample --port 8017` then Chrome
  (browser pane): system info loaded (12 frames · 5 videos · index version),
  video-filter dropdown populated from `/system/info`; search rendered the
  top-10 grid (#1 = L01_V001:001 score 0.9947); card click opened the detail
  pane with metadata + image (naturalWidth > 0 → image actually served);
  timeline strip ordered 1.0s*(current), 5.0s, 9.0s; submission preview
  rendered the CanonicalSubmission JSON with the search's query_id; video
  filter through the UI returned only L01_V004; empty query produced the
  error state "Enter a query first."; query history persisted in
  localStorage.

## 2026-07-26 Phase 6 — Evaluator

- `uv run hcmaic evaluate --index artifacts/sample --queries data/sample/queries.jsonl --qrels data/sample/qrels.jsonl`
  → mode deterministic-mock, 6/6 scored, Recall@1/5/10 = 1.0, MRR = 1.0,
  p50 0.165 ms, p95 0.325 ms; reports written
  (evaluation_report.json + per_query_results.jsonl). Fixture was NOT tuned
  after observing evaluation results (queries/qrels authored before the
  first evaluator run; only the plumbing was fixed afterwards).

## 2026-07-26 Phase 7 — Tests, lint, types

- `uv run pytest` → 105 passed (mock-only), then 111 passed with FAISS
  extra installed. 0 failed, 0 skipped (mock env skips the 6 FAISS tests).
- `uv run pytest --cov=src/hcmaic` → 95% total coverage (target ≥80%);
  only optional heavy providers (clip_real, faiss_index) excluded from
  measurement, faiss covered by its optional test module.
- `uv run ruff check src tests scripts` → clean (after 3 fixes).
- `uv run mypy src` → clean (after 1 fix: PIL Resampling enum).
- Commit `d372be4` = Phases 1–6 checkpoint.

## 2026-07-26 Optional paths

- FAISS: `uv sync --extra faiss` → faiss-cpu 1.14.3 installed cleanly on
  Windows/py3.11. Equivalence vs ExactNumpyIndex verified on 7 queries +
  filtered search (identical ids, scores within 1e-5); CLI
  `build-index --index faiss` + `search --index-provider faiss` verified.
- CUDA: not attempted (PyPI Windows torch wheel is CPU-only; CUDA recorded
  as unverified optional path — see FINAL_HANDOFF.md ticket for TV3).

## 2026-07-26 Independent audit and handoff

- Clean locked sync, lint, mypy, core tests, coverage, fixture E2E, and optional
  FAISS parity all pass. The final combined suite is 118 passed; core coverage
  is 95%.
- Added regression coverage and fixes for artifact drift, unsafe embeddings,
  provider-version drift, API path leakage, whitespace queries, timing
  validation, metadata HTML injection, and the browser favicon 404.
- `pip-audit` reports no known vulnerabilities after updating Pillow to 12.3.0.
- Playwright smoke verified search, filtering, detail image, timeline,
  submission preview, and zero console errors on the local server.
- `VERIFICATION_REPORT.md` and `FINAL_HANDOFF.md` describe evidence, limits,
  and the five team tickets. `Learn/` remained untouched.
- A release archive and SHA-256 manifest are created under `artifacts/` after
  this source checkpoint; generated artifacts are excluded from Git.

## 2026-07-26 Final packaging verification

- `scripts/verify.ps1` full run → all gates pass (env/import, mypy, ruff,
  118 tests + 95% coverage, fixture E2E validate/build/search/evaluate).
- Canonical package rebuilt from HEAD:
  `git archive --format=zip -o artifacts/hcmaic-system-v0.zip HEAD
  ":(exclude)upstream_reference/src/transnetv2/transnetv2-pytorch-weights.pth"`
  → 124 files; contains source, tests, fixture data, scripts, docs,
  pyproject + uv.lock; excludes weights/.venv/generated artifacts.
- `SHA256SUMS.txt` regenerated for both archives; `sha256sum -c` → OK for
  hcmaic-system-v0.zip and hcmaic-system-v1.zip.
- Gotcha recorded: `uv sync --extra faiss` (without `--extra clip`) prunes
  torch/transformers — always pass every wanted extra in one sync command.

## 2026-07-27 Real CLIP smoke (optional path)

- `uv sync --extra clip --extra faiss` → torch 2.13.0+cpu, transformers
  4.57.6, cuda: False (PyPI CPU wheel; CUDA remains unverified/optional).
- `uv run hcmaic build-index --input data/sample --output artifacts/sample-clip --provider clip`
  → downloaded openai/clip-vit-base-patch32 (~600 MB, HF cache 1.2 GB),
  built 12 frames dim 512; index_manifest records model/device/batch.
- `uv run hcmaic search --index artifacts/sample-clip --query "a solid red image" --top-k 3`
  → #1 L01_V001:001 (correct red keyframe) score 0.2877.
- `uv run hcmaic evaluate --index artifacts/sample-clip ...` → mode
  real-clip-smoke, 6/6 scored, Recall@1/5/10 = 1.0, MRR = 1.0,
  p50 18.855 ms, p95 43.043 ms. Smoke only — no BTC-scale quality claim.
- All optional paths now resolved: FAISS verified, real CLIP smoke verified,
  CUDA unavailable on this wheel (documented for TV3).
