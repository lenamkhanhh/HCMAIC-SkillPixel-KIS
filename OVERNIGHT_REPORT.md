# Overnight report — Milestone 1 session (2026-07-27)

Session goal: raw video → keyframe → mapping → validation → embedding →
retrieval smoke test, hardened, without breaking the existing baseline.

## Baseline at session start

- Branch `hcmaic-2026-foundation` at `a3f5025` (tag `v0-handoff` backup
  bundle exists in `artifacts/checkpoints/`).
- 118 tests passed, ruff clean, mypy clean, working tree clean.

## Files added

- `src/hcmaic/ingestion/video.py` — raw video ingestion module
  (probe, ffmpeg/opencv backends, uniform sampling, dedup, mapping +
  media-info + ingest-report writers, batch ingest).
- `tests/test_video_ingest.py` — 24 tests incl. parametrized cases.
- `tests/test_video_e2e.py` — CLI end-to-end + error-path tests.
- `MILESTONE_1_IMPLEMENTATION.md`, `OVERNIGHT_REPORT.md` (this file).

## Files modified

- `src/hcmaic/cli/main.py` — new `ingest-video` subcommand + help text.
- `src/hcmaic/ingestion/catalog.py` — `_METADATA_KEYS` += width, height.
- `pyproject.toml` — new `[video]` extra (`opencv-python-headless`);
  `opencv-python-headless` added to the dev group (tests generate fixture
  videos); `charset-normalizer>=3.3,<3.4.9` added to the `clip` extra.
- `README.md` — "Raw video ingestion (Milestone 1)" section.
- `DECISIONS.md` — D9 (backend order, uniform sampling, dedup, PTS rules).
- `uv.lock` — resolved for the above.

## Dependency changes

| Package | Why |
|---|---|
| `opencv-python-headless` 5.0.0 (extra `video` + dev) | Pure-pip video decode fallback; FFmpeg is not installed on this machine |
| `charset-normalizer` 3.4.7 (extra `clip`, pinned `<3.4.9`) | Fixes `RequestsDependencyWarning` — 3.4.9 dropped `__version__`, which requests' compat check reads |

## Commands run (essentials)

```text
uv run pytest / ruff check / mypy src            (before + after every phase)
uv sync --extra clip --extra faiss --extra video
uv run hcmaic ingest-video --input artifacts/smoke-videos/demo_video.avi \
    --output artifacts/smoke-dataset --interval 1.0
uv run hcmaic validate-data --input artifacts/smoke-dataset
uv run hcmaic build-index --input artifacts/smoke-dataset --output artifacts/smoke-index
uv run hcmaic search --index artifacts/smoke-index --query "a blue frame" --top-k 3
```

## Test results

- Before: 118 passed. After: **144 passed, 0 failed, 0 skipped locally**
  (video tests auto-skip only where cv2 is absent).
- Coverage 92% total; `video.py` 81% (uncovered lines are the ffmpeg
  backend, absent on this machine).
- ruff + mypy clean.
- Two test bugs found and fixed during the session (stdout parsing across
  capsys, missing tmp dir in a test); one real security bug found and fixed
  (explicit `--video-id` was not validated → path traversal, now rejected +
  regression tests).

## E2E result

CLI: two synthetic videos (3-color 4.5 s + 1-color 2 s) → `ingest-video` →
`validate-data` 0 errors → mock index → "a blue frame" returns the blue
segment of the right video (t within 1.0–3.0 s), "yellow sand" returns the
other video; timeline ordered; broken-video batch exits 1 with a written
`ingest_report.json`.

## Blockers

- None open. FFmpeg absence was handled by design (OpenCV fallback);
  a stale `hcmaic.exe` process briefly blocked `uv sync` (killed PID, noted).

## How to re-run everything

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC\system
uv sync --extra clip --extra faiss --extra video
uv run pytest
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
```

## Still not done (honest)

- ffmpeg backend never executed here (no ffmpeg on PATH) — runs
  automatically on a machine that has it.
- No shot detection; uniform sampling only.
- VFR handling is best-effort PTS with a recorded warning; unverified on a
  real VFR file.
- All retrieval evidence is fixture-scale. **This system is not AIC-ready**;
  it is a verified local foundation for the team to build on.
