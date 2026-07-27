# Milestone 1 — Raw video → keyframe → mapping → validation → embedding → retrieval

> Đây là implementation report của milestone cũ. Runbook hiện tại bằng tiếng
> Việt nằm tại `docs/competitive_v1/03_SETUP_AND_RUNBOOK.md`.

Date: 2026-07-27. Status: **implemented and verified on local fixtures**
(synthetic videos; not verified on BTC-scale real data).

## What was built

```text
raw video (MP4/MKV/AVI/MOV)
  -> probe metadata (width/height/fps/frame_count/duration, backend)
  -> uniform time sampling (default 2 s, --interval)
  -> near-duplicate drop (mean abs grayscale diff, 32x32 thumb, thr 2.0)
  -> keyframes/<video_id>/NNN.jpg
  -> keyframe_mapping.csv  (video_id,n,pts_time,fps,frame_idx,width,height)
  -> media-info/<video_id>.json (title,length,width,height,fps,backend,source_file)
  -> ingest_report.json (per-video counts, warnings, failures)
  -> existing validate-data / build-index / search pipeline, unchanged
```

## Module map

| File | Role |
|---|---|
| `src/hcmaic/ingestion/video.py` | Probe + extraction backends, dedup, mapping/metadata writers, batch ingest |
| `src/hcmaic/cli/main.py` | New `ingest-video` subcommand (`--input --output --interval --max-frames --video-id --force`) |
| `src/hcmaic/ingestion/catalog.py` | `_METADATA_KEYS` extended with `width`, `height` |
| `tests/test_video_ingest.py` | 24 unit tests (probe, sanitize, dedup, force, batch, traversal) |
| `tests/test_video_e2e.py` | CLI end-to-end: 2 videos → ingest → validate → index → search → timeline |

## Backend contract

1. **ffmpeg** — chosen when `ffmpeg` and `ffprobe` are both on PATH
   (`ffprobe` JSON probe; `fps=1/interval` filter extraction).
2. **opencv** — fallback via `uv sync --extra video`
   (`opencv-python-headless`; wheel bundles codecs, PTS from
   `CAP_PROP_POS_MSEC`, one recorded warning + `frame_idx/fps` fallback if
   the codec exposes no PTS).
3. Neither → `IngestError` telling the operator both install options.

On this machine FFmpeg is absent, so the OpenCV path is the one exercised
by tests; the ffmpeg functions are covered only by code review (see
"Not done" below).

## Failure handling

- Corrupt/empty/unsupported files → per-file `IngestError` collected in
  `ingest_report.json`; a batch continues past failures (exit code 1).
- Negative or beyond-duration timestamps → frame dropped + warning;
  kept timestamps are clamped into the declared duration so the existing
  validator always passes on ingested output.
- Duplicate consecutive frames → deterministic dedup (counted in report).
- Re-ingesting an existing `video_id` → refused without `--force`; with
  `--force` old frames and mapping rows are replaced atomically enough
  (dir removed, rows rewritten).
- Explicit `--video-id` is validated against the same regex as the dataset
  validator — path traversal (`../evil`, `a/b`, `x:y`) is rejected before
  anything is written.
- `media-info` records only the source file *name*, never a local path.

## Verification (all run on 2026-07-27)

```text
uv run pytest                       -> 144 passed, 0 failed
uv run pytest --cov=src/hcmaic      -> 92% total (video.py 81%; uncovered = ffmpeg-backend paths, no ffmpeg on this machine)
uv run ruff check src tests scripts -> clean
uv run mypy src                     -> clean
```

Manual smoke (OpenCV backend): 12-frame 4-color AVI → 6 candidates @1 s →
2 duplicates dropped → 4 keyframes → validate 0 errors → mock index →
query "a blue frame" → rank 1 = the blue keyframe at t=2000 ms.

## Not done / follow-ups

- ffmpeg backend untested on this machine (no ffmpeg installed); needs one
  run on a machine with FFmpeg — the tests auto-exercise it there.
- No shot detection (PySceneDetect/TransNetV2) — uniform sampling only;
  TV2 ticket.
- VFR videos rely on decoder PTS; the CFR fallback warns but is not
  verified against a real VFR file.
- MKV/H.264 depends on the backend's codec support; fixture uses MJPG/AVI
  (guaranteed in the OpenCV wheel). Real BTC videos should be smoke-tested.
- Not AIC-ready: everything above is fixture-scale evidence only.
