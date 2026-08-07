# HCMAIC SkillPixel KIS — implementation and verification report

Date: 2026-08-06  
Repository: [HCMAIC-SkillPixel-KIS](https://github.com/lenamkhanhh/HCMAIC-SkillPixel-KIS)  
Branch: `main`  
Quality status: `UNVALIDATED_ON_HCMAIC`

## Executive result

The raw-video-first SkillPixel TKIS/VKIS path is implemented and has been run
locally end to end. The verified run used 250 raw videos, 9,835 sampled frames,
100 queries (50 TKIS and 50 VKIS), real local SigLIP2, normalized float32 exact
FAISS `IndexFlatIP`, strict ID mapping, hybrid runtime routing, evidence export,
and submission validation. The output contains 100 answers for every query and
passes the final validator.

No official HCMAIC qrels were available. Therefore this report does not claim
Recall, MRR, SOTA, top-1 quality, or contest score. The only quality statement
that is justified is `UNVALIDATED_ON_HCMAIC`.

## Data and mapping contract

Raw videos remain the source of truth. The local command used the SkillPixel raw
video directory under `learn/skillpixel/Buoi_08_Mock_contest_KIS/DataMockTest`;
it did not read BTC keyframes, teammate mappings, CLIP feature files, or a
prebuilt external index.

The persisted mapping is:

```text
faiss_row -> feature_row -> frame_uid -> video_id -> video_filename
          -> keyframe_id/source_frame_idx -> timestamp_ms
```

The raw catalog records `n`, `pts_time`, `fps`, `frame_idx`,
`source_frame_idx`, `timestamp_ms`, and `sampling_policy`. Sampling in the
verified run was `uniform_stride_10_v1`. The round-trip test and final validator
checked `faiss_ntotal=9835`, `id_map_rows=9835`, `n_errors=0`; submission IDs
are generated from `video_filename,source_frame_idx`, never from `faiss_row`.

## Verified local run

Artifact root:

`D:\Kaggle\skillpixel-kis-local-infer-v8b`

Important outputs:

- raw manifest: `raw\dataset_manifest.json`
- FAISS catalog: `visual\V1\catalog.jsonl`
- normalized vectors: `visual\V1\embeddings.npy`
- mapping: `visual\V1\id_map.json`
- exact index: `visual\V1\index.faiss`
- index manifest: `visual\V1\index_manifest.json`
- top-100 evidence: `retrieval_evidence_top100.jsonl` and `.csv`
- top-20 evidence: `retrieval_evidence_top20.jsonl` and `.csv`
- query status: `query_status.jsonl`
- model registry: `model_registry.json`
- preflight/resource reports: `preflight_report.json`, `resource_report.json`
- submission: `submission_V1.csv` and `submission.csv`
- final validation: `validation_final.json`
- checksums: `checksums.sha256`

Measured output:

| Field | Result |
| --- | --- |
| Raw videos | 250 |
| Sampled frames/vectors | 9,835 |
| Index | exact FAISS `IndexFlatIP` |
| Embedding dimension | 768 |
| Provider | real SigLIP2 |
| Model | local `D:\Models\hf\siglip2-base-patch16-224` |
| Revision | `main` |
| Device | CPU; no fallback |
| Query mix | 50 TKIS + 50 VKIS |
| Query batch time | about 13.97 s |
| Peak reported RAM | about 1,206 MB |
| Submission | 100 queries, 101 CSV lines including header |
| Mapping errors | 0 |
| Final validation | `valid=true` |

SHA-256 from the verified run:

- `submission_V1.csv`: `563BD4EF467008BCC368DFFE91068BE3D20F1BADF98E328B69B16EE57AB165B4`
- `retrieval_evidence_top20.jsonl`: `852830A044E00B085C2B3511A6780F71E99860C232962B37F4EF7A6F12C3C485`
- `checksums.sha256`: `D6B967F46ACDA7DFBB75C21FD13929FD915D84739008728548A42993132A6DB1`

The checksum contract excludes only files written after the benchmark
(`inference_manifest.json` and `validation_final.json`); those files are
validated structurally, while stable evidence/artifact files remain hashed.

## Implemented channels and actual provider status

| Channel | Implementation | Actual evidence in this run |
| --- | --- | --- |
| Visual | SigLIP2 image/text provider, L2 normalization, exact FAISS | Ran successfully: real 768D provider, 9,835 vectors |
| Visual fallback | Real CLIP provider with explicit selection metadata | Implemented; not silently selected in the verified SigLIP2 run |
| Jina CLIP v2 | Native image/text adapter and strict benchmark runner | Local cache probe failed explicitly; no CLIP fallback; Kaggle benchmark prepared but no published result yet |
| OCR/BM25 | Real PaddleOCR adapter plus BM25 artifact/channel | Local weights unavailable; no fake artifact written; Kaggle v11 requested it, output still pending |
| Object | Real Ultralytics adapter plus object retrieval artifact/channel | Local weights unavailable; no fake artifact written; Kaggle v11 requested it, output still pending |
| ASR | Real faster-whisper adapter with timestamp-to-frame mapping | Disabled by policy locally; no audio/transcript artifact available |
| Fusion | Visual plus optional channel scores, RRF, dedup/diversity and bounded rerank | Executed with visual channel; optional channels are fail-closed and recorded |
| Reranker | Explicit real CrossEncoder path, local-cache/download policy | Implemented and tested; not promoted without qrels and not included in the verified baseline |

The local optional-stage commands failed closed with the expected errors:
PP-OCR model not cached, Ultralytics weights not cached, and Whisper weights not
cached. None of those failures produced an empty/mock submission artifact.

## Kaggle execution

Kaggle authentication was already verified by the user. The allowed API smoke
test `kaggle datasets list -p 1` / `kaggle kernels list -p 1` passed. The old CLI
version warning is informational only.

Source-only package uploaded:

`khanhss/hcmaic-skillpixel-kis-source-20260806`

The package contains only source/config/runner files. It excludes raw videos,
weights, embeddings, FAISS indexes, secrets, and tokens. The current package was
built from public-repo commit `811264b` and includes the explicit Jina runner.

Kernel:

`khanhss/skillpixel-kis-gpu-end-to-end-v1`

The v11 baseline job was dispatched with raw SkillPixel and query inputs and
without mounting the old self-generated index. It rebuilds the visual index
from raw videos. At the last bounded poll it was still
`KernelWorkerStatus.RUNNING`, and `kaggle kernels output` had not published
files. A Jina-enabled v12 package was prepared, but dispatch was rejected with
`Maximum batch GPU session count of 2 reached`. This is a runtime-slot blocker,
not an authentication failure. No Kaggle result is claimed until an output
manifest is actually downloaded and validated.

## Reproduce locally

Use a fresh environment and keep all paths explicit:

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC-SkillPixel-KIS
$env:UV_PROJECT_ENVIRONMENT = ".venv-kis"
$env:SKILLPIXEL_RAW_INPUT = "D:\Code\Code\AIO\Code\HCMAIC\learn\skillpixel\Buoi_08_Mock_contest_KIS\DataMockTest\videos"
$env:SKILLPIXEL_QUESTIONS = "D:\Code\Code\AIO\Code\HCMAIC\learn\skillpixel\Buoi_08_Mock_contest_KIS\DataMockTest\questions.csv"
$env:SKILLPIXEL_CORPUS = "D:\Code\Code\AIO\Code\HCMAIC\learn\skillpixel\Buoi_08_Mock_contest_KIS\DataMockTest\corpus.csv"
$env:SKILLPIXEL_RUN_ROOT = "D:\Kaggle\skillpixel-kis-local-infer-v8b"
$env:SKILLPIXEL_MODEL_PATH = "D:\Models\hf\siglip2-base-patch16-224"
$env:SKILLPIXEL_PROVIDER = "siglip2"
$env:SKILLPIXEL_LOCAL_FILES_ONLY = "true"

uv run python scripts\skillpixel_kis_build.py --config configs\skillpixel_kis.yaml --stage catalog
uv run python scripts\skillpixel_kis_build.py --config configs\skillpixel_kis.yaml --stage visual --model-id V1
uv run python scripts\skillpixel_kis_infer.py --config configs\skillpixel_kis.yaml --run-dir $env:SKILLPIXEL_RUN_ROOT --top-k 100
uv run python scripts\skillpixel_kis_validate.py --config configs\skillpixel_kis.yaml --run-dir $env:SKILLPIXEL_RUN_ROOT
```

Optional OCR/object/ASR stages require explicit cached model paths or the
explicit `--allow-model-download` flag. Jina uses
`scripts\skillpixel_kis_jina_benchmark.py` with `--allow-model-download` and
never falls back to another provider.

## Verification commands

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-kis"
uv run pytest
uv run ruff check src tests scripts
uv run mypy src
```

Observed result: `272 passed, 1 warning in 22.09s`; Ruff passed; Mypy passed on
68 source files. The warning is the existing Starlette/httpx deprecation
warning from the test environment.

## Commit history and rollback

The public repository preserves the copied SkillPixel history and adds focused
implementation/test commits. The latest relevant commits are:

```text
5f08c0b test(skillpixel): validate full artifact round-trip
811264b feat(kaggle): add explicit Jina benchmark job
fcdde19 fix(kis): exclude mutable manifests from checksums
0a9da31 fix(kis): bound hybrid artifact checksums
7e3eacf feat(text): add Jina text embedding provider
586b272 test(benchmark): cover Jina candidate runner
471c137 feat(rerank): add real reranker path
198ecc8 feat(kis): rebuild visual index from raw input
3217bde feat(kis): run full SkillPixel KIS inference
ff6183c feat(fusion): enable validated multi-channel RRF
e22ed22 feat(asr): add timestamped transcript artifacts
4765f47 feat(object): add real object detection artifacts
29b4552 feat(bm25): build OCR lexical index
75a47f4 feat(ocr): add real OCR weights and artifacts
b2e282c test(channels): add real provider artifact contracts
027f805 chore(repo): publish current SkillPixel KIS baseline
```

The working tree is clean and all listed commits are pushed to `origin/main`.
Rollback should be performed by a new, reviewable `git revert` of the selected
feature commits; do not reset or delete the repository history. The visual-only
baseline is the state before the hybrid/channel additions, represented by the
history immediately before `027f805` plus the preserved upstream commits.

## Known limitations

- No official HCMAIC qrels were present, so no quality promotion decision is
  possible.
- Kaggle v11 output and the Jina-enabled v12 output are not yet available for
  evidence inspection because the job remains running and the account reached
  the two-session GPU limit.
- Local machine has no cached PaddleOCR, Ultralytics, faster-whisper, or Jina
  weights. These providers fail closed rather than silently switching models.
- ASR, TRAKE, and Q&A are intentionally outside this implementation scope.
- Generated data/index/weights are kept outside Git. Re-run the commands above
  to regenerate them from raw input.

## Post-pause channel update (2026-08-07)

This section records the first externally executed optional channel after the
pause. It supersedes the older Kaggle-pending notes above for OCR only; no
generated output is committed to this repository.

### OCR GPU artifact: PASS at artifact and mapping level

- Kernel: `khanhss/skillpixel-kis-ocr-gpu-v1`, version 8.
- Job URL: https://www.kaggle.com/code/khanhss/skillpixel-kis-ocr-gpu-v1
- Raw source: `trieu241007/kis-skillpixel`, rebuilt independently from raw
  videos; BTC keyframes/mappings/features and teammate artifacts were not
  mounted.
- Requested device: CUDA on `NvidiaTeslaT4`; actual Paddle device: CUDA with
  device count 2 and device name `Tesla T4`.
- Provider actually executed: `paddleocr` 2.10.0, actual model/revision
  `PaddleOCR-legacy`. `PP-OCRv6` was the requested label, but the manifest is
  deliberately reported with the legacy runtime identity; no silent model
  substitution occurred.
- Input: 250 raw videos, 9,835 sampled frames, stride 10.
- OCR output: 7,700 non-empty OCR records. The remaining 2,135 scanned frames
  had no OCR record under the channel schema; they were not skipped as input.
- Dataset hash: `6f4fffefb26f09593abc15c4eb9ca2e77dde564a476fa6b44637da97df284b1e`.
- Raw catalog SHA256:
  `266780726a041a5f5cea40f91d38509b1b583de3f239ad46393bb3d9f7614bc9`.
- Mapping validation: 7,700/7,700 OCR records round-tripped through the raw
  catalog with zero errors for `frame_uid`, `video_id`, `video_filename`,
  `source_frame_idx`, and `timestamp_ms`.
- Quality gate: `UNVALIDATED_ON_HCMAIC` because no official HCMAIC qrels were
  available. This is an artifact-validity result, not a quality or SOTA claim.

Downloaded verification artifacts are kept outside Git:

- `D:\Kaggle\skillpixel-kis-ocr-gpu-v8-output\skillpixel-kis-run-v1\kaggle_ocr_job_manifest.json`
- `D:\Kaggle\skillpixel-kis-ocr-gpu-v8-output\skillpixel-kis-run-v1\raw\dataset_manifest.json`
- `D:\Kaggle\skillpixel-kis-ocr-gpu-v8-output\skillpixel-kis-run-v1\raw\catalog.jsonl`
- `D:\Kaggle\skillpixel-kis-ocr-gpu-v8-output\skillpixel-kis-run-v1\channels\ocr\V1\ocr.jsonl`
- `D:\Kaggle\skillpixel-kis-ocr-gpu-v8-output\skillpixel-kis-run-v1\channels\ocr\V1\ocr_manifest.json`
- `D:\Kaggle\skillpixel-kis-ocr-gpu-v8-output\skillpixel-kis-run-v1\channels\ocr\V1\channel_stage_manifest.json`

### Jina status at this checkpoint

The first Jina dispatch failed before repository startup because Kaggle
mounted the source package under a path different from the hard-coded path.
Read-only dataset inspection confirmed that the source dataset exists. The
runner was corrected to resolve the mounted package by its `pyproject.toml`
and `src/hcmaic` contract, then one controlled version 2 was dispatched with
the exact `NvidiaTeslaT4` accelerator. It is still `RUNNING` at the time of
this update; no Jina model, index, score, or fusion result is claimed yet.

The Jina runner is strict: it requests `jinaai/jina-clip-v2`, uses a separate
`visual/jina-clip-v2` index, and sets `allow_fallback=false`. A model/cache or
quota failure will be recorded as unavailable rather than silently converted
to CLIP or SigLIP2.
