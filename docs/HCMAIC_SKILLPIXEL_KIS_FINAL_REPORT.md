# HCMAIC SkillPixel KIS — implementation and verification report

Date: 2026-08-07
Repository: [HCMAIC-SkillPixel-KIS](https://github.com/lenamkhanhh/HCMAIC-SkillPixel-KIS)  
Branch: `main`  
Quality status: `UNVALIDATED_ON_HCMAIC`

## Executive result

The raw-video-first SkillPixel TKIS/VKIS path is implemented and has now been
run locally end to end with the canonical raw-video package and all available
real retrieval channels. The authoritative run used 250 raw videos, 9,835
sampled frames, 100 queries (50 TKIS and 50 VKIS), real local SigLIP2, normalized
float32 exact FAISS `IndexFlatIP`, OCR/BM25, object retrieval, timestamped ASR,
RRF, source-frame deduplication/diversity, and a cached real
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker. The output contains 100
answers for every query and passes the independent final validator.

No official HCMAIC qrels were available. Therefore this report does not claim
Recall, MRR, SOTA, top-1 quality, or contest score. The only quality statement
that is justified is `UNVALIDATED_ON_HCMAIC`.

## Authoritative final canonical hybrid run (2026-08-07)

Artifact root (kept outside Git):

`D:\Kaggle\skillpixel-kis-final-v1`

This run is the source of truth for the current report. Its `raw` directory is
a junction to the checksum-verified canonical raw package
`D:\Kaggle\skillpixel-kis-canonical-raw-v2`; no BTC keyframe/mapping/feature
artifact or teammate data was used. All four fused channels carry the same
dataset manifest hash
`6f4fffefb26f09593abc15c4eb9ca2e77dde564a476fa6b44637da97df284b1e` and raw
catalog SHA256
`266780726a041a5f5cea40f91d38509b1b583de3f239ad46393bb3d9f7614bc9`.

| Field | Verified result |
| --- | --- |
| Raw source | 250 videos, 9,835 stride-10 frames |
| Visual index | FAISS `IndexFlatIP`, 9,835 vectors, 768D, L2-normalized float32 |
| Visual provider | SigLIP2 `google/siglip2-base-patch16-224`, local revision `main`, CPU |
| OCR | PaddleOCR 2.10.0, actual runtime `PaddleOCR-legacy`, 7,700 records |
| Object | Ultralytics 8.4.115, `yolo11n.pt`, 15,949 detections |
| ASR | faster-whisper 1.2.1, `small`, 663 timestamped segments |
| Fusion | RRF, rank constant 60; visual/OCR/object/ASR weights all 1.0 |
| Reranker | sentence-transformers 5.7.0 CrossEncoder, cached local weights, CPU |
| Query mix | 50 TKIS + 50 VKIS, query order preserved |
| Query batch | 267,425 ms total; 2,674 ms mean/query |
| Peak reported RAM | 1,496 MB |
| Evidence | 10,000 top-100 rows and 2,000 top-20 rows |
| Submission | 100 query rows; 100 answers/query; validator PASS |
| Mapping | FAISS/catalog round-trip: 9,835 checked, 0 errors |

The final validation report is `validation_final.json` with `valid=true`,
`raw_video_source=true`, `btc_artifacts_used=false`, and
`quality_status=UNVALIDATED_ON_HCMAIC`. The submission is:

`D:\Kaggle\skillpixel-kis-final-v1\submission_V1.csv`

Stable output hashes:

- `submission_V1.csv`: `a5430ade4f7784d41f117b3321a14b580172b679998ffe6d978c03ad84a94b46`
- `retrieval_evidence_top20.jsonl`: `fd2f7a7623842c8d8a9142c59a3c7eca742d1cc0b5bca99aab6f212346d96839`
- `retrieval_evidence_top100.jsonl`: `fe79f6471d8e86bac7965e244ab111b8c2a3abd7b5072fe550f04ed0d294b751`
- `model_registry.json`: `f7488fdd15fb7a04dac562f9e2b94093679b856c08b699bdcfc66299e7ac1065`
- `checksums.sha256`: `9f587b1ccbf47ba5264faa7049c751e99f3a1d6f530e113673d969e21032410e`

The final evidence rows include `query_id`, `rank`, `video_id`,
`video_filename`, `frame_uid`, `source_frame_idx`, `timestamp_ms`, fused score,
rerank score, per-channel score/evidence and provider metadata. `submission.csv`
and `submission_V1.csv` are byte-identical.

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

## Historical visual-only local run

Artifact root (retained as an earlier visual-only baseline):

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
| Jina CLIP v2 | Native image/text adapter and strict separate-index runner | `JINA_UNAVAILABLE_KAGGLE_PROVIDER_ERROR`; no fallback and no mixed vector space |
| OCR/BM25 | Real PaddleOCR artifact plus BM25 retrieval channel | PASS: 7,700 records, same canonical dataset/catalog hashes |
| Object | Real Ultralytics artifact plus object retrieval channel | PASS: 15,949 detections, same canonical dataset/catalog hashes |
| ASR | Real faster-whisper artifact with timestamp-to-frame mapping | PASS: 663 segments, same canonical dataset/catalog hashes; runtime explicitly enabled for this run |
| Fusion | Visual plus OCR/object/ASR scores, RRF, source-frame dedup/diversity | PASS: all four channels executed for 100 queries |
| Reranker | Explicit real CrossEncoder path, local-cache/download policy | PASS: cached `ms-marco-MiniLM-L-6-v2`, sentence-transformers 5.7.0, CPU |

The historical local optional-stage probes failed closed when their model
weights were absent. The final run used independently verified Kaggle-produced
OCR/object/ASR artifacts copied outside Git; their manifests, checksums, and
mapping validators passed before the channels were enabled. No empty or mock
channel artifact was used.

## Kaggle execution

Kaggle authentication was already verified by the user. The allowed API smoke
test `kaggle datasets list -p 1` / `kaggle kernels list -p 1` passed. The old CLI
version warning is informational only.

Source-only package uploaded:

`khanhss/hcmaic-skillpixel-kis-source-20260806`

The package contains only source/config/runner files. It excludes raw videos,
weights, embeddings, FAISS indexes, secrets, and tokens. The current package was
built from public-repo commit `811264b` and includes the explicit Jina runner.

Historical kernel:

`khanhss/skillpixel-kis-gpu-end-to-end-v1`

The earlier v11/v12 jobs remain historical execution attempts. Their output is
not used by the authoritative final run: one had no published output at the
last bounded poll, and the Jina-enabled attempt was rejected by the two-session
GPU limit. The final OCR, object, and ASR artifacts used here were downloaded
from separate, completed GPU jobs and passed local checksum/mapping validation.
Jina remains unavailable and is not a blocker for the main submission.

## Reproduce the authoritative inference locally

Keep generated raw/channel/index artifacts outside Git and keep every path
explicit. The verified machine used `.venv-kis-final` because the older `uv`
environment had a broken NumPy namespace (`numpy.load` was unavailable). This
is an environment issue, not a model fallback.

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC-SkillPixel-KIS
$env:SKILLPIXEL_RAW_INPUT = "D:\Code\Code\AIO\Code\HCMAIC\learn\skillpixel\Buoi_08_Mock_contest_KIS\DataMockTest\videos"
$env:SKILLPIXEL_QUESTIONS = "D:\Code\Code\AIO\Code\HCMAIC\learn\skillpixel\Buoi_08_Mock_contest_KIS\DataMockTest\questions.csv"
$env:SKILLPIXEL_CORPUS = "D:\Code\Code\AIO\Code\HCMAIC\learn\skillpixel\Buoi_08_Mock_contest_KIS\DataMockTest\corpus.csv"
$env:SKILLPIXEL_RUN_ROOT = "D:\Kaggle\skillpixel-kis-final-v1"
$env:SKILLPIXEL_MODEL_ID = "V1"
$env:SKILLPIXEL_MODEL_PATH = "D:\Models\hf\siglip2-base-patch16-224"
$env:SKILLPIXEL_PROVIDER = "siglip2"
$env:SKILLPIXEL_LOCAL_FILES_ONLY = "true"
$env:SKILLPIXEL_ALLOW_MODEL_DOWNLOAD = "false"
$env:SKILLPIXEL_DEVICE = "cpu"
$env:SKILLPIXEL_ENABLE_ASR_RUNTIME = "true"
$env:SKILLPIXEL_RERANKER = "cross-encoder"
$env:SKILLPIXEL_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
$env:SKILLPIXEL_RERANKER_MODEL_PATH = "C:\Users\HP\.cache\huggingface\hub\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8"
$py = ".\.venv-kis-final\Scripts\python.exe"

& $py scripts\skillpixel_kis_infer.py --config configs\skillpixel_kis.yaml --run-dir $env:SKILLPIXEL_RUN_ROOT --top-k 100
& $py scripts\skillpixel_kis_validate.py --config configs\skillpixel_kis.yaml --run-dir $env:SKILLPIXEL_RUN_ROOT
```

To rebuild the visual index from the canonical raw root, run the catalog and
visual build stages first and point `SKILLPIXEL_RUN_ROOT` at the new run. OCR,
object, and ASR stages require their own explicit cached/GPU model packages;
their completed manifests must match the canonical dataset/catalog hashes before
copying them into `run\channels\{ocr,object,asr}\V1`. Jina uses
`scripts\skillpixel_kis_jina_benchmark.py` with a separate index and never
falls back to another provider.

## Verification commands

```powershell
uv run pytest
uv run ruff check src tests scripts
uv run mypy src

.\.venv-kis-final\Scripts\python.exe -m pytest
.\.venv-kis-final\Scripts\python.exe -m ruff check src tests scripts
.\.venv-kis-final\Scripts\python.exe -m mypy src
```

Observed fallback result: `.venv-kis-final` gave `280 passed, 1 warning in
27.79s`; Ruff passed; Mypy reported no issues in 68 source files. The warning
is the existing Starlette/httpx deprecation warning. The required `uv run
pytest` collection failed because its `.venv` has a broken NumPy/OpenCV install
(`numpy.core.multiarray` is unavailable), and `uv run mypy src` failed because
that same environment exposes NumPy without importable stubs. `uv run ruff`
passed. No source change or silent dependency replacement was made to hide the
environment blocker.

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
the exact `NvidiaTeslaT4` accelerator. Version 2 passed the T4 preflight and
rebuilt the 9,835-frame raw catalog, but ended `ERROR` when the Jina benchmark
subprocess returned exit status 2. No Jina model, index, score, or fusion
result was produced.

The failure is not a quota or GPU-allocation failure: the log records two
Tesla T4 devices and a successful raw catalog build. The wrapper captured the
child provider stderr and only exposed the outer `CalledProcessError`, so the
inner model/dependency exception is not available in the downloaded Kaggle
log. This is recorded as `JINA_UNAVAILABLE_KAGGLE_PROVIDER_ERROR`, with no
fallback to CLIP or SigLIP2 and no further retry.

### ASR GPU artifact: PASS at artifact and mapping level

- Kernel: `khanhss/skillpixel-kis-asr-gpu-v1`, version 1.
- Job URL: https://www.kaggle.com/code/khanhss/skillpixel-kis-asr-gpu-v1
- Raw source: `trieu241007/kis-skillpixel`; only raw videos and the public
  source package were mounted.
- Requested/actual runtime: `NvidiaTeslaT4`, CUDA, two Tesla T4 devices;
  `ffprobe` was available.
- Provider actually executed: `faster-whisper` 1.2.1, model `small`,
  `float16` compute on CUDA. No mock transcript and no CPU fallback were used.
- Input: 250 raw videos and 9,835 raw sampled frames under the same dataset
  hash and raw catalog SHA256 as the validated OCR artifact.
- Output: 663 timestamped transcript segments across 188 videos. Each record
  carries `start_ms`, `end_ms`, `frame_uid`, `video_id`, `video_filename`,
  `source_frame_idx`, and `timestamp_ms`.
- Mapping validation: 663/663 records round-tripped through the 9,835-row raw
  catalog with zero errors. The 663 segments map to 661 unique frames because
  multiple transcript segments can share the nearest sampled frame.
- Artifact checksums: all four job-manifest checksums matched downloaded files.
- Quality gate: `UNVALIDATED_ON_HCMAIC`; no qrels or promotion claim.

Downloaded ASR verification artifacts are kept outside Git:

- `D:\Kaggle\skillpixel-kis-asr-gpu-v1-output\skillpixel-kis-asr-run-v1\kaggle_asr_job_manifest.json`
- `D:\Kaggle\skillpixel-kis-asr-gpu-v1-output\skillpixel-kis-asr-run-v1\raw\dataset_manifest.json`
- `D:\Kaggle\skillpixel-kis-asr-gpu-v1-output\skillpixel-kis-asr-run-v1\raw\catalog.jsonl`
- `D:\Kaggle\skillpixel-kis-asr-gpu-v1-output\skillpixel-kis-asr-run-v1\channels\asr\V1\asr.jsonl`
- `D:\Kaggle\skillpixel-kis-asr-gpu-v1-output\skillpixel-kis-asr-run-v1\channels\asr\V1\asr_manifest.json`
- `D:\Kaggle\skillpixel-kis-asr-gpu-v1-output\skillpixel-kis-asr-run-v1\channels\asr\V1\channel_stage_manifest.json`

The Jina runner is strict: it requests `jinaai/jina-clip-v2`, uses a separate
`visual/jina-clip-v2` index, and sets `allow_fallback=false`. A model/cache or
quota failure will be recorded as unavailable rather than silently converted
to CLIP or SigLIP2.
