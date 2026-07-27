# Competitive Foundation v1 — next session handoff

Read this file completely before editing.

## Repository state

- Workspace: `D:\Code\Code\AIO\Code\HCMAIC\system`
- Branch: `hcmaic-2026-foundation`
- Starting checkpoint: `c0db285`
- Last code checkpoint before this documentation handoff: `7b2e69f`
- Remote `origin` is the public upstream, not a confirmed team remote.
- Nothing from this mission was pushed.

Resume with:

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC\system
git status --short --branch
git log --oneline -8
uv run pytest
```

Preserve `mock` and `exact-numpy` as mandatory offline fallbacks. Do not
download private data, model weights, or large public datasets without team
authorization. Never promote fixture or synthetic results to BTC evidence.

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| Audit/plan | Done | mission and reviewed implementation plan committed |
| Config/provenance | Done for software foundation | typed YAML, stable config hash, artifact/run provenance |
| Ingestion/timestamps | Partial | decoder timestamp parser and OpenCV fixture path tested; no local FFmpeg/VFR runtime |
| Shot/mapping | Partial | deterministic contracts/sampler tested; real detectors not run |
| Providers/multimodal | Partial | lazy registry, doctor, records/artifacts and mocks tested; real multilingual models not run |
| Fusion/retrieval | Done for contracts | RRF, weighted fusion, temporal expansion, reranker, feedback and orchestration tests |
| Scale/benchmark | Partial | exact/HNSW synthetic benchmark and frozen proxy harness; no BTC-scale run |
| API/UI/operator | Fixture verified | API regressions plus real browser search/detail/feedback/preview smoke |
| Documentation/handoff | Done | runbook, contracts, team board, BTC-arrival procedure and this file |

## Seven mandatory gates

Use only the mission statuses below.

| Gate | Status | Strongest evidence | Why it is not stronger |
|---|---|---|---|
| G1 Data and timestamp correctness | PARTIAL | fixture ingestion, FFmpeg parser, rollback regressions | FFmpeg absent; no real non-zero-start/VFR end-to-end run |
| G2 Shot/keyframe pipeline | PARTIAL | deterministic no-shot detector and within-shot sampling tests | PySceneDetect/TransNetV2 are interface-only |
| G3 Embedding-provider readiness | PARTIAL | mock fixture verified; lazy provider doctor | SigLIP2/Jina/real CLIP not executed in this mission |
| G4 Multimodal feature/index contract | PARTIAL | canonical OCR/ASR/caption records, hash validation and mocks | real extractors and per-modality indexes are not connected |
| G5 Fusion/temporal/reranker/feedback | PARTIAL | deterministic unit/orchestrator tests and feedback browser flow | production retrieval service remains visual-first |
| G6 Reproducible benchmark harness | PASS | frozen config plus data/query/qrels/config/code hashes and per-query reports | pass is harness correctness only, not retrieval quality |
| G7 Scale and mock-contest readiness | PARTIAL | 1k-vector HNSW synthetic run and browser operator smoke | no official dataset, competition scale, schema or contest rehearsal |

Do not call Competitive Foundation v1 complete while any mandatory gate is
`PARTIAL`.

## What was implemented

- staged video ingestion with validation, decoder-derived FFmpeg timestamps,
  explicit timestamp provenance, structured batch failures and tested rollback;
- typed configuration and provenance in index and benchmark manifests;
- extended legacy-compatible mapping and shot/sampling contracts;
- lazy provider registry and non-downloading capability doctor;
- canonical multimodal feature records and tamper-detecting artifacts;
- fusion, orchestration, temporal expansion, reranking and bounded local
  feedback contracts;
- optional FAISS HNSW implementation and exact-versus-ANN synthetic benchmark;
- reproducible proxy benchmark command and four-report output bundle;
- runtime/provider/fusion visibility, shot context, feedback controls and
  canonical submission preview in the API/UI.

The full changed-file inventory is available with:

```powershell
git diff --name-status c0db285..HEAD
```

## Verified evidence

Latest full source gates including this documentation pass:

```text
scripts/verify.ps1                    -> all gates passed
uv run pytest --cov=src/hcmaic        -> 184 passed, 90% total coverage
uv run ruff check src tests scripts   -> pass
uv run mypy src                       -> pass
node --check src/hcmaic/ui/static/app.js -> pass
git diff --check                      -> pass
secret-pattern scan                   -> no finding
```

The only environment-level failure is `uv pip check`, documented below; it is
unrelated duplicate installed metadata rather than a declared dependency
conflict.

Actual local synthetic ANN evidence:

```text
vectors=1000, dimension=64, queries=20, top_k=20
Recall@20=0.9925
p95=0.493 ms
index_bytes=400090
evidence=SYNTHETIC_SCALE_VERIFIED
```

Actual fixture/proxy benchmark evidence:

```text
queries=6
Recall@1/5/10/100=1.0
MRR=1.0
evidence=FIXTURE_VERIFIED / PROXY plumbing only
```

Actual browser evidence:

```text
GET /health                   200
GET /system/info              200
POST /search                  200
GET /frames/{id}              200
POST /feedback                200
POST /submit/preview          200
browser console               0 errors, 0 warnings
```

These numbers do not predict BTC performance.

## Known blockers and risks

- No official BTC dataset, rules, evaluation slice, submission schema or portal
  evidence exists locally.
- FFmpeg is unavailable locally; only its command/parser contract was tested.
- SigLIP2, Jina CLIP v2, real OCR/ASR/caption and CUDA were not executed.
- HNSW evidence is synthetic and too small to establish competition latency or
  quality.
- `--force` rollback is tested but is a multi-file best-effort transaction;
  simultaneous commit/rollback filesystem failure can still need recovery.
- Legacy mapping compatibility is intentionally permissive; the future BTC
  adapter must enforce the rich schema.
- `uv pip check` reports duplicate `charset-normalizer` metadata versions
  `3.4.7` and `3.4.9`; rebuild the virtual environment rather than silently
  mutating this shared environment.
- Feedback is bounded but session-local/in-memory. The app is a local contest
  operator tool, not an authenticated public service.

See `07_KNOWN_GAPS_AND_RISKS.md` for the complete register.

## Single next action

When the official BTC dataset arrives, do **BTC-01 only** first: perform the
read-only six-hour data audit in `06_WHEN_BTC_DATASET_ARRIVES.md`, record
hashes/schema/corruption/audio/language/scale facts, and freeze a legal
20–50-query validation slice. Do not change models or fusion before measuring
the unchanged incumbent on that slice.

If the dataset has not arrived, the highest-value independent engineering task
is F-02 in `TEAM_TASK_BOARD.md`: connect exactly one mock modality artifact to
an isolated per-modality index and prove visual retrieval still works when
that channel is disabled or fails.
