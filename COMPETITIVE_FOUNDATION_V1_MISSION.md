# HCMAIC 2026 — Competitive Foundation v1 Mission

> **Instruction for Claude/coding agents:** Read this file completely before
> changing code. First inspect the real repository and write the implementation
> plan. Then immediately self-review that plan and implement it phase by phase.
> Do not stop after producing only another plan.

## 1. Repository and verified starting point

- Workspace: `D:\Code\Code\AIO\Code\HCMAIC\system`
- Current branch at mission creation: `hcmaic-2026-foundation`
- Current HEAD at mission creation: `c0db285`
- Existing verified baseline:
  - 144 tests passed.
  - 92% total coverage.
  - Ruff passed.
  - Mypy passed.
  - Working tree was clean before this mission file was added.
- Existing pipeline:

```text
raw video
-> uniform sampling and deduplication
-> keyframe mapping
-> validation and catalog
-> mock or CLIP embeddings
-> exact NumPy or FAISS Flat index
-> retrieval service
-> FastAPI and operator UI
-> fixture evaluator
```

The Git remote named `origin` currently points to:

```text
https://github.com/hhlearntocode/SoftSignalsRetrievalSystems-AIC2025
```

This is the public upstream repository, not a confirmed team-owned repository.
Do **not** push, force-push, change the remote, merge into `main`, or create a PR
against that upstream as part of this mission.

## 2. Mission objective

Turn the existing software foundation into **Competitive Foundation v1**:

- ready for a future BTC dataset adapter;
- correct and evidence-backed for video timestamps;
- shot-aware while retaining a deterministic fallback;
- ready for multiple multilingual embedding providers;
- ready for visual, OCR, ASR, caption, metadata, and future video features;
- ready for per-modality retrieval, late fusion, temporal expansion, and a
  replaceable reranker;
- benchmarkable through frozen inputs, configs, hashes, metrics, and per-query
  evidence;
- usable from the interactive UI and a future automated-agent path;
- documented well enough for five team members and future coding-agent sessions
  to continue without the old chat history.

The official BTC 2026 dataset is not available locally. Therefore:

- do not optimize for an imagined dataset;
- do not claim BTC or competition quality;
- keep dataset-specific behavior behind adapters and configuration;
- use fixtures/proxy data only as plumbing and regression evidence;
- do not download private data, model weights, or large public datasets without
  explicit authorization;
- build model/provider interfaces now and distinguish interface tests from real
  model runtime evidence.

## 3. Evidence levels

Every feature/provider must be labeled with the strongest evidence actually
obtained:

1. `INTERFACE_ONLY` — contract/code exists but the real dependency was not run.
2. `FIXTURE_VERIFIED` — deterministic fixture or mocked dependency passed.
3. `REAL_RUNTIME_VERIFIED` — real dependency/model/backend was executed.
4. `PROXY_BENCHMARKED` — run on a frozen legal proxy dataset/query set.
5. `BTC_BENCHMARKED` — reserved for future official BTC evidence.

Never promote evidence to a higher level based only on source inspection,
documentation, a mock, or another machine's report.

## 4. Seven Competitive Foundation gates

Track every gate using exactly one status:

- `PASS`
- `PARTIAL`
- `BLOCKED`
- `NOT_APPLICABLE`
- `NOT_STARTED`

The seven gates are:

1. Data and timestamp correctness.
2. Shot/keyframe pipeline.
3. Embedding-provider readiness.
4. Multimodal feature and index contract.
5. Fusion, temporal retrieval, reranker, and feedback contracts.
6. Reproducible benchmark harness.
7. Scale and mock-contest readiness.

Do not say “Competitive Foundation v1 complete” unless every mandatory gate is
`PASS`, or `NOT_APPLICABLE` with specific evidence and justification.

## 5. Execution procedure

### Phase 0 — Audit and plan

Before implementation, read completely:

- `README.md`
- `GOAL.md`
- `DECISIONS.md`
- `PROGRESS.md`
- `FINAL_HANDOFF.md`
- `VERIFICATION_REPORT.md`
- `MILESTONE_1_IMPLEMENTATION.md`
- `OVERNIGHT_REPORT.md`
- relevant source and test files;
- `pyproject.toml`;
- `uv.lock`.

Inspect:

```powershell
git status --short --branch
git log --oneline --decorate -15
git remote -v
uv run pytest
uv run ruff check src tests scripts
uv run mypy src
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv pip check
```

Create `COMPETITIVE_FOUNDATION_V1_PLAN.md` containing:

- current architecture inventory;
- exact gap analysis for all seven gates;
- target architecture;
- exact files to add/change;
- phase dependency graph;
- compatibility and migration strategy;
- test and benchmark strategy;
- risk register;
- rollback/checkpoint strategy;
- acceptance criteria;
- deliberately deferred work;
- expected evidence level for every optional backend/model.

Self-review the plan before implementation:

- remove unnecessary rewrites;
- preserve existing contracts and fixture behavior where safe;
- confirm optional dependencies remain lazy;
- confirm no heavy model is needed by ordinary tests;
- ensure each phase can stop in a clean, testable state.

After the self-review, continue implementation without waiting for user
confirmation between safe phases.

### Phase 1 — Typed configuration and artifact provenance

Add project-native typed configuration for:

- dataset adapter;
- ingestion backend;
- shot detector;
- sampling policy;
- modality extractors;
- embedding providers;
- index providers;
- fusion method and weights;
- reranker;
- benchmark inputs;
- device, batch size, and seed.

Do not add a large configuration framework without evidence that it is needed.

Every generated runtime artifact must record:

- provider/model name and revision;
- preprocessing version;
- embedding dimension and normalization;
- dataset manifest hash;
- configuration hash;
- code commit SHA when available;
- index type and parameters;
- device;
- creation timestamp.

### Phase 2 — Correct video ingestion and shot contract

Fix the raw-video ingestion path.

Mandatory behavior:

1. Preserve timestamp evidence.
   - Do not reconstruct FFmpeg timestamps using `index * interval`.
   - Capture PTS/best-effort timestamp from the decoder.
   - Handle non-zero start times.
   - Record timestamp source: exact PTS, best-effort PTS, or CFR fallback.
   - Do not silently clamp incorrect timestamps merely to pass validation.

2. Make `--force` safe.
   - Generate into a staging location.
   - Validate the new output.
   - Replace the previous valid data only after successful generation.
   - A failed replacement must leave the old dataset usable.

3. Normalize recoverable per-video errors.
   - FFmpeg/ffprobe failure or invalid JSON.
   - PIL image decode error.
   - OpenCV decode/probe error.
   - Expected filesystem errors.
   - A batch should continue where safe and write a structured failure report.

4. Add a shot-detector interface.
   - Deterministic no-shot/uniform fallback.
   - Optional PySceneDetect provider.
   - Reserved TransNetV2 provider slot without automatic weight download.

5. Add a within-shot sampler.
   - At least one representative frame per shot.
   - Configurable extra frames for long shots.
   - Deterministic maximum frame count.
   - Post-sampling deduplication.

6. Extend the mapping contract while preserving old sample compatibility:

```text
video_id
shot_id
frame_id
frame_idx
pts_time
shot_start
shot_end
width
height
timestamp_source
ingestion_provider
sampling_policy
```

Required regression coverage:

- CFR fixture;
- non-zero start/timestamp parsing;
- invalid or out-of-range timestamp;
- short and long shots;
- duplicate frames;
- corrupt video;
- failed `--force` preserves old output;
- deterministic rerun;
- path traversal;
- mocked FFmpeg output parsing.

If FFmpeg is unavailable on the machine, mark the actual FFmpeg/VFR runtime
evidence `BLOCKED` or `INTERFACE_ONLY`; do not call it verified.

### Phase 3 — Embedding-provider registry

Keep:

- deterministic mock provider;
- CLIP ViT-B/32 as a control baseline.

Implement a provider registry selected through configuration rather than edits
to retrieval code.

Add lazy adapters/contracts for:

- SigLIP 2;
- Jina CLIP v2.

Provider requirements:

- lazy optional imports;
- no import-time downloads;
- configurable provider revision;
- configurable device and batch size;
- provider-owned image/text preprocessing;
- discovered and validated output dimension;
- recorded normalization contract;
- compatible image and text encoders from the same provider/revision;
- actionable errors when a dependency or model is unavailable.

Tests may use controlled fakes/mocks. A mocked provider is not real-model
evidence.

Add a diagnostic command such as:

```text
hcmaic provider-doctor --provider <provider>
```

It should report:

- dependency availability;
- configured model/revision;
- selected device;
- dimension when known;
- strongest evidence level;
- installation/run instructions.

Reserve an interface for future shot/segment-level video encoders. Do not
download or implement a heavyweight video model in this mission.

### Phase 4 — Multimodal feature contract

Create a generic feature record for:

- visual embeddings;
- OCR text;
- ASR text;
- captions;
- metadata/entities;
- future segment/video embeddings.

Every record must identify:

- `video_id`;
- frame, shot, or segment identity;
- start/end timestamp;
- modality;
- provider and revision;
- raw text or artifact reference where appropriate;
- confidence when available;
- artifact/content hash.

Add modality extractor/provider interfaces.

Minimum fixture support:

- deterministic mock OCR;
- deterministic mock ASR;
- deterministic mock caption provider.

Document slots for real providers such as multilingual OCR and Whisper ASR, but
do not automatically download/run large models.

Support:

- extracting one modality;
- validating its artifacts;
- building/rebuilding one modality index;
- disabling unavailable modalities without breaking visual retrieval.

### Phase 5 — Multi-index retrieval and fusion

Preserve the current simple visual search path while introducing a retrieval
orchestrator.

Support channels:

- visual;
- OCR;
- ASR;
- caption/metadata;
- future segment/video.

The canonical candidate/result must contain:

- frame/shot/segment identity;
- timestamp;
- raw per-modality scores;
- rank/normalized scores;
- evidence text where applicable;
- contributing providers;
- final fused score;
- explanation of modality contributions.

Implement deterministic baselines:

- Reciprocal Rank Fusion;
- configurable weighted late fusion.

Do not implement learned/adaptive fusion yet.

Add temporal expansion:

- neighboring keyframes;
- same-shot frames;
- previous/next shots;
- configurable temporal window;
- stable deduplication and ordering.

Add a reranker protocol:

```text
top-N candidates -> reranker -> top-K candidates
```

The default must be a no-op/passthrough reranker with timeout/fallback
contracts. Do not add a heavyweight VLM reranker yet.

Add a feedback contract:

- session ID;
- query revision;
- positive frame/shot;
- negative frame/shot;
- prior result set.

A simple deterministic feedback path is acceptable. Learned feedback and the
full conversational KISC agent are later competitive work.

Keep or document migration for existing API consumers.

### Phase 6 — Index strategy and scale contract

Keep:

- Exact NumPy;
- FAISS FlatIP as the exact correctness oracle.

Add a configurable scalable-index contract. If safe and supported, provide one
ANN baseline such as FAISS HNSW or IVF.

Requirements:

- exact-versus-ANN Recall@K measurement;
- deterministic ID mapping;
- recorded ANN parameters;
- no silent metric/normalization changes;
- index manifest validation.

Create a configurable synthetic scale benchmark that reports:

- vector count and dimension;
- build time;
- index size;
- memory where measurable;
- p50/p95 latency;
- throughput;
- ANN Recall@K against exact.

Do not claim a scale was tested unless it was actually executed on this
machine.

Internal engineering targets, when feasible:

- ANN Recall@100 at least `0.98` against exact;
- basic retrieval p95 below `500 ms` on explicitly recorded hardware.

These are not BTC official thresholds.

### Phase 7 — Reproducible benchmark harness

Add a command similar to:

```powershell
uv run hcmaic benchmark --config configs/competitive_v1.yaml
```

The harness must freeze and record:

- dataset manifest;
- query set;
- qrels;
- configuration;
- seed;
- providers and revisions;
- index/fusion/reranker settings;
- warmup and repeated-run policy;
- code/config/data hashes.

Report:

- Recall@1/5/10/100;
- MRR;
- timestamp error when ground truth permits;
- visual/OCR/ASR/action-temporal/Vietnamese/mixed-modality slices;
- p50/p95 latency;
- indexing time;
- resource usage where measurable;
- invalid/missing results;
- ANN recall compared with exact.

Expected outputs:

```text
benchmark_summary.json
per_query_results.jsonl
run_manifest.json
failure_cases.md
```

Create a legal deterministic proxy benchmark/fixture generator. Proxy scores
must always be labeled as proxy/plumbing evidence, never BTC scores.

Add an experiment ledger containing:

```text
experiment id
hypothesis
change
dataset hash
config hash
metric movement
slice movement
latency movement
decision
regression added
next experiment
```

### Phase 8 — Minimum UI/API readiness

Do not redesign the entire UI.

Add only the foundation surfaces needed by the team:

- modality score/evidence breakdown;
- exact timestamp display/open behavior;
- same-shot and neighboring-shot exploration;
- positive/negative feedback recording;
- provider/index/fusion version visibility;
- safe submission preview;
- structured response usable by a future automated agent.

Preserve existing behavior and add API/E2E regression tests for new fields and
fallbacks.

## 6. Documentation and human/agent handoff

Create:

```text
docs/competitive_v1/
├── 00_OVERVIEW.md
├── 01_ARCHITECTURE.md
├── 02_DATA_AND_ARTIFACT_CONTRACT.md
├── 03_SETUP_AND_RUNBOOK.md
├── 04_BENCHMARK_PROTOCOL.md
├── 05_TEAM_HANDOFF.md
├── 06_WHEN_BTC_DATASET_ARRIVES.md
├── 07_KNOWN_GAPS_AND_RISKS.md
└── NEXT_SESSION.md
```

Also create/update:

- `COMPETITIVE_FOUNDATION_V1.md`
- `COMPETITIVE_FOUNDATION_V1_PLAN.md`
- `TEAM_TASK_BOARD.md`
- `README.md` links to all entry points.

### `00_OVERVIEW.md`

Explain from first principles:

- the problem;
- offline and online pipelines;
- software-ready vs benchmark-ready vs competition-ready;
- seven-gate status;
- what the project does not yet prove.

### `01_ARCHITECTURE.md`

Include Mermaid diagrams for:

- offline ingestion/indexing;
- online multimodal retrieval;
- interactive flow;
- future automated-agent flow;
- provider and adapter boundaries;
- module dependencies and ownership.

### `02_DATA_AND_ARTIFACT_CONTRACT.md`

Document:

- all schemas and fields;
- timestamp semantics;
- examples;
- modality records;
- manifest/version/hash rules;
- compatibility with the existing sample/BTC-style layout;
- validation and fail-closed behavior.

### `03_SETUP_AND_RUNBOOK.md`

Provide copyable Windows PowerShell instructions for:

- fresh environment setup;
- optional dependency groups;
- fixture generation;
- video ingestion;
- modality extraction;
- validation;
- index building;
- CLI search;
- API/UI;
- benchmark;
- troubleshooting;
- CPU/GPU expectations;
- fresh-clone verification.

### `04_BENCHMARK_PROTOCOL.md`

Define:

- frozen inputs;
- query slices;
- metrics;
- repeated-run and warmup rules;
- exact-vs-ANN comparison;
- ablation rules;
- acceptable evidence;
- leakage prevention;
- why proxy metrics are not BTC metrics.

### `05_TEAM_HANDOFF.md`

Explain:

- current state and gate table;
- complete/partial/blocked modules;
- exact commands;
- how to add a dataset adapter;
- how to add a provider;
- how to add a modality;
- how to add an index;
- how to add a reranker;
- PR/review expectations;
- module owners/roles without inventing member names.

Use plain Vietnamese or clear bilingual terminology and expand acronyms on first
use.

### `06_WHEN_BTC_DATASET_ARRIVES.md`

Include:

- first 6-hour dataset audit;
- 24-hour vertical baseline;
- 48-hour model/modality benchmark;
- 72-hour baseline freeze and team assignment;
- decision paths for:
  - raw video vs supplied keyframes;
  - audio vs no audio;
  - supplied CLIP/object/caption features;
  - small vs large vector count;
  - official submission schema.

### `07_KNOWN_GAPS_AND_RISKS.md`

List:

- real backends/models not executed;
- missing BTC data/rules;
- FFmpeg/VFR evidence;
- hardware limits;
- proxy limitations;
- optional dependencies;
- technical debt;
- status, evidence, and next action for every unresolved item.

### `NEXT_SESSION.md`

This is mandatory and is written for the next Claude/coding-agent session.

It must contain:

- repository path;
- branch and HEAD;
- git status;
- last completed phase;
- seven-gate status table;
- exact files changed;
- exact commands run;
- latest tests and evidence levels;
- blockers;
- next single action;
- exact resume commands;
- unverified claims.

Every future coding-agent session must read `NEXT_SESSION.md` before changing
code.

### `TEAM_TASK_BOARD.md`

Do not assign member names. Use:

```text
Task ID
Priority
Gate
Module
Description
Dependencies
Definition of Done
Suggested owner role
Status
Evidence
```

Separate:

- foundation blockers;
- tasks triggered by the BTC dataset;
- later competitive tricks.

Do not mix future LLM/KISC tricks into the foundation blockers.

## 7. Deferred work

Unless needed for an interface/test, do not implement:

- full conversational KISC agent;
- learned/adaptive fusion;
- heavy VLM reranking;
- heavy video foundation models;
- domain fine-tuning;
- private/BTC data ingestion;
- official submission upload;
- production cloud deployment;
- model-weight packaging;
- advanced UI redesign.

These are post-foundation or post-dataset tasks.

## 8. Verification and completion

Run narrow tests after each phase.

Before reporting completion run:

```powershell
uv run pytest
uv run pytest --cov=src/hcmaic
uv run ruff check src tests scripts
uv run mypy src
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv pip check
git status --short
git diff --check
```

Requirements:

- existing tests remain passing;
- new critical behavior has regression tests;
- coverage does not materially regress without explanation;
- optional real-model/backend paths are labeled honestly;
- no `.venv`, cache, private data, raw video, model weights, generated indexes,
  Git bundles, secrets, or credentials are committed;
- no local absolute paths leak through API responses;
- fixture/proxy scores are labeled non-competitive.

The existing environment previously showed duplicate `charset-normalizer`
metadata. Re-check it. Do not claim the environment is clean if `uv pip check`
still fails. Avoid destructive environment deletion during this mission; provide
an exact clean-rebuild procedure if needed.

## 9. Local Git checkpoints

Preserve the existing `v0-handoff` checkpoint.

Use small local commits only when their relevant tests pass. Suggested commits:

1. `docs: define competitive foundation v1 plan and gates`
2. `fix: make video timestamps and replacement safe`
3. `feat: add configurable embedding and multimodal contracts`
4. `feat: add fusion temporal and reranker interfaces`
5. `feat: add reproducible competitive benchmark harness`
6. `docs: add team runbook and continuation handoff`

Do not push because `origin` is the upstream repository.

## 10. Final response contract

The final report must include:

- phase-by-phase implementation status;
- seven-gate status table;
- local commit hashes;
- files added/changed;
- exact test/verification commands and outcomes;
- actual model/backend/scale evidence;
- fixture-only evidence;
- blockers and risks;
- exact next team action;
- exact path to `docs/competitive_v1/NEXT_SESSION.md`.

Do not report “Competitive Foundation v1 complete” merely because interfaces or
documentation exist. Completion requires the gate evidence defined above.

