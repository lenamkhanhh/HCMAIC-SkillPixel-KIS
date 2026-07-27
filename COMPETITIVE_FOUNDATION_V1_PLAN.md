# Competitive Foundation v1 Plan

Date: 2026-07-27

This plan is based on the current `system/` tree only.
It reflects what is actually implemented, not the mission target.

## Current architecture inventory

- CLI entrypoint: `src/hcmaic/cli/main.py`
- Ingestion:
  - raw video ingestion: `src/hcmaic/ingestion/video.py`
  - mapping/catalog/manifest: `src/hcmaic/ingestion/mapping.py`, `catalog.py`, `manifest.py`
  - dataset validator: `src/hcmaic/ingestion/validator.py`
- Embeddings:
  - provider interface: `src/hcmaic/embedding/base.py`
  - deterministic fixture provider: `src/hcmaic/embedding/mock.py`
  - optional CLIP provider: `src/hcmaic/embedding/clip_real.py`
- Indexing:
  - exact NumPy index: `src/hcmaic/indexing/numpy_index.py`
  - optional FAISS index: `src/hcmaic/indexing/faiss_index.py`
  - artifact build/load: `src/hcmaic/indexing/artifacts.py`
- Retrieval and API:
  - retrieval service: `src/hcmaic/retrieval/service.py`
  - FastAPI app: `src/hcmaic/api/app.py`
- Evaluation:
  - offline evaluator: `src/hcmaic/evaluation/evaluator.py`
- Contracts:
  - frame/search/submission/validation models: `src/hcmaic/contracts/models.py`

## Verified baseline

- Offline fixture suite already existed and was passing before this mission.
- Current audit confirmed the core MVP path still exists:
  raw video -> keyframes -> mapping -> validation -> embeddings -> index -> retrieval -> API/UI -> evaluator.

## Gap analysis by mission gate

### Gate 1: Data and timestamp correctness

Status: PARTIAL

Implemented:

- raw video ingestion writes timestamps, frame indices, and media metadata
- validator rejects negative and out-of-range timestamps

Missing:

- explicit typed pipeline config for dataset/ingestion provenance
- stronger timestamp provenance fields in the manifest
- shot-aware timing contract

Insertion points:

- `src/hcmaic/config.py`
- `src/hcmaic/ingestion/video.py`
- `src/hcmaic/indexing/artifacts.py`
- `tests/test_video_ingest.py`
- `tests/test_artifacts.py`

### Gate 2: Shot/keyframe pipeline

Status: NOT_STARTED

Missing:

- shot detector abstraction
- shot metadata in `FrameRecord`
- fallback contract for uniform sampling vs shot segmentation

Insertion points:

- `src/hcmaic/config.py`
- `src/hcmaic/ingestion/video.py`
- `src/hcmaic/contracts/models.py`
- `tests/test_video_ingest.py`

### Gate 3: Embedding-provider readiness

Status: PARTIAL

Implemented:

- lazy registry for `mock` and `clip`
- optional CLIP provider loads on demand
- deterministic mock provider exists for tests

Missing:

- provider doctor / capability check
- registry for future multilingual providers
- explicit provider config and version provenance across the pipeline

Insertion points:

- `src/hcmaic/embedding/base.py`
- `src/hcmaic/embedding/clip_real.py`
- `src/hcmaic/config.py`
- `tests/test_mock_embedding.py`

### Gate 4: Multimodal feature and index contract

Status: NOT_STARTED

Missing:

- modality extractor interfaces
- multimodal artifact schema
- per-modality storage and index metadata

Insertion points:

- `src/hcmaic/contracts/models.py`
- `src/hcmaic/indexing/artifacts.py`
- `src/hcmaic/retrieval/service.py`

### Gate 5: Fusion, temporal retrieval, reranker, feedback contracts

Status: NOT_STARTED

Missing:

- fusion config and score composition contract
- late-fusion / RRF interface
- temporal expansion and reranker abstraction
- feedback record contract

Insertion points:

- `src/hcmaic/config.py`
- `src/hcmaic/contracts/models.py`
- `src/hcmaic/retrieval/service.py`
- `src/hcmaic/evaluation/evaluator.py`

### Gate 6: Reproducible benchmark harness

Status: PARTIAL

Implemented:

- offline evaluator returns Recall@1/5/10, MRR, and latency
- artifact manifests already capture dataset hash and code version

Missing:

- frozen benchmark input config
- per-query evidence export with config hash
- explicit benchmark runner command and report bundle

Insertion points:

- `src/hcmaic/config.py`
- `src/hcmaic/evaluation/evaluator.py`
- `src/hcmaic/cli/main.py`
- `tests/test_evaluator.py`

### Gate 7: Scale and mock-contest readiness

Status: PARTIAL

Implemented:

- exact NumPy fallback index
- optional FAISS provider
- deterministic fixture-safe pipeline

Missing:

- scalable index contract
- benchmarked large-corpus path
- mock-contest mode / frozen input bundle

Insertion points:

- `src/hcmaic/indexing/base.py`
- `src/hcmaic/indexing/artifacts.py`
- `src/hcmaic/retrieval/service.py`
- `src/hcmaic/cli/main.py`

## Target architecture

1. Typed foundation config becomes the single source of truth for:
   - dataset adapter
   - ingestion backend
   - shot detector
   - sampling policy
   - embedding providers
   - index providers
   - fusion / reranker
   - benchmark inputs
   - device / batch / seed
2. Every artifact records config hash, code version, and provider revision.
3. Ingestion emits timestamp provenance and safe replacement behavior.
4. Retrieval accepts multimodal score bundles and late-fusion candidates.
5. Evaluation exports frozen per-query evidence with a reproducible config hash.
6. Docs capture handoff, next session entrypoint, and evidence level.

## Phase dependency graph

```text
typed config + provenance
  -> ingestion + timestamp contract
  -> provider registry and doctor
  -> multimodal contracts
  -> fusion / reranker / temporal retrieval
  -> benchmark harness
  -> docs + handoff
```

## Compatibility and migration strategy

- Preserve current fixture behavior.
- Keep `mock` as the default offline provider.
- Keep `exact-numpy` as the mandatory fallback index.
- Add new contracts in a backward-compatible way, with defaults that match the
  current single-modality pipeline.
- Do not block ordinary tests on optional extras or large model downloads.

## Test and benchmark strategy

- Add one failing test per new contract before implementation.
- Keep small deterministic fixtures for red/green coverage.
- Use existing offline suite for regression protection.
- Treat optional CLIP/FAISS evidence as interface-level unless run on a machine
  that actually has the extra installed.

## Risk register

- Risk: overfitting the foundation to the fixture.
  - Mitigation: keep dataset-specific behavior behind adapters/config.
- Risk: optional backends silently become required.
  - Mitigation: lazy import and explicit doctor checks.
- Risk: timestamp/force behavior regresses the stable fixture path.
  - Mitigation: regression tests around the existing video suite.
- Risk: adding too much architecture before the team has BTC data.
  - Mitigation: ship contracts first, real implementations behind defaults.

## Rollback / checkpoint strategy

- Keep each phase separately testable.
- Preserve the current offline baseline.
- If a phase is incomplete, document the partial status in `NEXT_SESSION.md`
  and leave the repo in a runnable state.

## Acceptance criteria

- The seven gates are individually statused.
- Artifact manifests carry typed config and provenance.
- Optional providers remain lazy.
- Existing offline suite still passes.
- Handoff docs explain exactly what remains and where to continue.

## Deliberately deferred work

- Real BTC dataset adapter.
- Real multilingual provider downloads.
- Real shot segmentation model.
- Production deployment and external monitoring.

## Expected evidence level

- `mock` embedding provider: `FIXTURE_VERIFIED`
- `clip` provider: `INTERFACE_ONLY` until run on a machine with `uv sync --extra clip`
- `exact-numpy` index: `FIXTURE_VERIFIED`
- `faiss` index: `INTERFACE_ONLY` until run on a machine with `uv sync --extra faiss`
- BTC dataset adapter: `NOT_APPLICABLE` until official data is available
