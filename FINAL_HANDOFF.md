# HCMAIC v0 final handoff

## What is delivered

This branch is a tested end-to-end retrieval foundation derived from the pinned
SoftSignals upstream repository. It contains versioned data contracts, BTC
mapping validation, deterministic fixture generation, mock and optional CLIP
providers, exact NumPy and optional FAISS indexes, retrieval/API/UI plumbing,
evaluation, tests, and reproducible verification scripts.

The runnable boundary is `src/hcmaic/`. `upstream_reference/` is preserved for
provenance and is not the production import path. `Learn/` is outside the
write boundary.

## Quick start

```powershell
uv sync --locked --extra faiss
uv run python scripts/make_fixture.py
uv run hcmaic validate-data --input data/sample
uv run hcmaic build-index --input data/sample --output artifacts/sample
uv run hcmaic serve --index artifacts/sample --port 8017
```

In another terminal:

```powershell
uv run hcmaic search --index artifacts/sample --query "a solid red keyframe"
uv run hcmaic evaluate --index artifacts/sample `
  --queries data/sample/queries.jsonl --qrels data/sample/qrels.jsonl
```

Run the complete local gate with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv run pytest -q
```

## Team tickets

These are bounded next steps for the five-person team. Each ticket should add
tests and preserve the existing contracts and fail-closed artifact checks.

1. **TV1 — contracts and submission adapter**: map the current canonical
   submission JSON to the BTC schema, validate IDs/timestamps, and create
   a dry-run exporter. Do not upload or use credentials.
2. **TV2 — ingestion and timestamps**: adapt real BTC keyframe metadata and
   video/frame paths, preserve `pts_time`, and add malformed/VFR fixtures.
3. **TV3 — embeddings and index benchmark**: wire the approved CLIP checkpoint,
   record model/provider/dimension/device in manifests, compare exact/FAISS,
   and measure CPU/GPU latency. CUDA is unverified in this handoff.
4. **TV4 — query, rerank, and KISC**: implement query expansion and optional
   reranking behind a feature flag; evaluate paired ablations rather than
   changing the deterministic baseline silently.
5. **TV5 — UI, evaluator, and QA**: connect real evaluation files, add
   browser regression tests, harden accessibility/error states, and maintain
   the release checklist.

## Guardrails

Do not edit `Learn/`, commit model weights, expose local filesystem paths,
claim fixture scores as BTC scores, or upload a submission without an explicit
team-controlled browser handoff. Keep generated indexes and reports out of
Git; the package is rebuilt from the pinned source state.
