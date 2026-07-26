# Architectural decisions

Format: date — decision — reason — alternatives rejected.

## 2026-07-26 — D1: Move upstream runnables to `upstream_reference/`

The upstream `app.py`/`migrate_embeddings.py`/`backend/`/`src/`/`static/`
tree is preserved unmodified under `upstream_reference/` (git history keeps
full provenance). The mission-owned package lives in `src/hcmaic/`.
Reason: the upstream root scripts cannot run in this environment (missing
deps, hard-coded `D:/` paths, table/dim mismatches) and would make the
runnable tree ambiguous. Rejected: refactoring `app.py` in place — its global
state and import-time side effects (StaticFiles mount on `D:/keyframes`)
make incremental refactor slower than re-implementing behind the same
endpoint concepts with tests.

## 2026-07-26 — D2: Rebuild the operator UI compactly instead of porting 5.3k lines

The upstream `static/` UI is coupled to upstream endpoint shapes and contains
large unrelated features (YouTube embed, TRECVID panels, batch similarity
matrix UI). The mission UI re-implements the required operator loop (query,
top-K grid, detail, timeline, history, submission preview) in ~600 lines of
plain HTML/CSS/JS against the new API contract. Interaction patterns
(grid + detail + surrounding frames) follow the upstream design.
Rejected: adapting upstream JS wholesale — higher risk and slower under the
timebox; React — out of scope.

## 2026-07-26 — D3: Deterministic mock embedding = palette-token space

`DeterministicMockEmbeddingProvider` embeds images by quantizing an 8x8
downsample onto a fixed 8-color palette and projecting palette counts through
fixed seeded unit vectors; text is tokenized and color/shape vocabulary words
project through the same vectors (other tokens get tiny hashed weight).
Reason: gives *real* image-content→text retrieval signal on the fixture
(query "red" finds the red keyframe) so end-to-end plumbing, ranking, and
evaluation are meaningfully testable offline and deterministically.
Rejected: pure hash-of-bytes embeddings — no cross-modal signal, evaluation
would be vacuous; bundling a tiny real model — violates no-weights tests.

## 2026-07-26 — D4: SQLite is not used in v0; catalog is JSONL + in-memory maps

The catalog is `catalog.jsonl` (deterministic order) loaded into memory.
Fixture and early competition scale (hundreds of thousands of rows) fit
comfortably; artifacts stay diffable and hash-stable. The upstream SQLite
idea is kept as a later optimization documented for TV2.
Rejected: porting the upstream SQLite schema now — it duplicated video
metadata per keyframe row and disagreed with the serving code anyway.

## 2026-07-26 — D5: ExactNumpyIndex mandatory, FAISS optional extra

Core index is exact inner-product over L2-normalized float32 with
deterministic tie-break (score desc, then frame_id asc). FAISS
(`faiss-cpu`) is an optional `[faiss]` extra implementing the same
`SearchIndex` interface and verified against ExactNumpyIndex when installed.
Reason: mission mandate; FAISS wheels on Windows/3.11 are usually fine but
must never block the critical path.

## 2026-07-26 — D6: Real CLIP provider = `openai/clip-vit-base-patch32` via transformers, optional extra

ViT-B/32-class model (~600 MB, 512-dim), CPU path mandatory, CUDA optional
with batch size 8 for 4 GB VRAM. Installed via `[clip]` extra; never imported
by tests. Reason: transformers is the same stack the upstream used (smaller
model swap is a one-line config change for the team); open_clip rejected to
keep one optional heavy dependency, not two.

## 2026-07-26 — D7: Embedding dimension is a manifest property, not a constant

Dimension is recorded by the provider and written to `index_manifest.json`;
serving validates catalog/embeddings/id-map/manifest agreement at load time
and refuses to start on mismatch. Reason: the upstream 512-vs-1280 bug class
must be structurally impossible.

## 2026-07-26 — D8: Timestamps

BTC mapping gives `pts_time` (seconds, float). `timestamp_ms` is
`round(pts_time * 1000)`; `pts` is preserved as given. `frame_idx / fps` is
never used to derive timestamps (VFR hazard, per SkillPixel Buoi 2).

## 2026-07-26 — D9: Fail closed on artifact and provider drift

Artifact loading rejects manifest/hash/frame-count/dimension mismatches,
non-finite or non-normalized embeddings, and a text-provider version that
does not match the index manifest. Reason: stale or mixed artifacts must not
silently produce a competition result. Rejected: warning-only startup checks.

## 2026-07-26 — D10: Keep operator metadata text-safe and paths private

The API omits local dataset/artifact paths from `/system/info`, whitespace-only
queries are invalid, and UI metadata is rendered as text nodes rather than
HTML. Reason: browser-visible data must not become an injection or leak local
filesystem layout. Rejected: trusting fixture metadata or exposing debug paths.
