# Mission goal

Fork the pinned SoftSignals AIC2025 system and turn it into a complete,
reproducible, local HCMAIC keyframe-search MVP that a five-person team can
run, test, extend, and benchmark on this Windows laptop (Python 3.11 via uv,
CPU-first, 4 GB VRAM GPU optional) without network, model weights, FFmpeg, or
private data.

## Verified critical path

```text
keyframe images + keyframe mapping + optional metadata
  -> validation and normalized catalog
  -> image embeddings
  -> searchable vector index
  -> text query embedding
  -> ranked top-K keyframes
  -> FastAPI
  -> local operator UI
  -> timeline/evidence inspection
  -> canonical submission preview
  -> evaluator and reproducible reports
```

## Definition of Done (summary)

- `system/` self-contained; bootstrap documented and verified.
- Fixture dataset validates; catalog, deterministic embeddings, and
  ExactNumpyIndex build; CLI + API search return correctly mapped frames.
- Operator UI shows ranked keyframes, timeline, and submission preview.
- Evaluator produces Recall@1/5/10, MRR, p50/p95 latency.
- Full test suite passes offline; security checks pass.
- `VERIFICATION_REPORT.md` honest; final ZIP + SHA256 verified;
  `FINAL_HANDOFF.md` complete.

Real CLIP, FAISS, CUDA: optional; must never block the mandatory path.
Never claim competition retrieval quality from the synthetic fixture.
