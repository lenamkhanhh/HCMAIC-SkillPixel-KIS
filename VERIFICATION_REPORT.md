# HCMAIC system verification report

Date: 2026-07-26  
Scope: `system/` only. `Learn/` was read-only and unchanged.

## Result

The critical local retrieval path is verified and packaged. The implementation
is a deterministic, synthetic-fixture foundation; it is not evidence of final
BTC ranking quality, CUDA performance, or a completed competition submission.

| Gate | Result | Evidence |
|---|---|---|
| Pinned upstream and provenance | PASS | `UPSTREAM.md`, MIT `LICENSE`, pinned source commit `e2c52124e691fc2c71d187d8f587fbe1bcddc38b` |
| Locked environment | PASS | `uv sync --locked --python 3.11`; Python 3.11 |
| Import, CLI, build | PASS | `scripts/verify.ps1` |
| Lint and types | PASS | `uv run ruff check src tests scripts`; `uv run mypy src` |
| Tests | PASS | 118 passed with the FAISS extra installed |
| Coverage | PASS | 95% core coverage, target was at least 80% |
| Dependency audit | PASS | `pip-audit`: no known vulnerabilities; local `hcmaic` package is not on PyPI |
| Fixture end-to-end | PASS | validate, build, search, evaluate; 12 frames, 5 videos, 6 queries |
| API/UI smoke | PASS | FastAPI + Playwright browser run on `127.0.0.1:8026` |
| Package and checksum | PASS | See `artifacts/SHA256SUMS.txt` after the final package check |

## Verified behavior

- Fixture evaluation: Recall@1/5/10 = 1.0, MRR = 1.0; p50/p95 were about
  0.1–0.3 ms.
- `/health` reported 12 indexed frames, 5 videos, and the mock provider.
- The browser search for `a solid red keyframe` returned `L01_V001:001`;
  the detail image loaded at 64×48 pixels, the timeline showed 1/5/9 seconds,
  and the submission preview rendered the expected JSON.
- The final browser run had zero console errors and warnings. An initial
  `/favicon.ico` 404 was fixed with an inline data favicon.
- Artifact loading now fails closed on manifest/hash/frame-count/dimension
  drift, non-finite or non-normalized embeddings, and provider-version drift.
- `/system/info` does not expose local dataset or artifact paths. Metadata is
  rendered as DOM text nodes.

## Commands

```powershell
uv sync --locked --python 3.11
uv sync --locked --extra faiss
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv run pytest -q
uvx --from pip-audit pip-audit --path .venv\Lib\site-packages
```

## Real-CLIP smoke (added after the audit)

After the audit above, the optional CLIP path was exercised once on the
fixture (allowed: network available, mock critical path already green):

- `uv sync --extra clip --extra faiss` → torch 2.13.0+cpu,
  transformers 4.57.6 (PyPI Windows wheel is CPU-only; CUDA unavailable).
- `hcmaic build-index --provider clip` → 12 frames, dim 512; manifest
  records `openai/clip-vit-base-patch32`, device `cpu`, batch 16.
- CLI search "a solid red image" → rank 1 `L01_V001:001` (correct frame).
- Evaluator mode `real-clip-smoke`: 6/6 scored, Recall@1 = 1.0, MRR = 1.0,
  p50 18.9 ms, p95 43.0 ms (CPU text encoding dominates).

This is a smoke test on the 12-frame synthetic fixture only; it is not
evidence of BTC-scale retrieval quality or latency.

## Explicitly unverified

Real BTC data ingestion and license/permission for any supplied media,
CLIP retrieval quality at competition scale, GPU/CUDA execution (the
installed torch wheel is CPU-only), large-scale latency, and BTC portal
submission/evaluation remain team-owned follow-up work.
