# HCMAIC Competitive Foundation v1

Competitive Foundation v1 is a modular, benchmarkable extension of the
existing visual keyframe-search MVP. It does not claim competition readiness.

## Delivered foundation

- typed YAML configuration and hashed artifact provenance;
- staged/validated raw-video replacement and decoder-derived FFmpeg timestamps;
- legacy-compatible extended frame/shot/timestamp mapping;
- deterministic shot/sampling contracts and explicit heavy-detector slots;
- lazy embedding registry with mock, CLIP control, SigLIP2 and Jina CLIP v2 slots;
- canonical multimodal feature records plus mock OCR/ASR/caption providers;
- Reciprocal Rank Fusion (RRF), weighted late fusion, temporal expansion,
  passthrough reranker and session-feedback contracts;
- optional FAISS HNSW with exact-versus-ANN synthetic benchmarking;
- reproducible proxy benchmark reports with config/data/query/qrels/code hashes;
- API/UI surfaces for runtime identity, shot context, evidence scores and local
  feedback.

## Evidence boundary

| Evidence | Current meaning |
|---|---|
| `VERIFIED` | Executed locally on this checkout |
| `FIXTURE_VERIFIED` | Executed only on deterministic synthetic fixture data |
| `SYNTHETIC_SCALE_VERIFIED` | Executed on generated random vectors |
| `INTERFACE_ONLY` | Contract/fake/diagnostic exists; real backend not run |
| `BLOCKED` | Requires BTC data, official schema, hardware, dependency or weights |

Start with [the overview](docs/competitive_v1/00_OVERVIEW.md), then follow the
[setup runbook](docs/competitive_v1/03_SETUP_AND_RUNBOOK.md). Coding agents
must read [NEXT_SESSION.md](docs/competitive_v1/NEXT_SESSION.md) first.
