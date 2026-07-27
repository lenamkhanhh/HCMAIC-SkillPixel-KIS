# Tiến độ hệ thống

## Milestone đã hoàn thành

1. Fixture-safe keyframe retrieval MVP.
2. Raw-video ingest bằng OpenCV fallback và FFmpeg contract.
3. Timestamp provenance và safe `--force` staging/rollback.
4. Typed config và artifact provenance.
5. Shot/mapping contract.
6. Lazy embedding provider registry.
7. Multimodal feature/artifact contract.
8. RRF, weighted fusion, temporal expansion, reranker và feedback contract.
9. FAISS HNSW synthetic benchmark.
10. Frozen proxy benchmark harness.
11. API/UI runtime visibility, shot context, feedback và preview.
12. Tài liệu handoff tiếng Việt.

## Checkpoint

- Foundation checkpoint: `bf1168f`
- Branch: `hcmaic-2026-foundation`
- Verification: 184 test pass, coverage 90%.

## Chưa hoàn thành

- real BTC adapter/benchmark;
- real FFmpeg VFR validation;
- real embedding/OCR/ASR/caption provider;
- per-modality production index;
- fusion/orchestrator nối vào `RetrievalService`;
- CUDA/competition-scale benchmark;
- official submission adapter.

Task cụ thể nằm tại [TEAM_TASK_BOARD.md](TEAM_TASK_BOARD.md).
