# HCMAIC Competitive Foundation v1

Competitive Foundation v1 là nền tảng có module, test và benchmark harness để
team tiếp tục phát triển hệ thống video retrieval. Phiên bản này chưa được xem
là competition-ready.

## Những phần đã có

- Typed YAML config và artifact provenance có hash.
- Ingest video bằng staging, validate trước khi replace và có rollback test.
- Lưu decoder timestamp, `timestamp_source`, shot và sampling provenance.
- Mapping mới nhưng vẫn tương thích dataset cũ.
- Shot detector/sampler contract với deterministic fallback.
- Lazy embedding registry cho mock, CLIP, SigLIP2 và Jina CLIP v2.
- Feature contract cho visual, OCR, ASR, caption và metadata.
- RRF, weighted late fusion, temporal expansion và reranker interface.
- Optional FAISS HNSW cùng exact-versus-ANN synthetic benchmark.
- Frozen benchmark config và report theo từng query.
- API/UI hiển thị runtime, score, shot context, feedback và submission preview.

## Mức bằng chứng

| Nhãn | Ý nghĩa |
|---|---|
| `VERIFIED` | Đã chạy trực tiếp trên checkout hiện tại |
| `FIXTURE_VERIFIED` | Chỉ chạy bằng fixture deterministic |
| `SYNTHETIC_SCALE_VERIFIED` | Chạy trên vector synthetic |
| `INTERFACE_ONLY` | Chỉ có contract/code; backend hoặc model thật chưa chạy |
| `BLOCKED` | Thiếu dataset, schema, hardware, dependency hoặc weights |

Không được nâng mức bằng chứng dựa trên source inspection hoặc mock.

Tài liệu bắt đầu:

- [Report trạng thái](TRANG_THAI_HE_THONG.md)
- [Tổng quan](docs/competitive_v1/00_OVERVIEW.md)
- [Runbook](docs/competitive_v1/03_SETUP_AND_RUNBOOK.md)
- [Handoff session tiếp theo](docs/competitive_v1/NEXT_SESSION.md)
