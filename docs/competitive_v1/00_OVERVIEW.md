# 00 — Tổng quan Competitive Foundation v1

## Bài toán

HCMAIC Video Retrieval yêu cầu tìm đúng khoảnh khắc trong tập video lớn từ mô
tả text, chữ trong hình, lời nói, caption, metadata hoặc hội thoại.

```text
Offline: video/keyframe -> feature -> embedding -> index
Online: query -> search từng modality -> fusion -> temporal -> rerank -> kết quả
```

- `Embedding`: vector đại diện nội dung.
- `Index`: cấu trúc tìm vector gần nhất.
- `OCR`: nhận dạng chữ trong hình.
- `ASR`: nhận dạng lời nói.
- `Fusion`: hợp nhất kết quả từ nhiều modality.

## Ba mức sẵn sàng

- **Software-ready:** code, contract và test chạy ổn.
- **Benchmark-ready:** có frozen data/query/qrels, metric và baseline.
- **Competition-ready:** vượt gate bằng dữ liệu BTC, đúng rule/schema và chịu
  được scale ngày thi.

System hiện software-ready ở phần lớn foundation, nhưng chưa
competition-ready.

## Bảy gate

| Gate | Trạng thái | Bằng chứng mạnh nhất |
|---|---|---|
| G1 Data/timestamp | `PARTIAL` | fixture ingestion, FFmpeg parser, rollback test |
| G2 Shot/keyframe | `PARTIAL` | deterministic detector/sampler test |
| G3 Provider readiness | `PARTIAL` | mock verified, provider doctor |
| G4 Multimodal contract | `PARTIAL` | mock OCR/ASR/caption artifact |
| G5 Fusion/temporal/reranker | `PARTIAL` | module/orchestrator test + feedback E2E |
| G6 Benchmark harness | `PASS` | frozen hash và per-query report |
| G7 Scale/mock contest | `PARTIAL` | synthetic HNSW và browser E2E |

## Chưa được chứng minh

- chất lượng trên dataset BTC;
- real FFmpeg/VFR runtime;
- model SigLIP2/Jina/CLIP thật;
- real OCR/ASR/caption;
- CUDA và competition-scale latency;
- official submission.

Chi tiết mới nhất: [TRANG_THAI_HE_THONG.md](../../TRANG_THAI_HE_THONG.md).
