# 00 — Tổng quan Competitive Foundation v1

## Bài toán từ đầu

HCMAIC Video Retrieval yêu cầu tìm đúng khoảnh khắc trong một tập video lớn từ
mô tả chữ, chữ xuất hiện trong hình, lời nói, caption hoặc chuỗi hội thoại.
Hệ thống vì vậy có hai pipeline:

```text
Offline: video/keyframe -> metadata/shot/modality -> embedding -> index
Online: query -> search từng modality -> fusion -> temporal -> rerank -> kết quả
```

`Embedding` là vector số đại diện nội dung. `Index` là cấu trúc tìm vector gần
nhất. `Fusion` hợp nhất nhiều kênh như hình ảnh, OCR (nhận dạng chữ) và ASR
(nhận dạng tiếng nói).

## Ba mức sẵn sàng

- Software-ready: code, contract và test chạy ổn.
- Benchmark-ready: có dữ liệu/query/qrels thật, phép đo lặp lại và baseline so
  sánh.
- Competition-ready: vượt gate trên dữ liệu BTC, đúng luật và schema nộp bài,
  chịu được quy mô/phần cứng/ngày thi.

Repository hiện software-ready ở phần lớn contract, fixture-benchmark-ready,
nhưng chưa competition-ready.

## Bảy gate

| Gate | Status | Bằng chứng mạnh nhất |
|---|---|---|
| G1 Data/timestamp | PARTIAL | OpenCV fixture + parser FFmpeg + rollback; VFR thật chưa chạy |
| G2 Shot/keyframe | PARTIAL | no-shot detector và within-shot sampler có test; detector thật chưa chạy |
| G3 Provider readiness | PARTIAL | mock verified, provider doctor; SigLIP2/Jina/real CLIP chưa chạy |
| G4 Multimodal contract | PARTIAL | mock OCR/ASR/caption records; chưa nối index model thật |
| G5 Fusion/temporal/reranker/feedback | PARTIAL | unit/orchestrator tests + browser feedback; service vẫn visual-first |
| G6 Benchmark harness | PASS | bốn report, frozen hashes và per-query evidence |
| G7 Scale/mock contest | PARTIAL | synthetic HNSW benchmark + browser E2E; chưa có BTC-scale/mock contest chính thức |

## Chưa chứng minh

Không có BTC dataset/rules/schema chính thức, không có điểm portal, không có
benchmark model trên dữ liệu thật, không có FFmpeg/VFR runtime trên máy này,
không có CUDA, và không có submission thật.
