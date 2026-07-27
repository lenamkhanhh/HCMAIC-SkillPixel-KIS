# Report trạng thái hệ thống HCMAIC

Cập nhật tại commit `bf1168f` trên branch `hcmaic-2026-foundation`.

## Kết luận ngắn

System hiện **software-ready** và chạy ổn trên fixture. System **chưa
competition-ready** vì chưa có dataset BTC, chưa benchmark model thật và
production retrieval vẫn là visual-first.

## Pipeline đang chạy thật

```text
Offline:
video/keyframe
-> ingest + timestamp
-> mapping/catalog/validation
-> embedding
-> vector index
-> artifact có hash

Online:
query
-> query embedding
-> visual vector index
-> top-K keyframe
-> FastAPI/UI
-> timeline, feedback, submission preview
```

RRF, weighted late fusion, temporal expansion, reranker và multimodal
orchestrator đã có contract/test, nhưng chưa được nối hoàn toàn vào
`RetrievalService`.

## Bằng chứng đã kiểm tra

- `184 passed`, coverage `90%`.
- Ruff pass.
- Mypy pass trên 45 source files.
- Fixture validate -> build index -> search -> evaluate pass.
- Browser E2E cho search/detail/feedback/submission preview pass.
- Benchmark manifest khớp code commit và có hash của config/dataset/query/qrels.
- Fixture benchmark: 6 query, Recall@1/5/10/100 và MRR đều `1.0`.
- Synthetic HNSW: 1.000 vector x 64 chiều, Recall@20 `1.0`,
  p95 `0.125 ms`.

Các metric trên chỉ là fixture/synthetic evidence.

## Bảy gate

| Gate | Trạng thái | Phần còn thiếu |
|---|---|---|
| G1 Data/timestamp | `PARTIAL` | FFmpeg/VFR thật |
| G2 Shot/keyframe | `PARTIAL` | PySceneDetect/TransNetV2 thật |
| G3 Embedding provider | `PARTIAL` | CLIP/SigLIP2/Jina chạy thật |
| G4 Multimodal | `PARTIAL` | OCR/ASR/caption thật và index riêng |
| G5 Fusion/temporal/reranker | `PARTIAL` | nối vào production service |
| G6 Benchmark harness | `PASS` | pass về harness, không phải model quality |
| G7 Scale/mock contest | `PARTIAL` | BTC-scale, GPU, schema, mock contest |

## Blocker hiện tại

- Chỉ có `data/sample`, chưa có dataset BTC.
- Không có rule/schema submission chính thức.
- FFmpeg/ffprobe không có trên PATH.
- SigLIP2 và Jina doctor thấy dependency nhưng vẫn `INTERFACE_ONLY`.
- Chưa có real OCR/ASR/caption hoặc CUDA benchmark.
- Feedback chỉ lưu trong RAM.
- `uv pip check` báo trùng metadata `charset-normalizer 3.4.7` và `3.4.9`.
- `origin` là upstream công khai và tài khoản hiện tại chỉ có quyền `READ`.

## Việc tiếp theo

Nếu chưa có dataset BTC, làm **F-02**: nối một mock modality vào index riêng,
đưa nó qua orchestrator/fusion thật và chứng minh visual retrieval vẫn chạy khi
modality đó bị disable hoặc fail.

Khi BTC phát hành dataset, dừng đổi model và làm **BTC-01** trước: audit
read-only, hash/schema/corruption/audio/language/scale, rồi freeze 20–50 query
validation để đo incumbent không thay đổi.
