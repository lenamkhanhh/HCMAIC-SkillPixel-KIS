# 07 — Phần còn thiếu và rủi ro

| Gap/risk | Trạng thái | Việc tiếp theo |
|---|---|---|
| Chưa có dataset/rule/schema BTC | `BLOCKED` | chạy BTC-01 khi được phát hành |
| FFmpeg/VFR runtime | `INTERFACE_ONLY` | test video non-zero/VFR thật |
| PySceneDetect/TransNetV2 | `INTERFACE_ONLY` | chỉ implement sau paired shot study |
| SigLIP2/Jina CLIP v2 | `INTERFACE_ONLY` | cached-weight smoke có kiểm soát |
| OCR/ASR/caption thật | `INTERFACE_ONLY` | chọn provider sau data audit |
| Multimodal index nối vào API | `PARTIAL` | làm F-02 |
| Learned fusion/VLM rerank/KISC | `DEFERRED` | chỉ làm sau frozen baseline |
| CUDA và competition-scale latency | `BLOCKED` | benchmark GPU và vector count thật |
| Official submission | `BLOCKED` | adapter + dry run sau khi có schema |
| Browser E2E | `VERIFIED_LOCAL` | rerun khi API/UI contract đổi |
| Trùng charset-normalizer metadata | `ENVIRONMENT_DEFECT` | rebuild `.venv` an toàn |
| Overclaim từ fixture | `PERMANENT_RISK` | giữ evidence label trong mọi report |

Technical debt:

- feedback chỉ lưu trong RAM;
- production service còn visual-first;
- HNSW artifact chưa persist đầy đủ parameters riêng khỏi config;
- legacy mapping loader cố ý permissive;
- `--force` rollback là multi-file best-effort transaction.
