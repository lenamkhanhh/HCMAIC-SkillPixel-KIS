# Task board Competitive Foundation

Chưa gán tên thành viên. Lead chọn owner theo role sau khi audit dataset và
baseline.

| Task ID | Ưu tiên | Gate | Module | Việc cần làm | Phụ thuộc | Definition of Done | Role phù hợp | Trạng thái |
|---|---|---|---|---|---|---|---|---|
| F-01 | P0 | G1 | ingestion | Kiểm tra FFmpeg với non-zero timestamp/VFR thật | FFmpeg fixture | timestamp khớp ffprobe; rollback test pass | Data engineer | Blocked trên máy hiện tại |
| F-02 | P0 | G4/G5 | features/index | Nối một modality artifact vào index riêng và orchestrator | BTC-like fixture | visual vẫn chạy khi modality bị disable/fail | Multimodal engineer | Việc tiếp theo |
| F-03 | P0 | G3 | embedding | Smoke test SigLIP2 bằng cached weights | weights/GPU được duyệt | có dimension, normalization, latency, retrieval report | Visual ML engineer | Blocked |
| F-04 | P0 | G3 | embedding | Smoke test Jina CLIP v2 bằng cached weights | weights/GPU được duyệt | cùng gate với F-03 | Visual ML engineer | Blocked |
| F-05 | P1 | G7 | indexing | Persist HNSW index và parameters | quyết định scale | load validation + exact Recall@K report | Retrieval engineer | Partial |
| F-06 | P1 | G7 | UI | Browser E2E cho runtime/shot/feedback | server đang chạy | search/detail/feedback/preview pass, console sạch | UI/QA engineer | Done local |
| BTC-01 | P0 | G1 | data | Audit 6 giờ đầu khi có dataset BTC | dataset release | report hash/schema/corruption/audio/language/scale | Data engineer | Chờ BTC |
| BTC-02 | P0 | G6 | benchmark | Freeze 20–50 query/qrels và slices hợp lệ | BTC-01 | có version/hash và team review | Evaluation engineer | Chờ BTC |
| BTC-03 | P0 | G7 | submission | Official-schema dry run | rule/schema | payload validate; không upload thật | Backend engineer | Chờ BTC |
| C-01 | P2 | Later | fusion | Learned/adaptive fusion | frozen multimodal baseline | paired gain, không regression slice/latency | Retrieval ML engineer | Deferred |
| C-02 | P2 | Later | rerank | VLM reranker | GPU + top-N baseline | timeout/fallback + paired benchmark | Multimodal engineer | Deferred |
| C-03 | P2 | Later | KISC | Conversational agent | search/feedback API ổn định | scripted session eval + rollback | Agent engineer | Deferred |

## Quy tắc nhận task

1. Mỗi task có một owner chính và một reviewer.
2. Không thay nhiều primary factor trong cùng experiment.
3. Mọi thay đổi model/fusion phải có paired benchmark với incumbent.
4. Không merge nếu thiếu regression test hoặc evidence level.
5. Không dùng fixture score để tuyên bố chất lượng thi thật.
