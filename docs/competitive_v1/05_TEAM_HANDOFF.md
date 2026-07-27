# 05 — Handoff cho team 5 người

## Cách chia role đề xuất

| Thành viên | Role chính | Phạm vi |
|---|---|---|
| TV1 | Data/ingestion | dataset audit, timestamp, shot, mapping |
| TV2 | Visual ML | CLIP/SigLIP2/Jina, embedding benchmark |
| TV3 | Multimodal ML | OCR, ASR, caption, feature artifact |
| TV4 | Retrieval/backend | vector index, fusion, temporal, reranker, API |
| TV5 | Evaluation/UI/QA | query/qrels, metric, failure slices, UI, mock contest |

Lead giữ config chuẩn, acceptance gate và quyền quyết định `keep/reject`.

## Quy trình làm việc

1. Pull branch mới nhất.
2. Đọc `TRANG_THAI_HE_THONG.md` và task ID được giao.
3. Tạo branch riêng cho một task.
4. Viết failing test hoặc frozen benchmark trước.
5. Implement thay đổi nhỏ nhất.
6. Chạy narrow test rồi full verification.
7. Ghi experiment và evidence level.
8. Mở PR, yêu cầu một thành viên khác review.

## Contract khi merge

PR phải ghi:

- task ID và hypothesis;
- file đã thay đổi;
- command đã chạy;
- metric before/after;
- evidence level;
- blocker hoặc phần chưa verify;
- rollback/fallback;
- ảnh hưởng tới artifact/config hash.

Không merge model/fusion trick nếu chỉ có screenshot hoặc một điểm portal không
có control.
