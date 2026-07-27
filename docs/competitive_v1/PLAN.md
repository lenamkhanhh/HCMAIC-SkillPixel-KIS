# Plan tiếp tục Competitive Foundation v1

Đọc theo thứ tự:

1. `../../TRANG_THAI_HE_THONG.md`
2. `NEXT_SESSION.md`
3. `../../TEAM_TASK_BOARD.md`
4. `05_TEAM_HANDOFF.md`
5. `07_KNOWN_GAPS_AND_RISKS.md`

## Trình tự phát triển

```text
BTC release
-> audit read-only + hash
-> freeze query/qrels
-> chạy incumbent không thay đổi
-> nối từng modality
-> paired benchmark + failure slices
-> chỉ giữ measured gain
-> official-schema dry run
```

Trước khi có frozen baseline, không ưu tiên learned fusion, VLM reranker,
KISC automation hoặc model shopping diện rộng. Luôn giữ `mock` và
`exact-numpy` làm offline fallback.
