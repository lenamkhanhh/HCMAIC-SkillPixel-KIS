# Report verification hiện tại

## Gate code

| Kiểm tra | Kết quả |
|---|---|
| Mypy | Pass trên 45 source files |
| Ruff | Pass |
| Pytest | 184 passed |
| Coverage | 90% |
| Fixture validation | Pass |
| Build index | Pass |
| CLI search | Pass |
| Fixture evaluator | Pass |
| Browser E2E | Pass |
| `git diff --check` | Pass |

Có một dependency deprecation warning từ Starlette/httpx nhưng không làm test
fail.

## Dependency environment

`uv pip check` đang báo hai `charset-normalizer` dist-info cùng tồn tại:

- `3.4.7`
- `3.4.9`

Đây là environment defect, chưa phải source dependency conflict. Tạo lại
`.venv` riêng là hướng xử lý an toàn.

## Evidence boundary

- Mock/fixture: `FIXTURE_VERIFIED`.
- HNSW random vector: `SYNTHETIC_SCALE_VERIFIED`.
- SigLIP2/Jina/real OCR/ASR/caption: `INTERFACE_ONLY`.
- BTC quality/score: chưa có evidence.

Chi tiết gate cạnh tranh nằm tại
[TRANG_THAI_HE_THONG.md](TRANG_THAI_HE_THONG.md).
