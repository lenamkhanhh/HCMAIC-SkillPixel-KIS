# Handoff hiện tại cho team

File này chỉ là entrypoint. Trạng thái đầy đủ nằm tại:

- [TRANG_THAI_HE_THONG.md](TRANG_THAI_HE_THONG.md)
- [TEAM_TASK_BOARD.md](TEAM_TASK_BOARD.md)
- [Runbook](docs/competitive_v1/03_SETUP_AND_RUNBOOK.md)
- [Handoff session](docs/competitive_v1/NEXT_SESSION.md)

## Chạy system

```powershell
uv sync --locked --extra faiss --extra video
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv run hcmaic serve --index artifacts/verify --data-root data/sample --port 8000
```

## Trạng thái ngắn

- Foundation chạy ổn trên fixture.
- 184 test pass, coverage 90%.
- Production retrieval còn visual-first.
- Real model, BTC dataset, CUDA và official submission chưa được verify.
- Task tiếp theo khi chưa có BTC dataset: F-02.

Không dùng các report milestone cũ để thay thế report trạng thái hiện tại.
