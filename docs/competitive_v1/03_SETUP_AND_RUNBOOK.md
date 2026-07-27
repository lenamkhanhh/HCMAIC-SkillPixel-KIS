# 03 — Hướng dẫn cài đặt và vận hành

## Tạo environment

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC\system
uv python install 3.11
uv sync --locked --extra faiss --extra video
```

Chỉ thêm `--extra clip` khi cần chạy model thật:

```powershell
uv sync --locked --extra clip --extra faiss --extra video
```

Kiểm tra provider mà không tải weights:

```powershell
uv run hcmaic provider-doctor --provider siglip2
uv run hcmaic provider-doctor --provider jina-clip-v2
```

## Chạy fixture

```powershell
uv run python scripts/make_fixture.py
uv run hcmaic validate-data --input data/sample
uv run hcmaic build-index --input data/sample --output artifacts/sample
uv run hcmaic search --index artifacts/sample --query "a red bus" --top-k 10
uv run hcmaic serve --index artifacts/sample --port 8000
```

Mở `http://127.0.0.1:8000/`.

## Ingest raw video

```powershell
uv run hcmaic ingest-video `
  --input <video-hoac-folder> `
  --output data/myset `
  --interval 2

uv run hcmaic validate-data --input data/myset
```

`--force` tạo và validate staging output trước khi replace live files. System ưu
tiên FFmpeg khi có cả `ffmpeg` và `ffprobe`; nếu không thì dùng OpenCV.

## Benchmark

```powershell
uv run hcmaic benchmark `
  --config configs/competitive_v1.yaml `
  --out artifacts/benchmark/competitive-v1

uv run hcmaic scale-benchmark `
  --vectors 10000 --dimension 512 --queries 100 --top-k 100
```

## Verify toàn bộ

```powershell
uv run pytest
uv run pytest --cov=src/hcmaic
uv run ruff check src tests scripts
uv run mypy src
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv pip check
git diff --check
```

## Lỗi environment hiện biết

`uv pip check` có thể báo cùng tồn tại:

```text
charset_normalizer-3.4.7.dist-info
charset_normalizer-3.4.9.dist-info
```

Đây là duplicate installed metadata của environment hiện tại. Không xóa thủ
công trong environment dùng chung. Cách an toàn là tạo lại `.venv` riêng sau
khi xác nhận không có process đang dùng nó.

## Quy tắc evidence

- Fixture chỉ chứng minh plumbing.
- Synthetic benchmark chỉ chứng minh engineering path.
- `provider-doctor` không chứng minh model đã chạy.
- Chỉ dùng `REAL_RUNTIME_VERIFIED` sau khi backend/model thật được execute.
- Không gọi metric là BTC score nếu chưa chạy official dataset/evaluator.
