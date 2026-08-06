# HCMAIC 2026 — hệ thống tìm kiếm keyframe

Đây là hệ thống tìm kiếm khoảnh khắc trong video dành cho HCMAIC 2026 Bảng A.
Project được phát triển từ
[SoftSignalsRetrievalSystems-AIC2025](https://github.com/hhlearntocode/SoftSignalsRetrievalSystems-AIC2025)
theo giấy phép MIT; commit gốc và phần thay đổi được ghi tại
[UPSTREAM.md](UPSTREAM.md).

## SkillPixel KIS repository

This public repository is the preserved-history SkillPixel KIS checkout:
[HCMAIC-SkillPixel-KIS](https://github.com/lenamkhanhh/HCMAIC-SkillPixel-KIS).
Its main branch is copied from the raw-video-first implementation branch at
commit 7dbc1aa; no previous milestone commits were squashed or removed.

The production contract is:

    raw SkillPixel videos
      -> dense frames + source_frame_idx mapping
      -> real provider embeddings + normalized exact IndexFlatIP
      -> TKIS text / VKIS image retrieval
      -> optional validated OCR, object and ASR channels
      -> RRF/dedup/bounded rerank
      -> video_filename.mp4,source_frame_idx
      -> validated submission.csv and top-k evidence

Raw videos, query data, model weights, embedding matrices, FAISS indexes and
secrets are external inputs/artifacts and are never committed. Tests are
offline and local-files-only by default.

### Reproduce the verified visual baseline

From PowerShell, set the four SkillPixel paths explicitly and run:

    cd D:\Code\Code\AIO\Code\HCMAIC-SkillPixel-KIS
    $env:SKILLPIXEL_RAW_INPUT = "D:\path\to\SkillPixel\videos"
    $env:SKILLPIXEL_QUESTIONS = "D:\path\to\SkillPixel\questions.csv"
    $env:SKILLPIXEL_CORPUS = "D:\path\to\SkillPixel\corpus.csv"
    $env:SKILLPIXEL_SIGLIP2_MODEL = "D:\path\to\siglip2-base-patch16-224"
    $env:SKILLPIXEL_RUN_ROOT = "D:\path\to\run"
    $env:SKILLPIXEL_DEVICE = "cpu"
    $env:SKILLPIXEL_LOCAL_FILES_ONLY = "true"
    uv run python scripts\skillpixel_kis_build.py --config configs\skillpixel_kis.yaml --stage catalog
    uv run python scripts\skillpixel_kis_build.py --config configs\skillpixel_kis.yaml --stage visual
    uv run python scripts\skillpixel_kis_infer.py --config configs\skillpixel_kis.yaml --run-dir $env:SKILLPIXEL_RUN_ROOT --top-k 100
    uv run python scripts\skillpixel_kis_validate.py --config configs\skillpixel_kis.yaml --run-dir $env:SKILLPIXEL_RUN_ROOT

The verified SkillPixel rehearsal processed 250 raw videos, 9,835 sampled
frames and 100 queries (50 TKIS, 50 VKIS) with real SigLIP2 768D and exact
FAISS. Without official qrels, reports retain UNVALIDATED_ON_HCMAIC.

### Verification

## Latest SkillPixel KIS status (2026-08-06)

The current raw-video-first TKIS/VKIS report is
[docs/HCMAIC_SKILLPIXEL_KIS_FINAL_REPORT.md](docs/HCMAIC_SKILLPIXEL_KIS_FINAL_REPORT.md).
The verified run uses real local SigLIP2 (768D), normalized exact FAISS
`IndexFlatIP`, 250 raw videos, 9,835 sampled frames and 100 queries. The final
submission and evidence validate successfully. OCR, object, ASR and Jina are
implemented as explicit real-provider paths; unavailable weights or GPU jobs
are recorded as unavailable and never silently replaced by mock scores.

No official HCMAIC qrels are present, so the quality status remains
`UNVALIDATED_ON_HCMAIC` and no SOTA claim is made.

    uv run pytest
    uv run ruff check src tests scripts
    uv run mypy src

Do not enable network/model downloads implicitly. A GPU/Kaggle run must record
the requested and selected device, model revision, artifact hashes and
fallback reason in its manifest.

```text
keyframe + mapping CSV + metadata
  -> validate -> catalog -> embedding -> vector index
  -> CLI / FastAPI / giao diện tìm kiếm
  -> submission preview -> evaluator
```

Đường chạy mặc định hoạt động offline trên CPU với fixture có sẵn. Không cần
network, GPU, FFmpeg, model weights hoặc dữ liệu riêng tư để chạy test.

## Trạng thái hiện tại

- Code foundation chạy ổn: `184 passed`, coverage `90%`, Ruff và Mypy đều pass.
- Pipeline fixture từ validate đến search/evaluate chạy end-to-end.
- API/UI đã được browser E2E cho search, timeline, feedback và submission preview.
- Benchmark harness có hash của config, dataset, query, qrels và code commit.
- Chưa có dataset BTC chính thức.
- SigLIP2, Jina CLIP v2, OCR, ASR và caption thật chưa được chạy.
- Production retrieval hiện vẫn là visual-first; multimodal fusion mới có
  contract và test riêng.

Đọc [TRANG_THAI_HE_THONG.md](TRANG_THAI_HE_THONG.md) để xem report hiện tại và
[TEAM_TASK_BOARD.md](TEAM_TASK_BOARD.md) để nhận task.

## Cài đặt trên Windows

Yêu cầu: PowerShell, Git và `uv`.

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC\system
uv python install 3.11
uv sync --locked --extra faiss --extra video
```

Chỉ thêm `--extra clip` khi thực sự cần chạy model CLIP:

```powershell
uv sync --locked --extra clip --extra faiss --extra video
```

## Chạy nhanh bằng fixture

```powershell
uv run hcmaic validate-data --input data/sample
uv run hcmaic build-index --input data/sample --output artifacts/sample
uv run hcmaic search --index artifacts/sample `
  --query "a solid red keyframe" --top-k 5
uv run hcmaic serve --index artifacts/sample --port 8000
```

Sau đó mở `http://127.0.0.1:8000/`.

Chạy evaluator:

```powershell
uv run hcmaic evaluate `
  --index artifacts/sample `
  --queries data/sample/queries.jsonl `
  --qrels data/sample/qrels.jsonl
```

Các endpoint chính:

- `GET /health`
- `GET /system/info`
- `POST /search`
- `GET /frames/{frame_id}`
- `GET /videos/{video_id}/timeline`
- `POST /feedback`
- `POST /submit/preview`

## Ingest video

```powershell
uv run hcmaic ingest-video `
  --input <video-hoac-folder> `
  --output data/myset `
  --interval 2.0

uv run hcmaic validate-data --input data/myset
uv run hcmaic build-index --input data/myset --output artifacts/myset
```

Nếu máy có cả `ffmpeg` và `ffprobe`, system ưu tiên FFmpeg. Nếu không có,
system dùng OpenCV. Kết quả, warning và lỗi từng video được ghi vào
`<output>/ingest_report.json`.

Muốn ingest lại video đã tồn tại phải dùng `--force`. Dữ liệu mới được tạo và
validate trong staging trước khi thay dữ liệu cũ.

## Cấu trúc dataset tương thích BTC

```text
<dataset>/
├── keyframes/<video_id>/<nnn>.jpg
├── keyframe_mapping.csv
└── media-info/<video_id>.json
```

System cũng đọc được dạng mapping tách riêng:
`map-keyframes/<video_id>.csv`.

## Benchmark

```powershell
uv run hcmaic provider-doctor --provider siglip2

uv run hcmaic scale-benchmark `
  --vectors 1000 --dimension 64 --queries 20 --top-k 20

uv run hcmaic benchmark `
  --config configs/competitive_v1.yaml `
  --out artifacts/benchmark/competitive-v1
```

Kết quả fixture hoặc synthetic chỉ chứng minh pipeline chạy đúng. Không được
dùng chúng để kết luận chất lượng thi thật.

## Kiểm tra toàn bộ

```powershell
uv run pytest
uv run pytest --cov=src/hcmaic
uv run ruff check src tests scripts
uv run mypy src
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify.ps1
uv pip check
git diff --check
```

`uv pip check` trên environment hiện tại đang báo hai metadata
`charset-normalizer` cùng tồn tại. Xem hướng xử lý tại
[runbook](docs/competitive_v1/03_SETUP_AND_RUNBOOK.md).

## Tài liệu dành cho team

Đọc theo thứ tự:

1. [TRANG_THAI_HE_THONG.md](TRANG_THAI_HE_THONG.md)
2. [TEAM_TASK_BOARD.md](TEAM_TASK_BOARD.md)
3. [Tổng quan kiến trúc](docs/competitive_v1/00_OVERVIEW.md)
4. [Hướng dẫn cài đặt và vận hành](docs/competitive_v1/03_SETUP_AND_RUNBOOK.md)
5. [Handoff cho 5 thành viên](docs/competitive_v1/05_TEAM_HANDOFF.md)
6. [Khi BTC phát hành dataset](docs/competitive_v1/06_WHEN_BTC_DATASET_ARRIVES.md)
7. [Rủi ro và phần còn thiếu](docs/competitive_v1/07_KNOWN_GAPS_AND_RISKS.md)

Các file `PROGRESS.md`, `OVERNIGHT_REPORT.md`, `FINAL_HANDOFF.md` và
`VERIFICATION_REPORT.md` là lịch sử của các milestone trước. Trạng thái mới
nhất luôn nằm trong `TRANG_THAI_HE_THONG.md`.
