# Handoff Competitive Foundation v1

Đọc file này trước khi sửa code.

## Repository

- Workspace: `D:\Code\Code\AIO\Code\HCMAIC\system`
- Branch: `hcmaic-2026-foundation`
- Starting checkpoint: `c0db285`
- Foundation checkpoint trước đợt Việt hoá tài liệu: `bf1168f`
- `origin` là upstream công khai, không phải remote team.
- Khi publish cá nhân, dùng remote riêng; không thay hoặc force-push `origin`.

Resume:

```powershell
cd D:\Code\Code\AIO\Code\HCMAIC\system
git status --short --branch
git log --oneline -8
uv run pytest
```

## Trạng thái phase

| Phase | Trạng thái | Evidence |
|---|---|---|
| Audit/plan | Done | mission và plan đã commit |
| Config/provenance | Done cho foundation | typed YAML, config/artifact hash |
| Ingestion/timestamp | Partial | OpenCV fixture và FFmpeg parser test |
| Shot/mapping | Partial | deterministic contract/sampler |
| Provider/multimodal | Partial | lazy registry, doctor, mock artifact |
| Fusion/retrieval | Done về contract | RRF, weighted fusion, temporal, reranker test |
| Scale/benchmark | Partial | synthetic HNSW và frozen proxy harness |
| API/UI | Fixture verified | browser search/detail/feedback/preview pass |
| Documentation | Done | README, runbook, task board, BTC procedure bằng tiếng Việt |

## Seven gates

| Gate | Status | Vì sao chưa mạnh hơn |
|---|---|---|
| G1 Data/timestamp | `PARTIAL` | chưa chạy FFmpeg non-zero/VFR thật |
| G2 Shot/keyframe | `PARTIAL` | PySceneDetect/TransNetV2 interface-only |
| G3 Embedding provider | `PARTIAL` | real CLIP/SigLIP2/Jina chưa chạy |
| G4 Multimodal | `PARTIAL` | real extractor/index chưa nối |
| G5 Fusion/temporal/reranker | `PARTIAL` | production service còn visual-first |
| G6 Benchmark harness | `PASS` | pass về reproducibility, không phải model quality |
| G7 Scale/mock contest | `PARTIAL` | chưa có BTC-scale/schema/mock contest |

Không được gọi Competitive Foundation v1 complete khi còn gate `PARTIAL`.

## Verification gần nhất

```text
scripts/verify.ps1                    -> pass
pytest + coverage                     -> 184 passed, 90%
ruff                                  -> pass
mypy                                  -> pass
browser E2E                           -> pass, console sạch
```

Fixture benchmark:

```text
queries=6
Recall@1/5/10/100=1.0
MRR=1.0
evidence=FIXTURE_VERIFIED
```

Synthetic ANN:

```text
vectors=1000, dimension=64, queries=20, top_k=20
Recall@20=1.0
p95=0.125 ms
evidence=SYNTHETIC_SCALE_VERIFIED
```

Các số này không dự đoán BTC performance.

## Blocker

- chưa có official BTC dataset/rule/schema;
- FFmpeg/ffprobe không có trên PATH;
- real model và CUDA chưa chạy;
- production service còn visual-first;
- feedback chỉ lưu trong RAM;
- duplicate `charset-normalizer` metadata trong `.venv`;
- upstream remote không có quyền write.

## Single next action

Nếu chưa có BTC dataset: làm **F-02** trong `TEAM_TASK_BOARD.md`.

Nếu BTC dataset đã phát hành: làm **BTC-01** trong
`06_WHEN_BTC_DATASET_ARRIVES.md` trước mọi thay đổi model/fusion.
