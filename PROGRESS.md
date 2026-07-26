# Progress log

Newest entries at the bottom. Every entry: what ran, what resulted.

## 2026-07-26 Phase 0 — Bootstrap

- Read required context (`TEAM_PIPELINE_TASKS_5_NGUOI.md`, `TEAM_TASK_BOARD.md`,
  `00_NOTE_TONG_HOP.md`, SkillPixel Buoi 1+2 notes, BTC training texts 1-2,
  `lab1-fastapi-student/`). `Learn/` untouched.
- `system/` confirmed absent before start (`ls system` → exit 2).
- `git clone https://github.com/hhlearntocode/SoftSignalsRetrievalSystems-AIC2025 system`
  → OK; `git checkout e2c52124e691fc2c71d187d8f587fbe1bcddc38b` → HEAD at
  `e2c5212 clean`. LICENSE (MIT) present. History preserved.
- `git checkout -b hcmaic-2026-foundation` → branch created.
- Baseline inspection (reading, not executing — the heavy model download was
  skipped deliberately):
  - `app.py`: hard-coded `D:/...` DB/index/keyframes paths; `EMBEDDING_DIM=1280`;
    `laion/CLIP-ViT-bigG-14` default; reads table `keyframe_embeddings`;
    `googletrans` imported but absent from `requirements.txt`; startup
    swallows exceptions; `StaticFiles(directory="D:/keyframes")` mounted at
    import time → import fails without that directory.
  - `migrate_embeddings.py`: writes table `keyframes`, dim 512, local
    `image_retrieval.db` → serving/migration mismatch confirmed on all three.
  - No tests/CI/lockfile/evaluator anywhere in the tree.
- Tooling: git 2.52.0, uv 0.11.20, global Python 3.14.1 (not used),
  cpython 3.11 available for uv download. Git identity present (`Khanhdz`),
  not modified.
- Created `UPSTREAM.md`, `GOAL.md`, `DECISIONS.md`, `PROGRESS.md`.
