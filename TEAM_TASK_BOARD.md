# HCMAIC Competitive Foundation task board

No member names are assigned. Pick tasks by role after the baseline/data audit.

| Task ID | Priority | Gate | Module | Description | Dependencies | Definition of Done | Suggested owner role | Status | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| F-01 | P0 | G1 | ingestion | Real FFmpeg non-zero/VFR validation | FFmpeg fixture | timestamps match ffprobe ground truth; rollback test | Data engineer | Blocked locally | Interface tests |
| F-02 | P0 | G4 | features/index | Connect one modality artifact to its own index | BTC-like fixture | visual search survives disabled/failing modality | Multimodal engineer | Pending | Contracts green |
| F-03 | P0 | G3 | embedding | SigLIP2 cached-weight smoke | approved weights/GPU | dimension, normalization, latency, paired retrieval report | Visual ML engineer | Blocked | Doctor only |
| F-04 | P0 | G3 | embedding | Jina CLIP v2 cached-weight smoke | approved weights/GPU | same gates as F-03 | Visual ML engineer | Blocked | Doctor only |
| F-05 | P1 | G5 | indexing | Persist HNSW parameters/artifact | scale decision | load-time validation + exact Recall@K report | Retrieval engineer | Partial | 1k synthetic run |
| F-06 | P1 | G7 | UI | Browser E2E for feedback/shot/runtime surfaces | running server | no console errors; flow recorded | UI/QA engineer | Done locally | Playwright: search/detail/feedback/preview HTTP 200, console clean |
| BTC-01 | P0 | G1 | data | First 6-hour BTC audit | dataset release | hashes/schema/corruption/audio/features report | Data engineer | Triggered by BTC | None |
| BTC-02 | P0 | G6 | benchmark | Freeze 20–50 legal queries/qrels and slices | BTC-01 | versioned query/qrels hashes + review | Evaluation engineer | Triggered by BTC | None |
| BTC-03 | P0 | G7 | submission | Official dry-run adapter | rules/schema | validated payload; no upload | Backend engineer | Triggered by BTC | None |
| C-01 | P2 | Later | fusion | learned/adaptive fusion | frozen multimodal baseline | paired gain without slice/latency regression | Retrieval ML engineer | Deferred | None |
| C-02 | P2 | Later | rerank | VLM reranker | GPU + top-N baseline | timeout/fallback + paired benchmark | Multimodal engineer | Deferred | None |
| C-03 | P2 | Later | KISC | conversational agent | stable feedback/search API | scripted session eval + rollback | Agent engineer | Deferred | None |
