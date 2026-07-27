# 01 — Kiến trúc hệ thống

## Offline pipeline

```mermaid
flowchart LR
  A["Raw video / keyframe có sẵn"] --> B["Dataset adapter"]
  B --> C["Ingestion + timestamp provenance"]
  C --> D["Shot detector + sampler"]
  D --> E["Visual / OCR / ASR / caption record"]
  E --> F["Provider registry"]
  F --> G["Per-modality index"]
  G --> H["Manifest có version và hash"]
```

## Online pipeline mục tiêu

```mermaid
flowchart LR
  Q["Text / conversational query"] --> R["Channel retriever"]
  R --> V["Visual"]
  R --> O["OCR"]
  R --> A["ASR"]
  R --> C["Caption / metadata"]
  V --> F["RRF hoặc weighted late fusion"]
  O --> F
  A --> F
  C --> F
  F --> T["Temporal expansion"]
  T --> K["Bounded reranker"]
  K --> U["API / operator UI / submission preview"]
```

## Trạng thái kết nối thực tế

`RetrievalService` hiện embed query và search trực tiếp trên một visual index.
`RetrievalOrchestrator`, fusion, temporal expansion và reranker đã có module và
test, nhưng chưa được nối hoàn toàn vào service đang serve API/UI.

## Ranh giới module

| Module | Trách nhiệm | Role phù hợp |
|---|---|---|
| `ingestion/` | dataset, timestamp, shot | Data engineer |
| `embedding/`, `features/` | model và modality adapter | Multimodal ML engineer |
| `indexing/`, `retrieval/` | index, fusion, temporal, rerank | Retrieval engineer |
| `api/`, `ui/` | giao diện operator và future agent | Backend/frontend engineer |
| `benchmark/`, `evaluation/` | metric, frozen run, regression | Evaluation/QA engineer |

Optional model phải lazy-load. Đường chạy offline bắt buộc không được phụ thuộc
weights hoặc network.
