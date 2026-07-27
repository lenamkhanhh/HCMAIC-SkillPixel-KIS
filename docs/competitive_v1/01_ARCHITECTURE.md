# 01 — Kiến trúc

## Offline

```mermaid
flowchart LR
  A["Raw video / supplied keyframes"] --> B["Dataset adapter"]
  B --> C["Ingestion + timestamp provenance"]
  C --> D["Shot detector + sampler"]
  D --> E["Visual / OCR / ASR / caption records"]
  E --> F["Provider registry"]
  F --> G["Per-modality indexes"]
  G --> H["Versioned manifests + hashes"]
```

## Online

```mermaid
flowchart LR
  Q["Text / conversational query"] --> R["Channel retrievers"]
  R --> V["Visual"]
  R --> O["OCR"]
  R --> A["ASR"]
  R --> C["Caption/metadata"]
  V --> F["RRF or weighted late fusion"]
  O --> F
  A --> F
  C --> F
  F --> T["Temporal expansion"]
  T --> K["Bounded reranker"]
  K --> U["API / operator UI / submission preview"]
```

## Interactive and future agent

```mermaid
sequenceDiagram
  participant User
  participant UI
  participant API
  participant Retrieval
  User->>UI: query
  UI->>API: POST /search
  API->>Retrieval: canonical request
  Retrieval-->>API: explained ranked results
  API-->>UI: frames + scores + timestamps
  User->>UI: relevant / not relevant
  UI->>API: POST /feedback
  Note over API: Future KISC agent reuses the same contracts
```

## Boundaries and ownership

| Module | Responsibility | Suggested role |
|---|---|---|
| `ingestion/` | dataset, timestamps, shots | Data/ingestion engineer |
| `embedding/`, `features/` | model and modality adapters | Multimodal ML engineer |
| `indexing/`, `retrieval/` | indexes, fusion, temporal, rerank | Retrieval engineer |
| `api/`, `ui/` | operator and future-agent interface | Backend/frontend engineer |
| `benchmark/`, `evaluation/` | metrics, frozen runs, regression | Evaluation/QA engineer |

Dependencies point inward through contracts. Optional model imports are lazy;
the mandatory offline path never needs weights or network.
