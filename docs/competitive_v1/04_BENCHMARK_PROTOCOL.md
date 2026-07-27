# 04 — Benchmark protocol

## Freeze before running

Freeze dataset manifest, queries, qrels, YAML config, seed, provider/revision,
index/fusion/reranker settings, code commit, warmups and repeats. A run is not
comparable if one of these changes without a new experiment ID.

## Metrics

- Recall@1/5/10/100 and Mean Reciprocal Rank (MRR);
- timestamp error when temporal ground truth exists;
- p50/p95 latency, indexing time and invalid/missing results;
- exact-versus-Approximate Nearest Neighbor (ANN) Recall@K;
- query slices: visual, OCR, ASR, action/temporal, Vietnamese and mixed.

Current fixture has no trusted slice labels or temporal error ground truth, so
those fields are explicitly unavailable/unsliced.

## Repeats and ablations

- warm cache before timed repeats;
- at least three repeats for local comparisons;
- change one factor per experiment;
- compare each ANN index to Exact NumPy/FAISS FlatIP;
- report quality, latency and slice regressions together;
- never tune repeatedly on a held-out final test.

## Evidence and leakage

Proxy fixture results validate plumbing only. Synthetic vector results validate
index engineering only. Neither is a BTC score. Do not use private test labels,
portal feedback or future information as query-time features.

Required reports:

```text
benchmark_summary.json
per_query_results.jsonl
run_manifest.json
failure_cases.md
```
