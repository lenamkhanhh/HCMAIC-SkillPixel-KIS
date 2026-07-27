# 05 — Team handoff

## Current state

The legacy visual path remains the mandatory control. New modules provide
provider/modality/fusion/ANN/benchmark contracts, but optional real models have
not been benchmarked on BTC data.

## Add an adapter

1. Add a dataset adapter under `ingestion/` that emits canonical mapping rows.
2. Preserve original timestamps and evidence source.
3. Add malformed, legacy and path-safety tests.
4. Run validate -> catalog -> index -> search E2E.

## Add a provider or modality

1. Register model/revision and dependency checks without import-time loading.
2. Implement shared image/text preprocessing and discover dimension.
3. Emit `FeatureRecord` and hash-addressed artifacts.
4. Build only that modality index; visual must survive its failure.
5. Benchmark a paired control and label real versus fixture evidence.

## Add an index/reranker

Exact search is the oracle. Record parameters and measure Recall@K, latency and
memory. A reranker receives top-N and returns top-K; it must have timeout and
passthrough fallback before use in the operator path.

## PR expectations

- one task ID and one hypothesis;
- tests show RED then GREEN;
- no model weights/raw/private data/generated indexes in Git;
- exact commands and metrics in the PR;
- reviewer checks contracts, drift gates, latency and fallback;
- no fixture score presented as competition quality.

See `TEAM_TASK_BOARD.md` for role-based work without invented member names.
