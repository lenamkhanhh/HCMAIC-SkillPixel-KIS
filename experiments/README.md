# Experiment ledger

Create one JSONL row per controlled experiment:

```json
{"experiment_id":"EXP-001","hypothesis":"...","change":"...","dataset_hash":"...","config_hash":"...","metric_movement":{},"slice_movement":{},"latency_movement":{},"decision":"keep|reject|rerun","regression_added":"test path or none","next_experiment":"..."}
```

Change one primary factor at a time. Reference immutable run manifests instead
of copying unverified portal numbers into the ledger.
