# Nhật ký experiment

Mỗi controlled experiment ghi một dòng JSONL:

```json
{"experiment_id":"EXP-001","hypothesis":"...","change":"...","dataset_hash":"...","config_hash":"...","metric_movement":{},"slice_movement":{},"latency_movement":{},"decision":"keep|reject|rerun","regression_added":"test path hoặc none","next_experiment":"..."}
```

Mỗi experiment chỉ thay một yếu tố chính. Luôn tham chiếu `run_manifest` bất
biến; không chép điểm portal chưa xác minh vào nhật ký.
