# 04 — Benchmark protocol

## Mục tiêu

Mỗi benchmark phải trả lời một hypothesis rõ ràng và có thể rerun. Không thay
nhiều primary factor trong cùng một experiment.

## Input bắt buộc phải freeze

- dataset manifest/hash;
- query set và qrels hash;
- config hash;
- code commit SHA;
- provider/model revision;
- preprocessing version;
- index type/parameters;
- device, batch size, seed;
- warmup và repeat policy.

## Output bắt buộc

```text
benchmark_summary.json
per_query_results.jsonl
run_manifest.json
failure_cases.md
```

## Metric tối thiểu

- Recall@1/5/10/100;
- MRR;
- latency p50/p95;
- empty result và missing qrels count;
- per-query result;
- slice metric khi có data thật;
- ANN Recall@K khi dùng approximate index.

## Quy tắc quyết định

Chỉ `keep` một thay đổi khi:

1. chạy paired với incumbent trên cùng frozen input;
2. metric tổng hoặc slice mục tiêu tăng;
3. không gây regression nghiêm trọng ở slice khác;
4. latency/memory nằm trong budget;
5. có fallback và regression test.

Nếu evidence chỉ là fixture hoặc synthetic thì ghi đúng label, không suy diễn
ra chất lượng thi.
