# 06 — Khi BTC phát dataset

## 0–6 giờ: audit

- copy/read rules, licenses, schema and submission format;
- hash archive, list file counts/extensions, videos, keyframes, audio, metadata;
- measure durations/resolutions/FPS/VFR and corrupt/missing items;
- inspect provided CLIP/object/caption features and dimensions;
- create 20–50 hand-checked queries/qrels without looking at hidden test labels;
- choose adapter path; do not edit baseline models yet.

## 6–24 giờ: vertical baseline

Run one legal end-to-end slice: validate -> catalog -> control embedding -> exact
index -> search -> API/UI -> dry-run submission export. Freeze this commit and
its report before parallel model work.

## 24–48 giờ: model/modality matrix

Benchmark control CLIP, SigLIP2/Jina candidate, OCR, ASR and caption separately.
Use paired queries, identical data/config and measure quality slices plus
latency/resource cost. Reject channels that fail to add value.

## 48–72 giờ: freeze and assign

Freeze baseline v1, choose index/fusion, lock schemas and assign five roles:
data/ingestion, visual embeddings, text/audio modalities, retrieval/fusion and
API/UI/evaluation. Only then add competitive tricks behind flags.

## Decision paths

- Raw video: run timestamp/shot pipeline; supplied keyframes: preserve mapping.
- Audio present: test ASR; absent: disable ASR cleanly.
- Supplied features: verify IDs/dimension/normalization/license before reuse.
- Small vectors: exact index; large vectors: benchmark HNSW/IVF against exact.
- Official submission schema: implement a separate adapter and dry run; never
  upload from tests or without team-controlled authorization.
