# 02 — Data and artifact contract

## Canonical frame mapping

New generated mappings include:

```text
video_id,n,pts_time,fps,frame_idx,shot_id,frame_id,shot_start,shot_end,
width,height,timestamp_source,ingestion_provider,sampling_policy
```

Legacy mappings with `video_id,n,pts_time,fps,frame_idx` remain readable.
Their timestamp source becomes `legacy_mapping`; missing shot fields stay
`null`. `pts_time` is seconds from the source mapping/decoder. It is never
derived from `frame_idx/fps` unless the decoder explicitly forces a
`cfr_fallback`, which is recorded.

Allowed `timestamp_source` values used by current ingestion are
`exact_pts`, `best_effort_pts`, `cfr_fallback`, and `legacy_mapping`.

## FeatureRecord

Every modality record carries:

- video/entity ID and start/end milliseconds;
- modality (`visual`, `ocr`, `asr`, `caption`, `metadata`, `segment`);
- provider and revision;
- text or artifact reference;
- optional confidence;
- SHA-256 content hash and metadata.

Feature JSONL is canonical and fails closed on a mismatched file hash.

## Artifact provenance

New index manifests include normalized config, SHA-256 config hash, dataset
manifest hash, embedding provider/version/dimension/normalization, index type,
device/batch/seed, code SHA when available and creation timestamp.

Benchmark `run_manifest.json` additionally records query/qrels hashes, hardware,
warmup/repeat policy and evidence level.

## Validation

Validation rejects malformed timing, negative/out-of-duration timestamps,
invalid shot ranges, duplicate IDs, missing/unreadable images, path escape and
artifact drift. Optional modalities may be disabled without invalidating the
visual path.
