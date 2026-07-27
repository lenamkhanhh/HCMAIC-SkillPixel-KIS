# 02 — Data contract và artifact contract

## Frame mapping chuẩn

Mapping mới gồm:

```text
video_id,n,pts_time,fps,frame_idx,shot_id,frame_id,shot_start,shot_end,
width,height,timestamp_source,ingestion_provider,sampling_policy
```

Mapping cũ chỉ có `video_id,n,pts_time,fps,frame_idx` vẫn đọc được. Khi đó:

- `timestamp_source = legacy_mapping`;
- các field về shot là `null`;
- không tự coi dữ liệu legacy là exact timestamp.

Các giá trị `timestamp_source` hiện dùng:

- `exact_pts`
- `best_effort_pts`
- `cfr_fallback`
- `legacy_mapping`

## FeatureRecord

Mỗi modality record phải có:

- `video_id` hoặc entity ID;
- thời gian bắt đầu/kết thúc;
- modality: `visual`, `ocr`, `asr`, `caption`, `metadata`, `segment`;
- provider và revision;
- text hoặc artifact reference;
- confidence nếu có;
- SHA-256 content hash và metadata.

Feature JSONL phải ở canonical format và fail-closed khi hash không khớp.

## Artifact provenance

Index manifest phải ghi:

- normalized config và config hash;
- dataset manifest hash;
- embedding provider/version/dimension/normalization;
- index type và parameters;
- device, batch size, seed;
- code commit SHA nếu lấy được;
- creation timestamp.

Benchmark `run_manifest.json` ghi thêm query/qrels hash, hardware,
warmup/repeat policy và evidence level.

## Validation

Validator phải reject:

- timestamp âm hoặc vượt duration;
- shot range không hợp lệ;
- ID trùng;
- image thiếu hoặc không đọc được;
- path thoát khỏi dataset root;
- artifact/hash drift.

Optional modality được phép fail hoặc disable mà không phá visual path.
