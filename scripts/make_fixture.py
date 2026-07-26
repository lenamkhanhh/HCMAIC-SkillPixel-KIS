"""Regenerate the committed fixture dataset in data/sample.

Deterministic: fixed colors, sizes, and mapping. Run from system/:

    uv run python scripts/make_fixture.py

The fixture validates plumbing only — 5 mock videos, 12 keyframes, mapping
CSV, optional metadata, queries and qrels. It proves nothing about
competition retrieval quality.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"
SIZE = (64, 48)
FPS = 25.0

# video_id -> list of (n, colors). One color = solid; two = vertical split.
VIDEOS: dict[str, list[tuple[int, list[tuple[int, int, int]]]]] = {
    "L01_V001": [
        (1, [(220, 40, 40)]),                      # red
        (2, [(40, 70, 220)]),                      # blue
        (3, [(40, 180, 60)]),                      # green
    ],
    "L01_V002": [
        (1, [(230, 220, 50)]),                     # yellow
        (2, [(60, 200, 210)]),                     # cyan
        (3, [(240, 240, 240)]),                    # white
    ],
    "L01_V003": [
        (1, [(200, 60, 200)]),                     # magenta
        (2, [(15, 15, 15)]),                       # black
    ],
    "L01_V004": [
        (1, [(220, 40, 40), (240, 240, 240)]),     # red | white
        (2, [(40, 70, 220), (230, 220, 50)]),      # blue | yellow
    ],
    "L01_V005": [
        (1, [(40, 180, 60), (15, 15, 15)]),        # green | black
        (2, [(60, 200, 210), (200, 60, 200)]),     # cyan | magenta
    ],
}

MEDIA_INFO = {
    "L01_V001": {
        "title": "Fixture video one (red, blue, green)",
        "author": "hcmaic-fixture",
        "length": 60,
        "publish_date": "01/01/2026",
        "watch_url": "https://example.invalid/L01_V001",
    },
    "L01_V002": {
        "title": "Fixture video two (yellow, cyan, white)",
        "author": "hcmaic-fixture",
        "length": 60,
    },
    "L01_V004": {
        "title": "Fixture video four (split frames)",
        "author": "hcmaic-fixture",
        "length": 30,
    },
}

QUERIES = [
    {"query_id": "q1", "text": "a solid red keyframe", "task_type": "kis"},
    {"query_id": "q2", "text": "blue scene", "task_type": "kis"},
    {"query_id": "q3", "text": "yellow frame", "task_type": "kis"},
    {"query_id": "q4", "text": "black night shot", "task_type": "kis"},
    {"query_id": "q5", "text": "cyan and magenta pattern", "task_type": "kis"},
    {"query_id": "q6", "text": "green grass field", "task_type": "kis"},
]

QRELS = [
    {"query_id": "q1", "relevant_frame_ids": ["L01_V001:001"]},
    {"query_id": "q2", "relevant_frame_ids": ["L01_V001:002"]},
    {"query_id": "q3", "relevant_frame_ids": ["L01_V002:001"]},
    {"query_id": "q4", "relevant_frame_ids": ["L01_V003:002"]},
    {"query_id": "q5", "relevant_frame_ids": ["L01_V005:002"]},
    {"query_id": "q6", "relevant_frame_ids": ["L01_V001:003", "L01_V005:001"]},
]


def paint(colors: list[tuple[int, int, int]]) -> Image.Image:
    im = Image.new("RGB", SIZE, colors[0])
    if len(colors) == 2:
        half = Image.new("RGB", (SIZE[0] // 2, SIZE[1]), colors[1])
        im.paste(half, (SIZE[0] // 2, 0))
    return im


def main() -> None:
    mapping_rows = []
    for video_id, frames in VIDEOS.items():
        video_dir = ROOT / "keyframes" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        for n, colors in frames:
            paint(colors).save(video_dir / f"{n:03d}.jpg", quality=92)
            pts_time = float(n * 4 - 3)  # 1.0, 5.0, 9.0 ...
            mapping_rows.append(
                {
                    "video_id": video_id,
                    "n": n,
                    "pts_time": pts_time,
                    "fps": FPS,
                    "frame_idx": int(pts_time * FPS),
                }
            )

    with open(ROOT / "keyframe_mapping.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "n", "pts_time", "fps", "frame_idx"]
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    media_dir = ROOT / "media-info"
    media_dir.mkdir(parents=True, exist_ok=True)
    for video_id, info in MEDIA_INFO.items():
        with open(media_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    with open(ROOT / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in QUERIES:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(ROOT / "qrels.jsonl", "w", encoding="utf-8") as f:
        for q in QRELS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Fixture written to {ROOT}: {len(mapping_rows)} keyframes, "
          f"{len(VIDEOS)} videos, {len(QUERIES)} queries.")


if __name__ == "__main__":
    main()
