#!/usr/bin/env python3
"""Summarize YOLO+PEEK tracking JSONL output."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--graduate-hits", type=int, default=10)
    parser.add_argument("--video", default="")
    parser.add_argument("--compressed", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl)
    frames = 0
    peek_counts = []
    shadow_counts = []
    latencies = []
    source_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    shadow_class_counts: Counter[str] = Counter()
    peek_origin_ids: set[int] = set()
    graduated_peek_ids: set[int] = set()
    frames_with_graduated_peek = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frames += 1
            peek_counts.append(int(row.get("num_peek_detections", 0)))
            shadow_counts.append(int(row.get("num_shadow_yolo_detections", 0)))
            latencies.append(float(row.get("latency_ms", 0.0)))
            frame_has_graduated = False
            for det in row.get("shadow_yolo_detections", []):
                shadow_class_counts[str(det.get("class_name"))] += 1
            for track in row.get("tracks", []):
                source = str(track.get("source"))
                origin = str(track.get("origin", source))
                source_counts[source] += 1
                origin_counts[origin] += 1
                class_counts[str(track.get("class_name"))] += 1
                if origin == "peek":
                    track_id = int(track.get("id"))
                    peek_origin_ids.add(track_id)
                    if int(track.get("hits", 0)) >= args.graduate_hits:
                        graduated_peek_ids.add(track_id)
                        frame_has_graduated = True
            if frame_has_graduated:
                frames_with_graduated_peek += 1

    stats = {
        "name": args.name,
        "frames": frames,
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "peek_proposals_avg": sum(peek_counts) / len(peek_counts) if peek_counts else 0.0,
        "peek_proposals_min": min(peek_counts) if peek_counts else 0,
        "peek_proposals_max": max(peek_counts) if peek_counts else 0,
        "shadow_yolo_avg": sum(shadow_counts) / len(shadow_counts) if shadow_counts else 0.0,
        "shadow_yolo_min": min(shadow_counts) if shadow_counts else 0,
        "shadow_yolo_max": max(shadow_counts) if shadow_counts else 0,
        "source_counts": dict(source_counts),
        "origin_counts": dict(origin_counts),
        "class_counts": dict(class_counts),
        "shadow_class_counts": dict(shadow_class_counts),
        "peek_origin_ids": len(peek_origin_ids),
        "graduated_peek_ids": len(graduated_peek_ids),
        "frames_with_graduated_peek": frames_with_graduated_peek,
        "jsonl": str(path),
        "video": args.video,
        "compressed": args.compressed,
    }
    Path(args.output).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
