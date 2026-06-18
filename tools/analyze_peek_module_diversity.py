#!/usr/bin/env python3
"""Analyze PEEK proposal diversity across modules from tracker JSONL."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * q / 100)
    return values[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlap-iou", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines()]
    by_module: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    module_pairs: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    frame_module_sets: Counter[str] = Counter()

    for row in rows:
        frame = int(row["frame_index"])
        detections = [d for d in row.get("peek_detections", []) if d.get("module") is not None]
        modules_in_frame = sorted({str(d["module"]) for d in detections})
        for module in modules_in_frame:
            frame_module_sets[module] += 1
        for det in detections:
            module = str(det["module"])
            by_module[module].append((frame, det))
            total_counts[module] += 1

        for i, det in enumerate(detections):
            module = str(det["module"])
            overlaps_other = False
            for j, other in enumerate(detections):
                if i == j or str(other["module"]) == module:
                    continue
                if iou(det["xyxy"], other["xyxy"]) >= args.overlap_iou:
                    overlaps_other = True
                    pair = "-".join(sorted([module, str(other["module"])]))
                    module_pairs[pair] += 1
            if not overlaps_other:
                unique_counts[module] += 1

    summary = {}
    for module, items in sorted(by_module.items(), key=lambda kv: int(kv[0])):
        xs = []
        ys = []
        ws = []
        hs = []
        scores = []
        classes = Counter()
        frames = []
        for frame, det in items:
            x1, y1, x2, y2 = det["xyxy"]
            xs.append((x1 + x2) / 2.0)
            ys.append((y1 + y2) / 2.0)
            ws.append(x2 - x1)
            hs.append(y2 - y1)
            scores.append(float(det["score"]))
            classes[str(det.get("class_name"))] += 1
            frames.append(frame)
        total = total_counts[module]
        summary[module] = {
            "proposals": total,
            "frames_present": frame_module_sets[module],
            "proposals_per_present_frame": total / frame_module_sets[module] if frame_module_sets[module] else 0.0,
            "unique_fraction_vs_other_modules": unique_counts[module] / total if total else 0.0,
            "unique_proposals": unique_counts[module],
            "score_mean": statistics.mean(scores) if scores else 0.0,
            "cx_mean": statistics.mean(xs) if xs else 0.0,
            "cy_mean": statistics.mean(ys) if ys else 0.0,
            "cx_p10_p90": [pct(xs, 10), pct(xs, 90)],
            "cy_p10_p90": [pct(ys, 10), pct(ys, 90)],
            "w_mean": statistics.mean(ws) if ws else 0.0,
            "h_mean": statistics.mean(hs) if hs else 0.0,
            "w_p10_p90": [pct(ws, 10), pct(ws, 90)],
            "h_p10_p90": [pct(hs, 10), pct(hs, 90)],
            "classes": dict(classes.most_common()),
            "frame_range": [min(frames), max(frames)] if frames else [None, None],
        }

    out = {
        "jsonl": args.jsonl,
        "overlap_iou": args.overlap_iou,
        "modules": summary,
        "pairwise_overlap_events": dict(module_pairs.most_common()),
    }
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
