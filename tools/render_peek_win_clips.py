#!/usr/bin/env python3
"""Render side-by-side clips where learned PEEK recovery fixes a ByteTrack miss."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box.astype(np.float32)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def read_gt(path: Path) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            frame = int(float(parts[0]))
            mark = int(float(parts[6])) if len(parts) > 6 else 1
            label = int(float(parts[7])) if len(parts) > 7 else 1
            if mark == 0 or label != 1:
                continue
            visibility = float(parts[8]) if len(parts) > 8 else 1.0
            frames[frame].append(
                {
                    "id": int(float(parts[1])),
                    "xyxy": xywh_to_xyxy(np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])),
                    "visibility": visibility,
                }
            )
    return frames


def read_tracks(path: Path) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame_index"]) + 1
            for item in row.get("tracks", []):
                box = np.array(item["xyxy"], dtype=np.float32)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                frames[frame].append({**item, "xyxy": box})
    return frames


def read_image_paths(cache_jsonl: Path) -> dict[int, Path]:
    paths = {}
    repo = Path(__file__).resolve().parents[1]
    with cache_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame_index"]) + 1
            image = Path(row["image"])
            paths[frame] = image if image.is_absolute() else (repo / image).resolve()
    return paths


def match_gt(gt_items: list[dict], pred_items: list[dict], threshold: float) -> dict[int, tuple[int, float]]:
    if not gt_items or not pred_items:
        return {}
    costs = np.ones((len(gt_items), len(pred_items)), dtype=np.float32)
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            costs[gi, pi] = 1.0 - iou_xyxy(gt["xyxy"], pred["xyxy"])
    rows, cols = linear_sum_assignment(costs)
    matches = {}
    for gi, pi in zip(rows, cols):
        overlap = 1.0 - float(costs[gi, pi])
        if overlap >= threshold:
            matches[int(gi)] = (int(pi), overlap)
    return matches


def find_wins(gt, base, learned, iou_threshold: float, min_gap: int) -> list[dict]:
    events = []
    last_for_gt: dict[tuple[str, int], int] = {}
    for frame in sorted(gt):
        gt_items = gt.get(frame, [])
        base_matches = match_gt(gt_items, base.get(frame, []), iou_threshold)
        learned_items = learned.get(frame, [])
        peek_items = [item for item in learned_items if item.get("source") == "peek_learned"]
        peek_matches = match_gt(gt_items, peek_items, iou_threshold)
        for gi, (pi, overlap) in peek_matches.items():
            if gi in base_matches:
                continue
            gt_id = int(gt_items[gi]["id"])
            key = ("gt", gt_id)
            if frame - last_for_gt.get(key, -10_000) < min_gap:
                continue
            last_for_gt[key] = frame
            events.append(
                {
                    "frame": frame,
                    "gt_id": gt_id,
                    "visibility": float(gt_items[gi].get("visibility", 1.0)),
                    "iou": overlap,
                    "gt_box": gt_items[gi]["xyxy"],
                    "peek_box": peek_items[pi]["xyxy"],
                    "peek_track_id": int(peek_items[pi]["id"]),
                    "peek_score": float(peek_items[pi].get("score", 0.0)),
                    "peek_module": peek_items[pi].get("module"),
                }
            )
    events.sort(key=lambda item: (item["visibility"], item["iou"], item["peek_score"]), reverse=True)
    return events


def draw_box(img, box, color, label, thickness=2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(th + 6, y1 - 6)
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 4), color, -1)
        cv2.putText(img, label, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_arrow_callout(img, box, color, label):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cx = max(0, min(w - 1, int(round((x1 + x2) / 2))))
    cy = max(0, min(h - 1, int(round((y1 + y2) / 2))))
    if cx < w * 0.55:
        start = (min(w - 40, cx + 260), max(85, cy - 180))
        text_anchor = (max(20, start[0] - 190), max(82, start[1] - 18))
    else:
        start = (max(40, cx - 260), max(85, cy - 180))
        text_anchor = (min(w - 300, start[0] + 15), max(82, start[1] - 18))
    cv2.arrowedLine(img, start, (cx, cy), color, 10, cv2.LINE_AA, tipLength=0.18)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    tx, ty = text_anchor
    cv2.rectangle(img, (tx - 10, ty - th - 12), (tx + tw + 12, ty + 10), (20, 25, 35), -1)
    cv2.rectangle(img, (tx - 10, ty - th - 12), (tx + tw + 12, ty + 10), color, 3)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)


def annotate_panel(img, title, subtitle):
    cv2.rectangle(img, (0, 0), (img.shape[1], 58), (20, 25, 35), -1)
    cv2.putText(img, title, (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (18, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 220, 235), 1, cv2.LINE_AA)


def render_clip(event, image_paths, gt, base, learned, output_path: Path, window: int, fps: float, scale: float):
    frames = [frame for frame in range(event["frame"] - window, event["frame"] + window + 1) if frame in image_paths]
    if not frames:
        return False
    first = cv2.imread(str(image_paths[frames[0]]))
    if first is None:
        return False
    panel_w = int(first.shape[1] * scale)
    panel_h = int(first.shape[0] * scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 2, panel_h))
    for frame in frames:
        img = cv2.imread(str(image_paths[frame]))
        if img is None:
            continue
        left = img.copy()
        right = img.copy()
        for item in base.get(frame, []):
            draw_box(left, item["xyxy"], (255, 170, 40), f"BT {item['id']}", 1)
        for item in learned.get(frame, []):
            color = (255, 170, 40) if item.get("source") != "peek_learned" else (255, 0, 255)
            label = f"PEEK {item['id']}" if item.get("source") == "peek_learned" else f"BT {item['id']}"
            draw_box(right, item["xyxy"], color, label, 1 if item.get("source") != "peek_learned" else 3)

        target_gt = next((item for item in gt.get(frame, []) if int(item["id"]) == event["gt_id"]), None)
        if target_gt is not None:
            draw_box(left, target_gt["xyxy"], (0, 0, 255), "missed GT", 4)
            draw_box(right, target_gt["xyxy"], (0, 255, 255), "GT", 2)
            if frame == event["frame"]:
                draw_box(right, event["peek_box"], (255, 0, 255), f"corrected by PEEK {event['peek_track_id']}", 4)
                draw_arrow_callout(left, target_gt["xyxy"], (0, 0, 255), "ByteTrack miss")
                draw_arrow_callout(right, event["peek_box"], (255, 0, 255), "PEEK recovery")

        annotate_panel(left, "ByteTrack only", f"frame {frame}: missed GT id {event['gt_id']}")
        annotate_panel(right, "YOLO26 + ByteTrack + PEEK", f"frame {frame}: learned recovery highlighted in magenta")
        if scale != 1.0:
            left = cv2.resize(left, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        writer.write(np.concatenate([left, right], axis=1))
    writer.release()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--baseline-jsonl", required=True, type=Path)
    parser.add_argument("--learned-jsonl", required=True, type=Path)
    parser.add_argument("--cache-jsonl", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-gap", type=int, default=20)
    args = parser.parse_args()

    gt = read_gt(args.gt)
    baseline = read_tracks(args.baseline_jsonl)
    learned = read_tracks(args.learned_jsonl)
    image_paths = read_image_paths(args.cache_jsonl)
    events = find_wins(gt, baseline, learned, args.iou, args.min_gap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for index, event in enumerate(events[: args.max_clips], start=1):
        out = args.output_dir / f"{args.sequence}_win_{index:02d}_frame_{event['frame']:06d}_gt_{event['gt_id']}.mp4"
        if render_clip(event, image_paths, gt, baseline, learned, out, args.window, args.fps, args.scale):
            entry = {
                "clip": str(out),
                "sequence": args.sequence,
                "frame": int(event["frame"]),
                "gt_id": int(event["gt_id"]),
                "peek_track_id": int(event["peek_track_id"]),
                "peek_score": float(event["peek_score"]),
                "peek_module": event["peek_module"],
                "iou": float(event["iou"]),
                "visibility": float(event["visibility"]),
            }
            report.append(entry)
            print(json.dumps(entry))
    (args.output_dir / f"{args.sequence}_win_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"events={len(events)} clips={len(report)}")


if __name__ == "__main__":
    main()
