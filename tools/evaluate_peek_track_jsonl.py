#!/usr/bin/env python3
"""Evaluate track_yolo26_peek.py JSONL against YOLO tracking sidecar labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def read_gt(path: Path, width: int, height: int) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        rows.append(
            {
                "cls": cls,
                "track_id": int(float(parts[5])),
                "xyxy": np.array(
                    [
                        (xc - w / 2) * width,
                        (yc - h / 2) * height,
                        (xc + w / 2) * width,
                        (yc + h / 2) * height,
                    ],
                    dtype=np.float32,
                ),
            }
        )
    return rows


def read_pred(row: dict) -> list[dict]:
    preds = []
    for track in row.get("tracks", []):
        cls = track.get("class_id")
        if cls is None:
            continue
        preds.append(
            {
                "cls": int(cls),
                "track_id": int(track["id"]),
                "xyxy": np.array(track["xyxy"], dtype=np.float32),
                "source": str(track.get("source")),
                "origin": str(track.get("origin", track.get("source"))),
                "score": float(track.get("score", 0.0)),
            }
        )
    return preds


def match_frame(gt: list[dict], pred: list[dict], iou_thr: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    if not gt or not pred:
        return [], set(), set()
    cost = np.full((len(gt), len(pred)), 1e6, dtype=np.float32)
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["cls"] != p["cls"]:
                continue
            ov = iou(g["xyxy"], p["xyxy"])
            if ov >= iou_thr:
                cost[gi, pi] = 1.0 - ov
    rows, cols = linear_sum_assignment(cost)
    pairs = []
    used_g, used_p = set(), set()
    for gi, pi in zip(rows, cols):
        if cost[gi, pi] < 1e6:
            pairs.append((gi, pi, 1.0 - float(cost[gi, pi])))
            used_g.add(gi)
            used_p.add(pi)
    return pairs, used_g, used_p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()

    images = sorted(p for p in (args.dataset / "train").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(args.dataset / "train")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise FileNotFoundError(images[0])
    height, width = first.shape[:2]
    pred_rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]

    tp = fp = fn = idsw = 0
    class_tp: Counter[int] = Counter()
    class_fp: Counter[int] = Counter()
    class_fn: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    last_match_by_gt: dict[int, int] = {}
    gt_total = pred_total = 0

    for image, row in zip(images, pred_rows):
        gt = read_gt(args.dataset / "train" / "labels_track" / f"{image.stem}.txt", width, height)
        pred = read_pred(row)
        gt_total += len(gt)
        pred_total += len(pred)
        for p in pred:
            source_counts[p["source"]] += 1
            origin_counts[p["origin"]] += 1
        pairs, used_g, used_p = match_frame(gt, pred, args.iou_thr)
        tp += len(pairs)
        fp += len(pred) - len(used_p)
        fn += len(gt) - len(used_g)
        for gi, pi, _ in pairs:
            g, p = gt[gi], pred[pi]
            class_tp[g["cls"]] += 1
            pair_counts[(g["track_id"], p["track_id"])] += 1
            if g["track_id"] in last_match_by_gt and last_match_by_gt[g["track_id"]] != p["track_id"]:
                idsw += 1
            last_match_by_gt[g["track_id"]] = p["track_id"]
        for gi, g in enumerate(gt):
            if gi not in used_g:
                class_fn[g["cls"]] += 1
        for pi, p in enumerate(pred):
            if pi not in used_p:
                class_fp[p["cls"]] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    mota = 1.0 - (fn + fp + idsw) / gt_total if gt_total else 0.0
    assigned: dict[int, int] = {}
    idtp = 0
    for (gt_id, pred_id), n in pair_counts.most_common():
        if gt_id not in assigned and pred_id not in assigned.values():
            assigned[gt_id] = pred_id
            idtp += n
    idfp = tp - idtp + fp
    idfn = tp - idtp + fn
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0

    metrics = {
        "variant": args.name,
        "frames": len(pred_rows),
        "gt": gt_total,
        "pred": pred_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "mota_like": mota,
        "idf1_like": idf1,
        "id_switches": idsw,
        "class_tp": dict(sorted(class_tp.items())),
        "class_fp": dict(sorted(class_fp.items())),
        "class_fn": dict(sorted(class_fn.items())),
        "source_counts": dict(source_counts),
        "origin_counts": dict(origin_counts),
        "pair_counts": {f"{g}->{p}": n for (g, p), n in pair_counts.most_common()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        keys = ["variant", "precision", "recall", "mota_like", "idf1_like", "id_switches", "tp", "fp", "fn", "pred", "gt", "frames"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({k: metrics[k] for k in keys})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
