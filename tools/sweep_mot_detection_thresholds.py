#!/usr/bin/env python3
"""Sweep confidence thresholds for MOT-style YOLO detection files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = bx1 + b[:, 2][None, :], by1 + b[:, 3][None, :]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = a[:, 2:3] * a[:, 3:4]
    area_b = b[:, 2][None, :] * b[:, 3][None, :]
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def read_gt(path: Path) -> dict[int, list[np.ndarray]]:
    frames: dict[int, list[np.ndarray]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            mark = int(float(parts[6])) if len(parts) > 6 else 1
            label = int(float(parts[7])) if len(parts) > 7 else 1
            if mark == 0 or label != 1:
                continue
            frames[int(float(parts[0]))].append(np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32))
    return frames


def read_det(path: Path) -> list[tuple[int, float, np.ndarray]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            rows.append(
                (
                    int(float(parts[0])),
                    float(parts[6]),
                    np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32),
                )
            )
    return rows


def score_one(
    gt: dict[int, list[np.ndarray]],
    detections_by_frame: dict[int, list[tuple[float, np.ndarray]]],
    threshold: float,
    iou_thr: float,
) -> dict:
    pred: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame, items in detections_by_frame.items():
        for conf, box in items:
            if conf >= threshold:
                pred[frame].append(box)
    tp = fp = fn = 0
    for frame in sorted(set(gt) | set(pred)):
        gts = gt.get(frame, [])
        preds = pred.get(frame, [])
        if not gts:
            fp += len(preds)
            continue
        if not preds:
            fn += len(gts)
            continue
        costs = 1.0 - iou_matrix_xywh(np.stack(gts, axis=0), np.stack(preds, axis=0))
        rows, cols = linear_sum_assignment(costs)
        matched_g: set[int] = set()
        matched_p: set[int] = set()
        for gi, pi in zip(rows, cols):
            if 1.0 - float(costs[gi, pi]) >= iou_thr:
                matched_g.add(int(gi))
                matched_p.add(int(pi))
                tp += 1
        fp += len(preds) - len(matched_p)
        fn += len(gts) - len(matched_g)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-root", required=True, type=Path, help="Root containing train/<seq>/gt/gt.txt.")
    parser.add_argument("--det-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    rows = []
    totals = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in args.thresholds}
    for gt_file in sorted(args.mot_root.glob("train/*/gt/gt.txt")):
        seq = gt_file.parents[1].name
        det_file = args.det_dir / f"{seq}.txt"
        if not det_file.exists():
            continue
        gt = read_gt(gt_file)
        dets_by_frame: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)
        for frame, conf, box in read_det(det_file):
            dets_by_frame[frame].append((conf, box))
        for thr in args.thresholds:
            row = score_one(gt, dets_by_frame, thr, args.iou)
            row.update({"sequence": seq, "threshold": thr})
            rows.append(row)
            for key in ("tp", "fp", "fn"):
                totals[thr][key] += row[key]

    summary = []
    for thr, total in totals.items():
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary.append({"threshold": thr, **total, "precision": precision, "recall": recall, "f1": f1})
    summary.sort(key=lambda row: row["f1"], reverse=True)

    payload = {"det_dir": str(args.det_dir), "mot_root": str(args.mot_root), "iou": args.iou, "summary": summary, "per_sequence": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"][:8], indent=2))


if __name__ == "__main__":
    main()
