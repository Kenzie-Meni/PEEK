#!/usr/bin/env python3
"""Evaluate YOLO26 detections plus simple tracker variants on YOLO track labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Box:
    cls: int
    xyxy: np.ndarray
    conf: float = 1.0
    track_id: int = -1


@dataclass
class Track:
    track_id: int
    cls: int
    xyxy: np.ndarray
    conf: float
    missed: int = 0
    age: int = 1


def add_ultralytics(repo: Path) -> None:
    vendor = repo / "third_party" / "ultralytics"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def parse_label(path: Path, width: int, height: int, has_track: bool) -> list[Box]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height
        tid = int(float(parts[5])) if has_track and len(parts) >= 6 else -1
        boxes.append(Box(cls=cls, xyxy=np.array([x1, y1, x2, y2], dtype=np.float32), track_id=tid))
    return boxes


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


def center_dist_norm(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    ac = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0])
    bc = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
    return float(np.linalg.norm(ac - bc) / math.hypot(width, height))


def match_boxes(gt: list[Box], pred: list[Box], iou_thr: float = 0.5, class_aware: bool = True):
    if not gt or not pred:
        return [], set(), set()
    cost = np.ones((len(gt), len(pred)), dtype=np.float32) * 1e6
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if class_aware and g.cls != p.cls:
                continue
            ov = iou(g.xyxy, p.xyxy)
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


def run_tracker(frames: list[list[Box]], width: int, height: int, cfg: dict) -> list[list[Box]]:
    tracks: list[Track] = []
    next_id = 1
    output: list[list[Box]] = []
    for dets in frames:
        dets = [d for d in dets if d.conf >= cfg["conf"]]
        matches: list[tuple[int, int]] = []
        used_t, used_d = set(), set()
        if tracks and dets:
            cost = np.ones((len(tracks), len(dets)), dtype=np.float32) * 1e6
            for ti, t in enumerate(tracks):
                for di, d in enumerate(dets):
                    class_penalty = 0.0 if (not cfg["class_aware"] or t.cls == d.cls) else cfg["class_penalty"]
                    ov = iou(t.xyxy, d.xyxy)
                    dist = center_dist_norm(t.xyxy, d.xyxy, width, height)
                    if ov >= cfg["min_iou"] or dist <= cfg["max_center_dist"]:
                        cost[ti, di] = (1.0 - ov) + cfg["dist_weight"] * dist + class_penalty
            rows, cols = linear_sum_assignment(cost)
            for ti, di in zip(rows, cols):
                if cost[ti, di] <= cfg["max_cost"]:
                    matches.append((ti, di))
                    used_t.add(ti)
                    used_d.add(di)

        for ti, di in matches:
            d = dets[di]
            tracks[ti].xyxy = d.xyxy.copy()
            tracks[ti].cls = d.cls
            tracks[ti].conf = d.conf
            tracks[ti].missed = 0
            tracks[ti].age += 1

        for ti, t in enumerate(tracks):
            if ti not in used_t:
                t.missed += 1

        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tracks.append(Track(next_id, d.cls, d.xyxy.copy(), d.conf))
            next_id += 1

        tracks = [t for t in tracks if t.missed <= cfg["max_missed"]]
        output.append([Box(t.cls, t.xyxy.copy(), t.conf, t.track_id) for t in tracks if t.missed == 0 or cfg["emit_missed"]])
    return output


def evaluate(gt_frames: list[list[Box]], pred_frames: list[list[Box]], iou_thr: float, class_aware: bool):
    gt_total = sum(len(x) for x in gt_frames)
    pred_total = sum(len(x) for x in pred_frames)
    tp = fp = fn = 0
    cls_tp = Counter()
    cls_fp = Counter()
    cls_fn = Counter()
    last_match_by_gt: dict[int, int] = {}
    idsw = 0
    pair_counts = Counter()
    matched_rows = []
    for fi, (gts, preds) in enumerate(zip(gt_frames, pred_frames)):
        pairs, used_g, used_p = match_boxes(gts, preds, iou_thr=iou_thr, class_aware=class_aware)
        tp += len(pairs)
        fp += len(preds) - len(used_p)
        fn += len(gts) - len(used_g)
        for gi, pi, ov in pairs:
            g, p = gts[gi], preds[pi]
            cls_tp[g.cls] += 1
            pair_counts[(g.track_id, p.track_id)] += 1
            if g.track_id in last_match_by_gt and last_match_by_gt[g.track_id] != p.track_id:
                idsw += 1
            last_match_by_gt[g.track_id] = p.track_id
            matched_rows.append((fi, g.track_id, p.track_id, g.cls, p.cls, ov, p.conf))
        for gi, g in enumerate(gts):
            if gi not in used_g:
                cls_fn[g.cls] += 1
        for pi, p in enumerate(preds):
            if pi not in used_p:
                cls_fp[p.cls] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    mota = 1.0 - (fn + fp + idsw) / gt_total if gt_total else 0.0
    assigned = {}
    idtp = 0
    for (gt_id, pred_id), n in pair_counts.most_common():
        if gt_id not in assigned and pred_id not in assigned.values():
            assigned[gt_id] = pred_id
            idtp += n
    idfp = tp - idtp + fp
    idfn = tp - idtp + fn
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    return {
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
        "class_tp": dict(sorted(cls_tp.items())),
        "class_fp": dict(sorted(cls_fp.items())),
        "class_fn": dict(sorted(cls_fn.items())),
        "matched_rows": matched_rows,
        "pair_counts": {f"{g}->{p}": n for (g, p), n in pair_counts.most_common()},
    }


def collect_images(dataset: Path) -> list[Path]:
    train = dataset / "train"
    return sorted([p for p in train.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def predict(weights: Path, images: list[Path], imgsz: int, device: str) -> tuple[list[list[Box]], int, int]:
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(weights))
    frames = []
    width = height = 0
    for idx, image in enumerate(images):
        if idx == 0:
            with Image.open(image) as im:
                width, height = im.size
        result = model.predict(str(image), imgsz=imgsz, device=device, verbose=False)[0]
        boxes = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            conf = result.boxes.conf.detach().cpu().numpy()
            for c, cf, b in zip(cls, conf, xyxy):
                if c < 0 or c > 3:
                    continue
                boxes.append(Box(int(c), b.astype(np.float32), float(cf)))
        frames.append(boxes)
    return frames, width, height


def default_variants() -> dict[str, dict]:
    base = {
        "conf": 0.10,
        "min_iou": 0.15,
        "max_center_dist": 0.12,
        "dist_weight": 1.5,
        "class_aware": True,
        "class_penalty": 10.0,
        "max_cost": 1.15,
        "max_missed": 8,
        "emit_missed": False,
    }
    variants = {
        "det_highconf_iou": {**base, "conf": 0.25, "min_iou": 0.30, "max_center_dist": 0.08, "max_missed": 4},
        "det_lowconf_recall": {**base, "conf": 0.05, "min_iou": 0.10, "max_center_dist": 0.16, "max_missed": 8},
        "custom_balanced": base,
        "custom_sticky": {**base, "conf": 0.06, "min_iou": 0.05, "max_center_dist": 0.20, "max_missed": 18, "max_cost": 1.25},
        "custom_classagnostic": {**base, "conf": 0.08, "class_aware": False, "min_iou": 0.10, "max_center_dist": 0.18, "max_missed": 12},
        "custom_strict_class": {**base, "conf": 0.15, "min_iou": 0.20, "max_center_dist": 0.10, "max_missed": 6},
    }
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    add_ultralytics(repo)
    args.out.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.dataset)
    if not images:
        raise FileNotFoundError(f"No images in {args.dataset / 'train'}")

    pred_frames, width, height = predict(args.weights, images, args.imgsz, args.device)
    gt_frames = [
        parse_label(args.dataset / "train" / "labels_track" / f"{image.stem}.txt", width, height, has_track=True)
        for image in images
    ]

    summary_rows = []
    all_metrics = {}
    for name, cfg in default_variants().items():
        tracked = run_tracker(pred_frames, width, height, cfg)
        metrics = evaluate(gt_frames, tracked, iou_thr=0.5, class_aware=True)
        metrics["variant"] = name
        metrics["config"] = cfg
        all_metrics[name] = metrics
        summary_rows.append(
            {
                "variant": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mota_like": metrics["mota_like"],
                "idf1_like": metrics["idf1_like"],
                "id_switches": metrics["id_switches"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "pred": metrics["pred"],
                "gt": metrics["gt"],
            }
        )
        with (args.out / f"{name}_tracks.jsonl").open("w") as f:
            for image, boxes in zip(images, tracked):
                f.write(json.dumps({
                    "image": image.name,
                    "boxes": [
                        {
                            "cls": b.cls,
                            "conf": b.conf,
                            "track_id": b.track_id,
                            "xyxy": [round(float(x), 3) for x in b.xyxy.tolist()],
                        }
                        for b in boxes
                    ],
                }) + "\n")

    # Detector-only upper bound from raw predictions after a light confidence cut.
    detector_frames = [[b for b in frame if b.conf >= 0.05] for frame in pred_frames]
    det_metrics = evaluate(gt_frames, detector_frames, iou_thr=0.5, class_aware=True)
    det_metrics["variant"] = "detector_only_conf005_no_tracking"
    all_metrics["detector_only_conf005_no_tracking"] = det_metrics
    summary_rows.append(
        {
            "variant": "detector_only_conf005_no_tracking",
            "precision": det_metrics["precision"],
            "recall": det_metrics["recall"],
            "mota_like": det_metrics["mota_like"],
            "idf1_like": det_metrics["idf1_like"],
            "id_switches": det_metrics["id_switches"],
            "tp": det_metrics["tp"],
            "fp": det_metrics["fp"],
            "fn": det_metrics["fn"],
            "pred": det_metrics["pred"],
            "gt": det_metrics["gt"],
        }
    )

    summary_rows.sort(key=lambda r: (float(r["idf1_like"]), float(r["mota_like"]), float(r["recall"])), reverse=True)
    with (args.out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    slim = {
        k: {kk: vv for kk, vv in v.items() if kk not in {"matched_rows"}}
        for k, v in all_metrics.items()
    }
    (args.out / "metrics.json").write_text(json.dumps(slim, indent=2))
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
