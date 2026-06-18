#!/usr/bin/env python3
"""Summarize YOLO bbox failure modes on a labeled image split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

from PIL import Image


NAMES = {0: "antenna", 1: "body", 2: "solar", 3: "thruster"}


@dataclass
class Box:
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def yolo_label_to_box(line: str, width: int, height: int) -> Box | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    xc, yc, w, h = (float(value) for value in parts[1:5])
    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height
    return Box(cls=cls, xyxy=(x1, y1, x2, y2))


def size_bin(box: Box, image_area: float) -> str:
    frac = box.area / image_area if image_area else 0.0
    if frac < 0.005:
        return "tiny<0.5%"
    if frac < 0.02:
        return "small<2%"
    if frac < 0.08:
        return "medium<8%"
    return "large>=8%"


def load_labels(label_path: Path, width: int, height: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        box = yolo_label_to_box(line, width, height)
        if box is not None:
            boxes.append(box)
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "third_party" / "ultralytics"))
    from ultralytics import YOLO  # type: ignore

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    model = YOLO(args.weights)

    gt_by_class: Counter[int] = Counter()
    matched_by_class: Counter[int] = Counter()
    missed_by_class: Counter[int] = Counter()
    fp_by_class: Counter[int] = Counter()
    miss_size_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    gt_size_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    confusion: Counter[tuple[int, int]] = Counter()
    image_misses: Counter[str] = Counter()
    low_conf_matches: list[tuple[float, str, str]] = []

    for result in model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.7,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    ):
        image_path = Path(result.path)
        with Image.open(image_path) as image:
            width, height = image.size
        image_area = float(width * height)
        gts = load_labels(label_dir / f"{image_path.stem}.txt", width, height)
        preds = [
            Box(
                cls=int(cls),
                xyxy=tuple(float(value) for value in xyxy),
                conf=float(conf),
            )
            for xyxy, cls, conf in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            )
        ]

        for gt in gts:
            gt_by_class[gt.cls] += 1
            gt_size_by_class[gt.cls][size_bin(gt, image_area)] += 1

        used_preds: set[int] = set()
        for gt in gts:
            candidates = [
                (iou(gt, pred), idx, pred)
                for idx, pred in enumerate(preds)
                if idx not in used_preds and pred.cls == gt.cls
            ]
            best_same = max(candidates, default=(0.0, -1, None), key=lambda item: item[0])
            if best_same[0] >= args.iou and best_same[2] is not None:
                used_preds.add(best_same[1])
                matched_by_class[gt.cls] += 1
                if best_same[2].conf < 0.25:
                    low_conf_matches.append((best_same[2].conf, image_path.name, NAMES[gt.cls]))
                continue

            missed_by_class[gt.cls] += 1
            miss_size_by_class[gt.cls][size_bin(gt, image_area)] += 1
            image_misses[image_path.name] += 1

            wrong_class = [
                (iou(gt, pred), idx, pred)
                for idx, pred in enumerate(preds)
                if idx not in used_preds and pred.cls != gt.cls
            ]
            best_wrong = max(wrong_class, default=(0.0, -1, None), key=lambda item: item[0])
            if best_wrong[0] >= args.iou and best_wrong[2] is not None:
                confusion[(gt.cls, best_wrong[2].cls)] += 1

        for idx, pred in enumerate(preds):
            if idx in used_preds:
                continue
            overlaps = [iou(pred, gt) for gt in gts]
            if max(overlaps, default=0.0) < args.iou:
                fp_by_class[pred.cls] += 1

    print(f"images={len(image_paths)} conf={args.conf} match_iou={args.iou}")
    print("\nPer-class recall proxy at fixed threshold")
    for cls in sorted(NAMES):
        gt = gt_by_class[cls]
        matched = matched_by_class[cls]
        missed = missed_by_class[cls]
        recall = matched / gt if gt else 0.0
        print(
            f"{NAMES[cls]:>8}: gt={gt:4d} matched={matched:4d} missed={missed:4d} "
            f"recall={recall:.3f} fp={fp_by_class[cls]:4d}"
        )
        print(f"          gt_sizes={dict(gt_size_by_class[cls])}")
        print(f"        miss_sizes={dict(miss_size_by_class[cls])}")

    print("\nLikely class confusions, shown as GT -> prediction")
    for (gt_cls, pred_cls), count in confusion.most_common():
        print(f"{NAMES[gt_cls]} -> {NAMES[pred_cls]}: {count}")

    print("\nImages with most misses")
    for image_name, count in image_misses.most_common(args.top):
        print(f"{image_name}: {count}")

    print("\nLowest-confidence true matches")
    for conf, image_name, cls_name in sorted(low_conf_matches)[: args.top]:
        print(f"{conf:.3f} {cls_name} {image_name}")


if __name__ == "__main__":
    main()
