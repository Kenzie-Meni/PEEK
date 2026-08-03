#!/usr/bin/env python3
"""Create YOLO bbox and tracking sidecar labels from the satellite COCO export."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CLASS_MAP = {
    "antenna": {"yolo4": 0, "legacy5": 0, "track_id": 10},
    "body": {"yolo4": 1, "legacy5": 1, "track_id": 20},
    "solar1": {"yolo4": 2, "legacy5": 2, "track_id": 30},
    "solar2": {"yolo4": 2, "legacy5": 3, "track_id": 31},
    "thruster": {"yolo4": 3, "legacy5": 4, "track_id": 40},
}


def norm_name(name: str) -> str:
    return name.strip().lower()


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    x1 = max(0.0, min(float(width), float(x)))
    y1 = max(0.0, min(float(height), float(y)))
    x2 = max(0.0, min(float(width), float(x) + float(w)))
    y2 = max(0.0, min(float(height), float(y) + float(h)))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None
    xc = (x1 + x2) / 2.0 / width
    yc = (y1 + y2) / 2.0 / height
    return xc, yc, bw / width, bh / height


def fmt_box(cls: int, box: tuple[float, float, float, float], track_id: int | None = None) -> str:
    parts = [str(cls), *(f"{v:.8f}" for v in box)]
    if track_id is not None:
        parts.append(str(track_id))
    return " ".join(parts)


def write_yaml(path: Path, dataset_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: train",
                "val: train",
                "test: train",
                "names:",
                "  0: antenna",
                "  1: body",
                "  2: solar",
                "  3: thruster",
                "",
                "# Tracking sidecar labels are in train/labels_track/ and use: cls xc yc w h track_id",
                "# This file is for tracker/eval bookkeeping; the Roboflow export stores images directly in train/.",
                "track_label_dir: train/labels_track",
                "track_format: cls xc yc w h track_id",
                "track_ids:",
                "  10: antenna",
                "  20: body",
                "  30: solar1",
                "  31: solar2",
                "  40: thruster",
                "",
                "# Legacy 5-class sidecars are also written to train/labels_track_5cls/",
                "# with solar1 and solar2 kept as separate classes.",
                "legacy_5cls_track_label_dir: train/labels_track_5cls",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root of the Roboflow COCO dataset with train/_annotations.coco.json",
    )
    args = parser.parse_args()

    root = args.dataset_root
    ann_path = root / "train" / "_annotations.coco.json"
    data = json.loads(ann_path.read_text())

    categories = {cat["id"]: norm_name(cat["name"]) for cat in data["categories"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    labels_dir = root / "train" / "labels"
    track_dir = root / "train" / "labels_track"
    legacy_dir = root / "train" / "labels_track_5cls"
    labels_dir.mkdir(exist_ok=True)
    track_dir.mkdir(exist_ok=True)
    legacy_dir.mkdir(exist_ok=True)

    yolo_counts: Counter[int] = Counter()
    track_counts: Counter[int] = Counter()
    skipped_categories: Counter[str] = Counter()
    skipped_boxes = 0
    total_rows = 0

    for image in data["images"]:
        width = int(image["width"])
        height = int(image["height"])
        stem = Path(image["file_name"]).stem
        yolo_rows: list[str] = []
        track_rows: list[str] = []
        legacy_rows: list[str] = []

        for ann in anns_by_image.get(image["id"], []):
            category = categories.get(ann["category_id"], "")
            mapping = CLASS_MAP.get(category)
            if mapping is None:
                skipped_categories[category or str(ann["category_id"])] += 1
                continue
            box = clamp_bbox(ann["bbox"], width, height)
            if box is None:
                skipped_boxes += 1
                continue
            yolo_cls = mapping["yolo4"]
            legacy_cls = mapping["legacy5"]
            track_id = mapping["track_id"]
            yolo_rows.append(fmt_box(yolo_cls, box))
            track_rows.append(fmt_box(yolo_cls, box, track_id))
            legacy_rows.append(fmt_box(legacy_cls, box, track_id))
            yolo_counts[yolo_cls] += 1
            track_counts[track_id] += 1
            total_rows += 1

        (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_rows) + ("\n" if yolo_rows else ""))
        (track_dir / f"{stem}.txt").write_text("\n".join(track_rows) + ("\n" if track_rows else ""))
        (legacy_dir / f"{stem}.txt").write_text("\n".join(legacy_rows) + ("\n" if legacy_rows else ""))

    write_yaml(root / "data_track_yolo26.yaml", root)

    print(f"images: {len(data['images'])}")
    print(f"tracking rows: {total_rows}")
    print(f"yolo4 class counts: {dict(sorted(yolo_counts.items()))}")
    print(f"track id counts: {dict(sorted(track_counts.items()))}")
    print(f"skipped categories: {dict(skipped_categories)}")
    print(f"skipped boxes: {skipped_boxes}")
    print(f"wrote: {labels_dir}")
    print(f"wrote: {track_dir}")
    print(f"wrote: {legacy_dir}")
    print(f"wrote: {root / 'data_track_yolo26.yaml'}")


if __name__ == "__main__":
    main()
