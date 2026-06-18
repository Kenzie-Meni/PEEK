#!/usr/bin/env python3
"""Build an oversampled training list for tiny antenna/thruster recall."""

from __future__ import annotations

import argparse
from pathlib import Path


TARGET_CLASSES = {0, 3}  # antenna, thruster


def target_score(label_path: Path, tiny_area: float, small_area: float) -> int:
    score = 0
    if not label_path.exists():
        return score
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in TARGET_CLASSES:
            continue
        area = float(parts[3]) * float(parts[4])
        if area < tiny_area:
            score += 2
        elif area < small_area:
            score += 1
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox")
    parser.add_argument("--out", default="/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox/train_smallrecall.txt")
    parser.add_argument("--yaml-out", default="/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox/data_smallrecall.yaml")
    parser.add_argument("--tiny-area", type=float, default=0.005)
    parser.add_argument("--small-area", type=float, default=0.02)
    parser.add_argument("--max-extra", type=int, default=3)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    image_dir = dataset / "train" / "images"
    label_dir = dataset / "train" / "labels"
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    lines: list[str] = []
    repeated = 0
    for image_path in images:
        lines.append(str(image_path.resolve()))
        score = target_score(label_dir / f"{image_path.stem}.txt", args.tiny_area, args.small_area)
        extra = min(args.max_extra, score)
        repeated += extra
        lines.extend([str(image_path.resolve())] * extra)

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_out = Path(args.yaml_out)
    yaml_out.write_text(
        "\n".join(
            [
                f"path: {dataset.resolve()}",
                f"train: {out.resolve()}",
                "val: valid/images",
                "test:",
                "  - test_gh10018/images",
                "  - test_tracking/images",
                "  - test_v77/images",
                "names:",
                "  0: antenna",
                "  1: body",
                "  2: solar",
                "  3: thruster",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"base_images={len(images)} total_entries={len(lines)} repeated_entries={repeated}")
    print(f"list={out}")
    print(f"yaml={yaml_out}")


if __name__ == "__main__":
    main()
