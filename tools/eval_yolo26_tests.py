#!/usr/bin/env python3
"""Evaluate a YOLO26 checkpoint on each prepared PEEK test split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


DEFAULT_TESTS = {
    "gh10018": "/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox/data_test_gh10018.yaml",
    "tracking": "/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox/data_test_tracking.yaml",
    "v77": "/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox/data_test_v77.yaml",
}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ultralytics_root = repo / "third_party" / "ultralytics"
    sys.path.insert(0, str(ultralytics_root))

    os.environ.setdefault("ULTRALYTICS_DIR", str(repo))

    from ultralytics import YOLO  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(repo / "runs" / "detect" / "peek_yolo26s_bbox_mapmax_960" / "weights" / "best.pt"),
    )
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=str(repo / "runs" / "detect" / "eval_tests"))
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    for imgsz in args.imgsz:
        for test_name, data_yaml in DEFAULT_TESTS.items():
            print(f"\n=== Evaluating {test_name} at imgsz={imgsz}: {data_yaml} ===", flush=True)
            model = YOLO(args.weights)
            metrics = model.val(
                data=data_yaml,
                split="test",
                imgsz=imgsz,
                device=args.device,
                batch=args.batch,
                project=args.project,
                name=f"{test_name}_{imgsz}",
                conf=args.conf,
                iou=args.iou,
                exist_ok=True,
                plots=True,
            )
            box = metrics.box
            print(
                f"{test_name} imgsz={imgsz}: mAP50={box.map50:.5f}, "
                f"mAP50-95={box.map:.5f}, precision={box.mp:.5f}, recall={box.mr:.5f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
