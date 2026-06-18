#!/usr/bin/env python3
"""Train the local YOLO26 small model on the prepared PEEK bbox dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ultralytics_root = repo / "third_party" / "ultralytics"
    sys.path.insert(0, str(ultralytics_root))

    os.environ.setdefault("ULTRALYTICS_DIR", str(repo))

    from ultralytics import YOLO  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox/data.yaml")
    parser.add_argument("--weights", default=str(repo / "weights" / "yolo26s.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default=str(repo / "runs" / "detect"))
    parser.add_argument("--name", default="peek_yolo26s_bbox")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-period", type=int, default=-1)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        scale=args.scale,
        degrees=args.degrees,
        translate=args.translate,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        fliplr=args.fliplr,
        erasing=args.erasing,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        seed=args.seed,
        deterministic=args.deterministic,
        save_period=args.save_period,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
