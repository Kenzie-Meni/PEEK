#!/usr/bin/env python3
"""Export frame-wise YOLO detections for MOT-style image sequences."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    image_dir: Path


def add_ultralytics() -> None:
    repo = Path(__file__).resolve().parents[1]
    vendor = repo / "third_party" / "ultralytics"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def is_image_dir(path: Path) -> bool:
    return path.is_dir() and any(child.suffix.lower() in IMAGE_EXTS for child in path.iterdir() if child.is_file())


def find_sequences(root: Path) -> list[SequenceInfo]:
    sequences: list[SequenceInfo] = []
    for img1 in sorted(root.rglob("img1")):
        if is_image_dir(img1):
            sequences.append(SequenceInfo(name=img1.parent.name, image_dir=img1))
    if sequences:
        return sequences
    for path in sorted(root.rglob("*")):
        if is_image_dir(path):
            sequences.append(SequenceInfo(name=path.name, image_dir=path))
    return sequences


def frame_number(path: Path, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Dataset root or split root containing MOT img1 dirs.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="COCO class ids; 0=person.")
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    add_ultralytics()
    from ultralytics import YOLO  # type: ignore

    sequences = find_sequences(args.root)
    if args.max_seqs:
        sequences = sequences[: args.max_seqs]
    if not sequences:
        raise FileNotFoundError(f"No MOT-style image sequences under {args.root}")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    manifest = []

    for seq in sequences:
        images = sorted(path for path in seq.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
        if args.max_frames:
            images = images[: args.max_frames]
        out_path = args.out / f"{seq.name}.txt"
        rows: list[str] = []
        frames = 0
        detections = 0
        for fallback_frame, image_path in enumerate(images, start=1):
            frame_id = frame_number(image_path, fallback_frame)
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                classes=args.classes,
                verbose=False,
            )[0]
            frames += 1
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = [float(v) for v in box]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                rows.append(
                    f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.6f},{int(cls_id)},-1,-1"
                )
                detections += 1
        out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        manifest.append(
            {
                "sequence": seq.name,
                "image_dir": str(seq.image_dir),
                "frames": frames,
                "detections": detections,
                "mot_det_file": str(out_path),
            }
        )
        print(f"{seq.name}: frames={frames} detections={detections} -> {out_path}", flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest: {args.out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
