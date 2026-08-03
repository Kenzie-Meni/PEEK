#!/usr/bin/env python3
"""Run Ultralytics ByteTrack on MOT sequences and score with the same evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ultralytics"))

from peek.utils.paths import configure_ultralytics_dir  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def source_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def encode_result(result) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    ids = boxes.id.detach().cpu().numpy()
    tracks = []
    for box, score, cls_id, track_id in zip(xyxy, conf, cls, ids):
        tracks.append(
            {
                "id": int(track_id),
                "xyxy": [float(v) for v in box],
                "score": float(score),
                "class_id": int(cls_id),
                "class_name": "person" if int(cls_id) == 0 else str(int(cls_id)),
                "source": "bytetrack",
                "origin": "bytetrack",
                "module": None,
                "modules": [],
                "age": 1,
                "hits": 1,
                "missed": 0,
            }
        )
    return tracks


def run_sequence(
    model,
    img_dir: Path,
    jsonl_path: Path,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    tracker_cfg: str,
    max_frames: int,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    paths = source_images(img_dir)
    if max_frames:
        paths = paths[:max_frames]
    start = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(path)
            result = model.track(
                source=frame,
                persist=True,
                tracker=tracker_cfg,
                imgsz=imgsz,
                device=device,
                conf=conf,
                iou=iou,
                classes=[0],
                verbose=False,
            )[0]
            tracks = encode_result(result)
            handle.write(
                json.dumps(
                    {
                        "frame_index": index,
                        "latency_ms": 0.0,
                        "num_yolo_detections": len(tracks),
                        "num_shadow_yolo_detections": 0,
                        "num_peek_detections": 0,
                        "peek_detections": [],
                        "shadow_yolo_detections": [],
                        "tracks": tracks,
                    }
                )
                + "\n"
            )
    elapsed = time.perf_counter() - start
    print(f"{img_dir.name}: frames={len(paths)} elapsed_s={elapsed:.2f} fps={len(paths) / elapsed if elapsed else 0:.2f}")


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def summarize(metrics_dir: Path, output_dir: Path) -> None:
    totals = {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0, "gt": 0, "pred": 0}
    mot = {"objects": 0.0, "idf1": 0.0, "mota": 0.0, "motp": 0.0, "switches": 0, "frags": 0}
    for path in sorted(metrics_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in totals:
            totals[key] += int(data[key])
        mm = data.get("motmetrics", {})
        objects = float(mm.get("num_objects", 0.0) or 0.0)
        mot["objects"] += objects
        mot["idf1"] += float(mm.get("idf1", 0.0) or 0.0) * objects
        mot["mota"] += float(mm.get("mota", 0.0) or 0.0) * objects
        mot["motp"] += float(mm.get("motp", 0.0) or 0.0) * objects
        mot["switches"] += int(float(mm.get("num_switches", 0.0) or 0.0))
        mot["frags"] += int(float(mm.get("num_fragmentations", 0.0) or 0.0))
    precision = totals["tp"] / totals["pred"] if totals["pred"] else 0.0
    recall = totals["tp"] / totals["gt"] if totals["gt"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mota_like = 1.0 - (totals["fn"] + totals["fp"] + totals["id_switches"]) / totals["gt"] if totals["gt"] else 0.0
    row = {
        "variant": "bytetrack",
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mota_like": mota_like,
        "idf1_like": 2 * totals["tp"] / (2 * totals["tp"] + totals["fp"] + totals["fn"]) if (2 * totals["tp"] + totals["fp"] + totals["fn"]) else 0.0,
        "mot_idf1": mot["idf1"] / mot["objects"] if mot["objects"] else 0.0,
        "mot_mota": mot["mota"] / mot["objects"] if mot["objects"] else 0.0,
        "mot_motp": mot["motp"] / mot["objects"] if mot["objects"] else 0.0,
        "mot_switches": mot["switches"],
        "mot_fragmentations": mot["frags"],
    }
    (output_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-root", required=True, type=Path)
    parser.add_argument("--weights", default="weights/yolo26s.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs/track/mot_bytetrack_benchmark")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    configure_ultralytics_dir()
    from ultralytics import YOLO  # noqa: WPS433

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    jsonl_dir = args.output_dir / "jsonl"
    model = YOLO(args.weights)

    for seq_dir in sorted((args.mot_root / "train").glob("*")):
        img_dir = seq_dir / "img1"
        gt_file = seq_dir / "gt" / "gt.txt"
        if not img_dir.exists() or not gt_file.exists():
            continue
        jsonl_path = jsonl_dir / f"{seq_dir.name}_bytetrack.jsonl"
        metrics_path = metrics_dir / f"{seq_dir.name}_bytetrack.json"
        if not metrics_path.exists():
            run_sequence(model, img_dir, jsonl_path, args.device, args.imgsz, args.conf, args.iou, args.tracker, args.max_frames)
            run(
                [
                    sys.executable,
                    str(REPO / "tools/evaluate_mot_jsonl.py"),
                    "--jsonl",
                    str(jsonl_path),
                    "--gt",
                    str(gt_file),
                    "--output",
                    str(metrics_path),
                    "--iou",
                    "0.5",
                ],
                args.output_dir / "logs" / f"{seq_dir.name}_bytetrack_eval.log",
            )
    summarize(metrics_dir, args.output_dir)


if __name__ == "__main__":
    main()
