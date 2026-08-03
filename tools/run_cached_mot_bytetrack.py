#!/usr/bin/env python3
"""Run ByteTrack from cached YOLO detections and evaluate on MOT."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "ultralytics"))

from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402


class CachedResults:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32, copy=False)
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, index):
        return CachedResults(self.xyxy[index], self.conf[index], self.cls[index])

    @property
    def xywh(self) -> np.ndarray:
        xyxy = np.atleast_2d(self.xyxy)
        out = np.empty_like(xyxy, dtype=np.float32)
        out[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        out[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        out[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        out[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return out


def cached_results(row: dict) -> CachedResults:
    detections = row.get("all_yolo_detections", [])
    if not detections:
        return CachedResults(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
    return CachedResults(
        np.array([det["xyxy"] for det in detections], dtype=np.float32),
        np.array([det["score"] for det in detections], dtype=np.float32),
        np.array([0 if det.get("class_id") is None else det["class_id"] for det in detections], dtype=np.float32),
    )


def run_sequence(cache_jsonl: Path, out_jsonl: Path, args_ns: SimpleNamespace) -> None:
    tracker = BYTETracker(args_ns, frame_rate=30)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with cache_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            tracks = tracker.update(cached_results(row))
            encoded = []
            for item in tracks:
                x1, y1, x2, y2, tid, score, cls, _idx = item.tolist()
                encoded.append(
                    {
                        "id": int(tid),
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(score),
                        "class_id": int(cls),
                        "source": "bytetrack",
                        "origin": "bytetrack",
                        "module": None,
                        "modules": [],
                        "age": 1,
                        "hits": 1,
                        "missed": 0,
                    }
                )
            dst.write(json.dumps({"frame_index": int(row["frame_index"]), "tracks": encoded}) + "\n")


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def summarize(metrics_dir: Path, output_dir: Path) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0, "gt": 0, "pred": 0}
    for path in metrics_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in totals:
            totals[key] += int(data[key])
    precision = totals["tp"] / totals["pred"] if totals["pred"] else 0.0
    recall = totals["tp"] / totals["gt"] if totals["gt"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row = {
        "variant": "bytetrack_cached",
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mota_like": 1.0 - (totals["fn"] + totals["fp"] + totals["id_switches"]) / totals["gt"] if totals["gt"] else 0.0,
        "idf1_like": 2 * totals["tp"] / (2 * totals["tp"] + totals["fp"] + totals["fn"]) if (2 * totals["tp"] + totals["fp"] + totals["fn"]) else 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--mot-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--track-high-thresh", type=float, default=0.25)
    parser.add_argument("--track-low-thresh", type=float, default=0.10)
    parser.add_argument("--new-track-thresh", type=float, default=0.25)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--fuse-score", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tracker_args = SimpleNamespace(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        fuse_score=args.fuse_score,
    )
    metrics_dir = args.output_dir / "metrics"
    jsonl_dir = args.output_dir / "jsonl"
    for cache_jsonl in sorted((args.cache_dir / "jsonl").glob("*.jsonl")):
        seq = cache_jsonl.stem
        metrics = metrics_dir / f"{seq}_bytetrack.json"
        if metrics.exists():
            continue
        out_jsonl = jsonl_dir / f"{seq}_bytetrack.jsonl"
        run_sequence(cache_jsonl, out_jsonl, tracker_args)
        run(
            [
                sys.executable,
                str(REPO / "tools/evaluate_mot_jsonl.py"),
                "--jsonl",
                str(out_jsonl),
                "--gt",
                str(args.mot_root / "train" / seq / "gt" / "gt.txt"),
                "--output",
                str(metrics),
            ],
            args.output_dir / "logs" / f"{seq}_bytetrack_eval.log",
        )
    print(json.dumps(summarize(metrics_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
