#!/usr/bin/env python3
"""Run YOLO26+PEEK tracker variants on MOT image sequences and score them."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    name: str
    args: tuple[str, ...]


VARIANTS = [
    Variant("yolo_track", ("--no-peek-proximity-gate", "--peek-z", "99", "--shadow-yolo-conf", "0.004")),
    Variant("peek_m12", ("--peek-modules", "12", "--peek-graduate-hits", "3")),
    Variant(
        "peek_selected_union",
        (
            "--peek-modules",
            "4",
            "5",
            "6",
            "7",
            "12",
            "--peek-union-clusters",
            "--peek-cluster-min-modules",
            "1",
            "--peek-cluster-min-area",
            "220",
            "--peek-cluster-min-short-side",
            "10",
            "--peek-graduate-hits",
            "3",
        ),
    ),
    Variant(
        "peek_selected_union_kalman",
        (
            "--peek-modules",
            "4",
            "5",
            "6",
            "7",
            "12",
            "--peek-union-clusters",
            "--peek-cluster-min-modules",
            "1",
            "--peek-cluster-min-area",
            "220",
            "--peek-cluster-min-short-side",
            "10",
            "--peek-graduate-hits",
            "3",
            "--motion-model",
            "constant_velocity",
            "--motion-process-noise",
            "4.0",
            "--motion-measurement-noise",
            "25.0",
        ),
    ),
]


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-root", required=True, type=Path, help="MOT root with train/<seq>/img1 and gt.")
    parser.add_argument("--weights", default="weights/yolo26s.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs/track/mot_peek_benchmark")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--variants", nargs="*", default=[variant.name for variant in VARIANTS])
    args = parser.parse_args()

    selected = [variant for variant in VARIANTS if variant.name in set(args.variants)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq_dir in sorted((args.mot_root / "train").glob("*")):
        img_dir = seq_dir / "img1"
        gt_file = seq_dir / "gt" / "gt.txt"
        if not img_dir.exists() or not gt_file.exists():
            continue
        for variant in selected:
            stem = f"{seq_dir.name}_{variant.name}"
            jsonl = args.output_dir / "jsonl" / f"{stem}.jsonl"
            metrics_path = args.output_dir / "metrics" / f"{stem}.json"
            log_path = args.output_dir / "logs" / f"{stem}.log"
            cmd = [
                sys.executable,
                str(REPO / "tools/track_yolo26_peek.py"),
                "--source",
                str(img_dir),
                "--weights",
                args.weights,
                "--jsonl",
                str(jsonl),
                "--output",
                str(args.output_dir / "videos" / f"{stem}.mp4"),
                "--device",
                args.device,
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
                "--iou",
                str(args.iou),
                "--no-video",
                *variant.args,
            ]
            if args.max_frames:
                cmd.extend(["--max-frames", str(args.max_frames)])
            run(cmd, log_path)
            eval_cmd = [
                sys.executable,
                str(REPO / "tools/evaluate_mot_jsonl.py"),
                "--jsonl",
                str(jsonl),
                "--gt",
                str(gt_file),
                "--output",
                str(metrics_path),
                "--iou",
                "0.5",
            ]
            run(eval_cmd, args.output_dir / "logs" / f"{stem}_eval.log")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({"sequence": seq_dir.name, "variant": variant.name, **metrics})

    summary = {}
    for row in rows:
        item = summary.setdefault(row["variant"], {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0, "gt": 0, "pred": 0})
        for key in item:
            item[key] += int(row[key])
    summary_rows = []
    for variant, total in summary.items():
        precision = total["tp"] / total["pred"] if total["pred"] else 0.0
        recall = total["tp"] / total["gt"] if total["gt"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        mota = 1.0 - (total["fn"] + total["fp"] + total["id_switches"]) / total["gt"] if total["gt"] else 0.0
        idf1 = 2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"]) if (2 * total["tp"] + total["fp"] + total["fn"]) else 0.0
        summary_rows.append({"variant": variant, **total, "precision": precision, "recall": recall, "f1": f1, "mota_like": mota, "idf1_like": idf1})
    summary_rows.sort(key=lambda row: (row["idf1_like"], row["mota_like"], row["f1"]), reverse=True)

    (args.output_dir / "summary.json").write_text(json.dumps({"summary": summary_rows, "per_sequence": rows}, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["variant"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
