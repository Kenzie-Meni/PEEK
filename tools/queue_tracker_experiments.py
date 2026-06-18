#!/usr/bin/env python3
"""Queue YOLO26+PEEK tracker visual experiments and summarize failure signals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
from typing import Iterable

import cv2


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/rwhite/mambaforge/envs/peek/bin/python")
SOURCE = Path("/home/rwhite/NFS/All_Team/Ryan/datasets/GSFC")
WEIGHTS = Path("weights/yolo26s_peek_bbox_best.pt")
OUTDIR = Path("runs/track/experiments")


@dataclass(frozen=True)
class Experiment:
    name: str
    graduate_hits: int
    args: tuple[str, ...]
    note: str


BASE_ARGS = (
    "--imgsz",
    "640",
    "--conf",
    "0.18",
    "--iou",
    "0.45",
    "--peek-modules",
    "12",
    "16",
    "19",
    "21",
    "--peek-dog",
    "--peek-dog-small",
    "2.0",
    "--peek-dog-large",
    "9.0",
    "--peek-z",
    "1.9",
    "--peek-min-area",
    "160",
    "--peek-max-area-frac",
    "0.30",
    "--peek-max-regions-per-module",
    "3",
    "--peek-min-extent",
    "0.12",
    "--peek-max-aspect-ratio",
    "4.0",
    "--peek-min-short-side",
    "12",
    "--peek-border-margin",
    "10",
    "--peek-focus-z",
    "0.55",
    "--peek-focus-local-z",
    "0.95",
    "--peek-focus-padding",
    "0.30",
    "--peek-focus-min-area-frac",
    "0.40",
    "--peek-focus-max-regions-per-track",
    "1",
    "--peek-proximity-gate",
    "--peek-anchor-max-distance-frac",
    "0.12",
    "--peek-focus-max-tracks",
    "4",
    "--peek-focus-max-missed",
    "3",
    "--peek-nms-iou",
    "0.35",
    "--peek-nms-max-candidates",
    "50",
    "--shadow-yolo",
    "--shadow-yolo-conf",
    "0.004",
    "--shadow-yolo-as-peek-anchor",
    "--suppress-peek-yolo-iou",
    "0.10",
    "--suppress-peek-yolo-containment",
    "0.55",
    "--explain",
)


EXPERIMENTS = (
    Experiment(
        name="suzuki_dog_tight_h8",
        graduate_hits=8,
        args=BASE_ARGS + ("--peek-graduate-hits", "8"),
        note="Current best visual baseline: DoG+Suzuki, no YOLO union gate.",
    ),
    Experiment(
        name="suzuki_dog_tight_h12",
        graduate_hits=12,
        args=BASE_ARGS + ("--peek-graduate-hits", "12"),
        note="Same regions, stricter temporal proof before showing PEEK-origin tracks.",
    ),
    Experiment(
        name="suzuki_dog_fewroi_h8",
        graduate_hits=8,
        args=BASE_ARGS
        + (
            "--peek-max-regions-per-module",
            "2",
            "--peek-graduate-hits",
            "8",
        ),
        note="Lower ROI budget to suppress weak off-object regions.",
    ),
    Experiment(
        name="suzuki_dog_highz_h8",
        graduate_hits=8,
        args=BASE_ARGS
        + (
            "--peek-z",
            "2.2",
            "--peek-min-area",
            "180",
            "--peek-graduate-hits",
            "8",
        ),
        note="Higher PEEK support threshold for cleaner proposals.",
    ),
    Experiment(
        name="suzuki_dog_shape_strict_h8",
        graduate_hits=8,
        args=BASE_ARGS
        + (
            "--peek-min-extent",
            "0.18",
            "--peek-max-aspect-ratio",
            "4.0",
            "--peek-border-margin",
            "12",
            "--peek-graduate-hits",
            "8",
        ),
        note="Reject skinny/sparse contours that often become far-off graduated tracks.",
    ),
    Experiment(
        name="suzuki_dog_balanced_h10",
        graduate_hits=10,
        args=BASE_ARGS
        + (
            "--peek-z",
            "2.0",
            "--peek-min-area",
            "180",
            "--peek-max-regions-per-module",
            "2",
            "--peek-min-extent",
            "0.14",
            "--peek-graduate-hits",
            "10",
        ),
        note="Balanced stricter contour and temporal settings.",
    ),
    Experiment(
        name="suzuki_dog_redundancy_strict_h8",
        graduate_hits=8,
        args=BASE_ARGS
        + (
            "--peek-nms-iou",
            "0.25",
            "--suppress-peek-yolo-iou",
            "0.05",
            "--suppress-peek-yolo-containment",
            "0.40",
            "--peek-graduate-hits",
            "8",
        ),
        note="Aggressively remove stacked PEEK boxes and regions already explained by YOLO.",
    ),
    Experiment(
        name="suzuki_dog_redundancy_loose_h10",
        graduate_hits=10,
        args=BASE_ARGS
        + (
            "--peek-nms-iou",
            "0.50",
            "--suppress-peek-yolo-iou",
            "0.18",
            "--suppress-peek-yolo-containment",
            "0.75",
            "--peek-graduate-hits",
            "10",
        ),
        note="Keep more PEEK while still removing clear YOLO duplicates.",
    ),
    Experiment(
        name="suzuki_dog_softsupport_pad22_h8",
        graduate_hits=8,
        args=BASE_ARGS
        + (
            "--peek-union-gate",
            "--peek-union-pad",
            "0.22",
            "--peek-union-min-iou",
            "0.0",
            "--peek-graduate-hits",
            "8",
        ),
        note="Loose object-support gate to reject distant regions without tight YOLO boxing.",
    ),
    Experiment(
        name="suzuki_dog_softsupport_pad35_h10",
        graduate_hits=10,
        args=BASE_ARGS
        + (
            "--peek-union-gate",
            "--peek-union-pad",
            "0.35",
            "--peek-union-min-iou",
            "0.0",
            "--peek-graduate-hits",
            "10",
        ),
        note="Very loose object-support gate plus stricter graduation.",
    ),
    Experiment(
        name="suzuki_dog_scale_wide_h8",
        graduate_hits=8,
        args=BASE_ARGS
        + (
            "--peek-dog-large",
            "13.0",
            "--peek-graduate-hits",
            "8",
        ),
        note="Wider background subtraction to see whether far blobs disappear.",
    ),
    Experiment(
        name="suzuki_dog_scale_fine_h8",
        graduate_hits=8,
        args=BASE_ARGS
        + (
            "--peek-dog-small",
            "1.2",
            "--peek-dog-large",
            "7.0",
            "--peek-graduate-hits",
            "8",
        ),
        note="Finer DoG scale for smaller component-like regions.",
    ),
)


def run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def complete_jsonl(path: Path, min_frames: int = 1000) -> bool:
    if not path.exists():
        return False
    frames = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for frames, _ in enumerate(handle, start=1):
                pass
    except OSError:
        return False
    return frames >= min_frames


def video_info(path: Path) -> dict[str, float | int | str | None]:
    if not path.exists():
        return {"frames": 0, "width": None, "height": None, "fps": None, "size_bytes": 0}
    cap = cv2.VideoCapture(str(path))
    try:
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "size_bytes": path.stat().st_size,
        }
    finally:
        cap.release()


def summarize_jsonl(path: Path, graduate_hits: int) -> dict[str, object]:
    frames = 0
    peek_counts = []
    latency = []
    source_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    peek_origin_ids: set[int] = set()
    graduated_peek_ids: set[int] = set()
    frames_with_graduated_peek = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frames += 1
            peek_counts.append(int(row.get("num_peek_detections", 0)))
            latency.append(float(row.get("latency_ms", 0.0)))
            frame_has_graduated = False
            for track in row.get("tracks", []):
                source = str(track.get("source"))
                origin = str(track.get("origin", source))
                class_name = str(track.get("class_name"))
                track_id = int(track.get("id"))
                hits = int(track.get("hits", 0))
                source_counts[source] += 1
                origin_counts[origin] += 1
                class_counts[class_name] += 1
                if origin == "peek":
                    peek_origin_ids.add(track_id)
                    if hits >= graduate_hits:
                        graduated_peek_ids.add(track_id)
                        frame_has_graduated = True
            if frame_has_graduated:
                frames_with_graduated_peek += 1

    return {
        "frames": frames,
        "peek_proposals_avg": sum(peek_counts) / len(peek_counts) if peek_counts else 0.0,
        "peek_proposals_min": min(peek_counts) if peek_counts else 0,
        "peek_proposals_max": max(peek_counts) if peek_counts else 0,
        "latency_ms_avg": sum(latency) / len(latency) if latency else 0.0,
        "source_counts": dict(source_counts),
        "origin_counts": dict(origin_counts),
        "class_counts": dict(class_counts),
        "peek_origin_ids": len(peek_origin_ids),
        "graduated_peek_ids": len(graduated_peek_ids),
        "frames_with_graduated_peek": frames_with_graduated_peek,
    }


def write_worker_summary(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = [
        "name",
        "status",
        "device",
        "frames",
        "peek_proposals_avg",
        "peek_proposals_min",
        "peek_proposals_max",
        "latency_ms_avg",
        "peek_origin_ids",
        "graduated_peek_ids",
        "frames_with_graduated_peek",
        "compressed_size_mb",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    assigned = [exp for idx, exp in enumerate(EXPERIMENTS) if idx % args.num_workers == args.worker_index]
    print(
        f"tracker experiment worker={args.worker_index}/{args.num_workers} "
        f"device={args.device} assigned={len(assigned)}",
        flush=True,
    )

    rows = []
    for exp in assigned:
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        output = OUTDIR / f"{exp.name}.mp4"
        compressed = OUTDIR / f"{exp.name}_compressed.mp4"
        jsonl = OUTDIR / f"{exp.name}.jsonl"
        stats_path = OUTDIR / f"{exp.name}_stats.json"
        print(f"\n[{start}] START {exp.name}: {exp.note}", flush=True)

        status = "ok"
        if args.force or not complete_jsonl(jsonl) or not compressed.exists():
            cmd = [
                str(PYTHON),
                "-u",
                "tools/track_yolo26_peek.py",
                "--source",
                str(SOURCE),
                "--weights",
                str(WEIGHTS),
                "--output",
                str(output),
                "--jsonl",
                str(jsonl),
                "--device",
                str(args.device),
                *exp.args,
            ]
            if args.max_frames:
                cmd += ["--max-frames", str(args.max_frames)]
            run(cmd, REPO)
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(output),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "28",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(compressed),
                ],
                REPO,
            )
        else:
            status = "skipped_complete"
            print(f"Skipping complete experiment: {exp.name}", flush=True)

        stats = summarize_jsonl(jsonl, exp.graduate_hits)
        stats.update(
            {
                "name": exp.name,
                "status": status,
                "device": str(args.device),
                "graduate_hits": exp.graduate_hits,
                "note": exp.note,
                "output": str(output),
                "compressed": str(compressed),
                "jsonl": str(jsonl),
                "video": video_info(output),
                "compressed_video": video_info(compressed),
                "compressed_size_mb": round(compressed.stat().st_size / (1024 * 1024), 2)
                if compressed.exists()
                else 0.0,
            }
        )
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(stats)
        write_worker_summary(rows, OUTDIR / f"summary_worker{args.worker_index}.csv")
        print(
            "DONE {name}: avg_peek={avg:.2f} peek_ids={ids} graduated={grad} "
            "frames_grad={frames_grad} compressed={size:.2f}MB".format(
                name=exp.name,
                avg=float(stats["peek_proposals_avg"]),
                ids=int(stats["peek_origin_ids"]),
                grad=int(stats["graduated_peek_ids"]),
                frames_grad=int(stats["frames_with_graduated_peek"]),
                size=float(stats["compressed_size_mb"]),
            ),
            flush=True,
        )

    print(f"worker={args.worker_index} complete", flush=True)


if __name__ == "__main__":
    main()
