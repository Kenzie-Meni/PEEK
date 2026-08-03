#!/usr/bin/env python3
"""Run selected PEEK tracker variants and evaluate each against track labels."""

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


BASE = (
    "--imgsz",
    "640",
    "--conf",
    "0.18",
    "--iou",
    "0.45",
    "--peek-modules",
    "4",
    "5",
    "6",
    "7",
    "12",
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
    "--no-video",
)


VARIANTS = (
    Variant("peek_m12_only", BASE + ("--peek-modules", "12", "--peek-graduate-hits", "8")),
    Variant(
        "peek_m12_only_kalman",
        BASE
        + (
            "--peek-modules",
            "12",
            "--peek-graduate-hits",
            "8",
            "--motion-model",
            "constant_velocity",
            "--motion-process-noise",
            "1.0",
            "--motion-measurement-noise",
            "10.0",
        ),
    ),
    Variant("peek_selected_balanced_h8", BASE + ("--peek-graduate-hits", "8")),
    Variant(
        "peek_selected_balanced_h8_kalman",
        BASE
        + (
            "--peek-graduate-hits",
            "8",
            "--motion-model",
            "constant_velocity",
            "--motion-process-noise",
            "1.0",
            "--motion-measurement-noise",
            "10.0",
        ),
    ),
    Variant(
        "peek_selected_cluster_union",
        BASE
        + (
            "--peek-union-clusters",
            "--peek-cluster-iou",
            "0.10",
            "--peek-cluster-center-frac",
            "0.35",
            "--peek-cluster-min-modules",
            "1",
            "--peek-cluster-min-area",
            "180",
            "--peek-cluster-min-short-side",
            "12",
            "--peek-graduate-hits",
            "8",
        ),
    ),
    Variant(
        "peek_selected_cluster_union_kalman_smooth",
        BASE
        + (
            "--peek-union-clusters",
            "--peek-cluster-iou",
            "0.10",
            "--peek-cluster-center-frac",
            "0.35",
            "--peek-cluster-min-modules",
            "1",
            "--peek-cluster-min-area",
            "180",
            "--peek-cluster-min-short-side",
            "12",
            "--peek-graduate-hits",
            "8",
            "--motion-model",
            "constant_velocity",
            "--motion-process-noise",
            "0.4",
            "--motion-measurement-noise",
            "16.0",
        ),
    ),
    Variant(
        "peek_selected_cluster_union_kalman_responsive",
        BASE
        + (
            "--peek-union-clusters",
            "--peek-cluster-iou",
            "0.10",
            "--peek-cluster-center-frac",
            "0.35",
            "--peek-cluster-min-modules",
            "1",
            "--peek-cluster-min-area",
            "180",
            "--peek-cluster-min-short-side",
            "12",
            "--peek-graduate-hits",
            "8",
            "--motion-model",
            "constant_velocity",
            "--motion-process-noise",
            "4.0",
            "--motion-measurement-noise",
            "6.0",
        ),
    ),
    Variant(
        "peek_softsupport_pad22",
        BASE
        + (
            "--peek-union-gate",
            "--peek-union-pad",
            "0.22",
            "--peek-union-min-iou",
            "0.0",
            "--peek-graduate-hits",
            "8",
        ),
    ),
    Variant(
        "peek_redundancy_strict",
        BASE
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
    ),
    Variant(
        "peek_highz_clean",
        BASE
        + (
            "--peek-z",
            "2.2",
            "--peek-min-area",
            "180",
            "--peek-max-regions-per-module",
            "2",
            "--peek-graduate-hits",
            "10",
        ),
    ),
)


def run(cmd: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for variant in VARIANTS:
        log_path = out / f"{variant.name}.log"
        jsonl = out / f"{variant.name}.jsonl"
        metrics = out / f"{variant.name}_metrics.json"
        run(
            [
                sys.executable,
                str(REPO / "tools" / "track_yolo26_peek.py"),
                "--source",
                str(dataset / "train"),
                "--weights",
                str(weights),
                "--jsonl",
                str(jsonl),
                "--output",
                str(out / f"{variant.name}.mp4"),
                "--device",
                args.device,
                *variant.args,
            ],
            log_path,
        )
        run(
            [
                sys.executable,
                str(REPO / "tools" / "evaluate_peek_track_jsonl.py"),
                "--dataset",
                str(dataset),
                "--jsonl",
                str(jsonl),
                "--output",
                str(metrics),
                "--name",
                variant.name,
            ],
            log_path,
        )
        row = json.loads(metrics.read_text())
        summary_rows.append(
            {
                "variant": row["variant"],
                "precision": row["precision"],
                "recall": row["recall"],
                "mota_like": row["mota_like"],
                "idf1_like": row["idf1_like"],
                "id_switches": row["id_switches"],
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
                "pred": row["pred"],
                "gt": row["gt"],
                "frames": row["frames"],
            }
        )
        (out / "summary_partial.csv").write_text(to_csv(summary_rows), encoding="utf-8")

    summary_rows.sort(key=lambda r: (r["idf1_like"], r["mota_like"], r["recall"]), reverse=True)
    (out / "summary.csv").write_text(to_csv(summary_rows), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row[k]) for k in keys))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
