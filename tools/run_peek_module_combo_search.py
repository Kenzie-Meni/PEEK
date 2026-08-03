#!/usr/bin/env python3
"""Search PEEK module combinations for tracker performance."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


BASE_ARGS = (
    "--imgsz",
    "640",
    "--conf",
    "0.18",
    "--iou",
    "0.45",
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
    "--peek-graduate-hits",
    "8",
    "--no-video",
)


def combo_name(combo: tuple[int, ...], suffix: str = "") -> str:
    base = "mods_" + "_".join(str(module) for module in combo)
    return base + suffix


def generate_combos(modules: list[int], sizes: list[int], include_cluster: bool) -> list[tuple[str, tuple[str, ...]]]:
    variants: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for size in sizes:
        for combo in itertools.combinations(modules, size):
            args = ("--peek-modules", *(str(module) for module in combo))
            items = [(combo_name(combo), args)]
            if include_cluster and size >= 2:
                items.append(
                    (
                        combo_name(combo, "_cluster"),
                        args
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
                        ),
                    )
                )
            for item in items:
                if item not in seen:
                    variants.append(item)
                    seen.add(item)
    return variants


def run(cmd: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)


def write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: (row["idf1_like"], row["mota_like"], row["recall"]), reverse=True)
    keys = [
        "variant",
        "modules",
        "precision",
        "recall",
        "mota_like",
        "idf1_like",
        "id_switches",
        "tp",
        "fp",
        "fn",
        "pred",
        "gt",
        "frames",
    ]
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--modules", type=int, nargs="+", default=[4, 5, 6, 7, 12, 15, 16, 19, 21])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--include-cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion-model", choices=["none", "constant_velocity"], default="none")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    variants = generate_combos(args.modules, args.sizes, args.include_cluster)
    variants = [item for index, item in enumerate(variants) if index % args.num_shards == args.shard_index]

    rows: list[dict] = []
    for name, module_args in variants:
        jsonl = out / f"{name}.jsonl"
        metrics = out / f"{name}_metrics.json"
        log_path = out / f"{name}.log"
        if metrics.exists():
            row = json.loads(metrics.read_text())
        else:
            track_cmd = [
                sys.executable,
                str(REPO / "tools" / "track_yolo26_peek.py"),
                "--source",
                str(dataset / "train"),
                "--weights",
                str(weights),
                "--jsonl",
                str(jsonl),
                "--output",
                str(out / f"{name}.mp4"),
                "--device",
                args.device,
                *BASE_ARGS,
                *module_args,
                "--motion-model",
                args.motion_model,
            ]
            if args.max_frames:
                track_cmd.extend(["--max-frames", str(args.max_frames)])
            run(track_cmd, log_path)
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
                    name,
                ],
                log_path,
            )
            row = json.loads(metrics.read_text())
        row["modules"] = name.removeprefix("mods_").removesuffix("_cluster").replace("_", "+")
        rows.append(row)
        write_summary(rows, out / f"summary_shard{args.shard_index}.csv")

    write_summary(rows, out / f"summary_shard{args.shard_index}.csv")


if __name__ == "__main__":
    main()
