#!/usr/bin/env python3
"""Search PEEK layers on MOT sequences with optional motion models."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


BASE_ARGS = (
    "--imgsz",
    "640",
    "--conf",
    "0.10",
    "--iou",
    "0.45",
    "--peek-dog",
    "--peek-dog-small",
    "2.0",
    "--peek-dog-large",
    "9.0",
    "--peek-z",
    "1.0",
    "--peek-min-area",
    "120",
    "--peek-max-area-frac",
    "0.35",
    "--peek-max-regions-per-module",
    "4",
    "--peek-min-extent",
    "0.08",
    "--peek-max-aspect-ratio",
    "4.0",
    "--peek-min-short-side",
    "12",
    "--peek-border-margin",
    "8",
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
    "--peek-graduate-hits",
    "3",
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


@dataclass(frozen=True)
class Variant:
    name: str
    modules: tuple[int, ...]
    motion_model: str
    extra_args: tuple[str, ...] = ()


def module_group_variants(modules: list[int], motion_models: list[str]) -> list[Variant]:
    groups: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = [
        ("backbone_early", (0, 1, 2, 3, 4), ()),
        ("backbone_mid", (4, 5, 6, 7), ()),
        ("backbone_deep", (8, 9, 10), ()),
        ("neck_early", (12, 13, 15, 16), ()),
        ("neck_deep", (18, 19, 21, 22), ()),
        (
            "selected_union",
            (4, 5, 6, 7, 12),
            (
                "--peek-union-clusters",
                "--peek-cluster-iou",
                "0.10",
                "--peek-cluster-center-frac",
                "0.35",
                "--peek-cluster-min-modules",
                "1",
                "--peek-cluster-min-area",
                "220",
                "--peek-cluster-min-short-side",
                "12",
            ),
        ),
    ]
    variants: list[Variant] = []
    for motion in motion_models:
        suffix = "_kf" if motion == "constant_velocity" else "_nomotion"
        for module in modules:
            variants.append(Variant(f"m{module}{suffix}", (module,), motion))
        for name, combo, extra in groups:
            variants.append(Variant(f"{name}{suffix}", combo, motion, extra))
    return variants


def parse_variant(path: Path) -> tuple[str, str]:
    stem = path.stem
    for suffix in ("_nomotion", "_kf"):
        token = suffix + "."
        if token in stem:
            name = stem.split(token)[0] + suffix
            seq = stem.split(token, 1)[1]
            return name, seq
    raise ValueError(f"Could not parse metrics filename: {path}")


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def aggregate(metrics_dir: Path) -> list[dict]:
    variants: dict[str, dict] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        variant, _ = parse_variant(path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = variants.setdefault(
            variant,
            {
                "variant": variant,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "id_switches": 0,
                "gt": 0,
                "pred": 0,
                "mot_idf1_weighted": 0.0,
                "mot_mota_weighted": 0.0,
                "mot_motp_weighted": 0.0,
                "mot_objects": 0.0,
                "mot_switches": 0,
                "mot_fragmentations": 0,
            },
        )
        for key in ("tp", "fp", "fn", "id_switches", "gt", "pred"):
            row[key] += int(metrics[key])
        mm = metrics.get("motmetrics", {})
        objects = float(mm.get("num_objects", 0.0) or 0.0)
        row["mot_objects"] += objects
        row["mot_idf1_weighted"] += float(mm.get("idf1", 0.0) or 0.0) * objects
        row["mot_mota_weighted"] += float(mm.get("mota", 0.0) or 0.0) * objects
        row["mot_motp_weighted"] += float(mm.get("motp", 0.0) or 0.0) * objects
        row["mot_switches"] += int(float(mm.get("num_switches", 0.0) or 0.0))
        row["mot_fragmentations"] += int(float(mm.get("num_fragmentations", 0.0) or 0.0))

    rows = []
    for row in variants.values():
        precision = row["tp"] / row["pred"] if row["pred"] else 0.0
        recall = row["tp"] / row["gt"] if row["gt"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        mota_like = 1.0 - (row["fn"] + row["fp"] + row["id_switches"]) / row["gt"] if row["gt"] else 0.0
        idf1_like = 2 * row["tp"] / (2 * row["tp"] + row["fp"] + row["fn"]) if (2 * row["tp"] + row["fp"] + row["fn"]) else 0.0
        objects = row.pop("mot_objects")
        row.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mota_like": mota_like,
                "idf1_like": idf1_like,
                "mot_idf1": row.pop("mot_idf1_weighted") / objects if objects else 0.0,
                "mot_mota": row.pop("mot_mota_weighted") / objects if objects else 0.0,
                "mot_motp": row.pop("mot_motp_weighted") / objects if objects else 0.0,
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: (item["mot_idf1"], item["idf1_like"], item["mot_mota"]), reverse=True)
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = [
        "variant",
        "mot_idf1",
        "mot_mota",
        "mot_motp",
        "mot_switches",
        "mot_fragmentations",
        "precision",
        "recall",
        "f1",
        "mota_like",
        "idf1_like",
        "id_switches",
        "tp",
        "fp",
        "fn",
        "pred",
        "gt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-root", required=True, type=Path)
    parser.add_argument("--weights", default="weights/yolo26s.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs/track/mot_peek_module_search")
    parser.add_argument("--device", default="0")
    parser.add_argument("--modules", type=int, nargs="+", default=list(range(23)))
    parser.add_argument("--motion-models", nargs="+", choices=["none", "constant_velocity"], default=["none", "constant_velocity"])
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    variants = module_group_variants(args.modules, args.motion_models)
    variants = [variant for index, variant in enumerate(variants) if index % args.num_shards == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    logs_dir = args.output_dir / "logs"
    jsonl_dir = args.output_dir / "jsonl"

    for variant, seq_dir in itertools.product(variants, sorted((args.mot_root / "train").glob("*"))):
        img_dir = seq_dir / "img1"
        gt_file = seq_dir / "gt" / "gt.txt"
        if not img_dir.exists() or not gt_file.exists():
            continue
        stem = f"{variant.name}.{seq_dir.name}"
        metrics_path = metrics_dir / f"{stem}.json"
        if metrics_path.exists():
            continue
        jsonl_path = jsonl_dir / f"{stem}.jsonl"
        log_path = logs_dir / f"{stem}.log"
        track_cmd = [
            sys.executable,
            str(REPO / "tools/track_yolo26_peek.py"),
            "--source",
            str(img_dir),
            "--weights",
            args.weights,
            "--jsonl",
            str(jsonl_path),
            "--output",
            str(args.output_dir / "videos" / f"{stem}.mp4"),
            "--device",
            args.device,
            *BASE_ARGS,
            "--peek-modules",
            *(str(module) for module in variant.modules),
            "--motion-model",
            variant.motion_model,
            *variant.extra_args,
        ]
        if variant.motion_model == "constant_velocity":
            track_cmd.extend(["--motion-process-noise", "4.0", "--motion-measurement-noise", "25.0"])
        if args.max_frames:
            track_cmd.extend(["--max-frames", str(args.max_frames)])
        run(track_cmd, log_path)
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
            logs_dir / f"{stem}_eval.log",
        )
        rows = aggregate(metrics_dir)
        write_summary(rows, args.output_dir / "summary.csv")
        write_summary([row for row in rows if row["variant"] in {item.name for item in variants}], args.output_dir / f"summary_shard{args.shard_index}.csv")

    rows = aggregate(metrics_dir)
    write_summary(rows, args.output_dir / "summary.csv")
    write_summary([row for row in rows if row["variant"] in {item.name for item in variants}], args.output_dir / f"summary_shard{args.shard_index}.csv")
    print(json.dumps(rows[:10], indent=2))


if __name__ == "__main__":
    main()
