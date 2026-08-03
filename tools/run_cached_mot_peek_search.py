#!/usr/bin/env python3
"""Run PEEK tracker layer search from cached YOLO/PEEK artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from peek.tracking import (  # noqa: E402
    PEEKAssistedTracker,
    TrackedDetection,
    bbox_area,
    bbox_center_distance,
    bbox_iou,
    bbox_overlap_fraction,
)


@dataclass(frozen=True)
class Variant:
    name: str
    modules: tuple[int, ...]
    motion_model: str
    union_clusters: bool = False


def variants_for(modules: list[int], motion_models: list[str]) -> list[Variant]:
    groups = [
        ("backbone_early", (0, 1, 2, 3, 4), False),
        ("backbone_mid", (4, 5, 6, 7), False),
        ("backbone_deep", (8, 9, 10), False),
        ("neck_early", (12, 13, 15, 16), False),
        ("neck_deep", (18, 19, 21, 22), False),
        ("selected_union", (4, 5, 6, 7, 12), True),
    ]
    out: list[Variant] = []
    for motion in motion_models:
        suffix = "_kf" if motion == "constant_velocity" else "_nomotion"
        for module in modules:
            out.append(Variant(f"m{module}{suffix}", (module,), motion))
        for name, combo, union_clusters in groups:
            out.append(Variant(f"{name}{suffix}", combo, motion, union_clusters))
    return out


def det_from_dict(data: dict, default_source: str) -> TrackedDetection:
    return TrackedDetection(
        xyxy=np.array(data["xyxy"], dtype=np.float32),
        score=float(data["score"]),
        cls=None if data.get("class_id") is None else int(data["class_id"]),
        source=str(data.get("source", default_source)),
        module=None if data.get("module") is None else int(data["module"]),
        modules=tuple(int(m) for m in data.get("modules", [])),
    )


def nms(detections: Sequence[TrackedDetection], iou_threshold: float = 0.35, max_candidates: int = 50) -> list[TrackedDetection]:
    ranked = sorted(detections, key=lambda det: (det.score, bbox_area(det.xyxy)), reverse=True)
    if max_candidates > 0:
        ranked = ranked[:max_candidates]
    kept: list[TrackedDetection] = []
    for det in ranked:
        if all(bbox_iou(det.xyxy, existing.xyxy) < iou_threshold for existing in kept):
            kept.append(det)
    return kept


def suppress_peek_explained_by_yolo(
    peek: Sequence[TrackedDetection],
    yolo: Sequence[TrackedDetection],
    iou_threshold: float = 0.10,
    containment_threshold: float = 0.55,
) -> list[TrackedDetection]:
    kept = []
    for det in peek:
        redundant = False
        for yd in yolo:
            if bbox_iou(det.xyxy, yd.xyxy) >= iou_threshold or bbox_overlap_fraction(det.xyxy, yd.xyxy) >= containment_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(det)
    return kept


def gate_by_anchor_distance(
    peek: Sequence[TrackedDetection],
    yolo: Sequence[TrackedDetection],
    shadow: Sequence[TrackedDetection],
    tracks,
    height: int,
    width: int,
    distance_frac: float = 0.12,
) -> list[TrackedDetection]:
    anchors = [det.xyxy for det in yolo]
    anchors.extend(det.xyxy for det in shadow)
    for track in tracks:
        if getattr(track, "origin", track.source) == "yolo" and track.missed <= 8:
            anchors.append(track.xyxy)
    if not anchors:
        return []

    frame_diag = float(np.hypot(width, height))
    base_distance = distance_frac * frame_diag
    kept = []
    for det in peek:
        x1, y1, x2, y2 = det.xyxy.astype(np.float32)
        det_diag = float(np.hypot(x2 - x1, y2 - y1))
        for anchor in anchors:
            ax1, ay1, ax2, ay2 = anchor.astype(np.float32)
            anchor_diag = float(np.hypot(ax2 - ax1, ay2 - ay1))
            if bbox_center_distance(det.xyxy, anchor) <= base_distance + 0.5 * (det_diag + anchor_diag):
                kept.append(det)
                break
    return kept


def union_cluster_peek(
    detections: Sequence[TrackedDetection],
    height: int,
    width: int,
    iou_threshold: float = 0.10,
    center_frac: float = 0.35,
    min_area: int = 220,
    min_short_side: int = 12,
    max_area_fraction: float = 0.12,
) -> list[TrackedDetection]:
    candidates = [
        det
        for det in detections
        if bbox_area(det.xyxy) >= min_area
        and min(float(det.xyxy[2] - det.xyxy[0]), float(det.xyxy[3] - det.xyxy[1])) >= min_short_side
    ]
    if len(candidates) <= 1:
        return candidates
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, det in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            other = candidates[j]
            if bbox_iou(det.xyxy, other.xyxy) >= iou_threshold:
                union(i, j)
                continue
            center_limit = center_frac * max(
                float(np.hypot(det.xyxy[2] - det.xyxy[0], det.xyxy[3] - det.xyxy[1])),
                float(np.hypot(other.xyxy[2] - other.xyxy[0], other.xyxy[3] - other.xyxy[1])),
            )
            if bbox_center_distance(det.xyxy, other.xyxy) <= center_limit:
                union(i, j)

    groups: dict[int, list[TrackedDetection]] = defaultdict(list)
    for i, det in enumerate(candidates):
        groups[find(i)].append(det)

    merged = []
    for group in groups.values():
        boxes = np.stack([det.xyxy for det in group], axis=0)
        xyxy = np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()], dtype=np.float32)
        if bbox_area(xyxy) < min_area or bbox_area(xyxy) > max_area_fraction * height * width:
            continue
        if min(float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])) < min_short_side:
            continue
        modules = sorted({int(m) for det in group for m in (det.modules or (() if det.module is None else (det.module,)))})
        best = max(group, key=lambda det: det.score)
        merged.append(
            TrackedDetection(
                xyxy=xyxy,
                score=max(float(det.score) for det in group),
                cls=best.cls,
                source="peek",
                module=modules[0] if len(modules) == 1 else None,
                modules=tuple(modules),
            )
        )
    return sorted(merged, key=lambda det: (det.score, bbox_area(det.xyxy)), reverse=True)


def encode_track(track) -> dict:
    return {
        "id": int(track.track_id),
        "xyxy": [float(x) for x in track.xyxy],
        "score": float(track.score),
        "class_id": None if track.cls is None else int(track.cls),
        "source": track.source,
        "origin": getattr(track, "origin", track.source),
        "module": None if getattr(track, "module", None) is None else int(track.module),
        "modules": [int(m) for m in getattr(track, "modules", ())],
        "age": int(track.age),
        "hits": int(track.hits),
        "missed": int(track.missed),
    }


def run_cached_variant(cache_dir: Path, seq: str, variant: Variant, jsonl_path: Path) -> None:
    tracker = PEEKAssistedTracker(
        min_yolo_conf=0.10,
        motion_model=variant.motion_model,
        motion_process_noise=4.0,
        motion_measurement_noise=25.0,
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = cache_dir / "jsonl" / f"{seq}.jsonl"
    with source_path.open("r", encoding="utf-8") as src, jsonl_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            yolo = [det_from_dict(item, "yolo") for item in row["yolo_detections"]]
            shadow = [det_from_dict(item, "yolo") for item in row["shadow_yolo_detections"]]
            peek = [
                det_from_dict(item, "peek")
                for item in row["peek_detections"]
                if item.get("module") in variant.modules or set(item.get("modules", [])) & set(variant.modules)
            ]
            if variant.union_clusters:
                peek = union_cluster_peek(peek, int(row["height"]), int(row["width"]))
            else:
                peek = nms(peek)
            peek = suppress_peek_explained_by_yolo(peek, yolo)
            peek = gate_by_anchor_distance(peek, yolo, shadow, tracker.tracks, int(row["height"]), int(row["width"]))
            tracks = tracker.update(yolo, peek)
            dst.write(
                json.dumps(
                    {
                        "frame_index": int(row["frame_index"]),
                        "num_yolo_detections": len(yolo),
                        "num_shadow_yolo_detections": len(shadow),
                        "num_peek_detections": len(peek),
                        "tracks": [encode_track(track) for track in tracks],
                    }
                )
                + "\n"
            )


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def summarize(metrics_dir: Path, out_csv: Path) -> None:
    totals: dict[str, dict] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        variant = path.stem.split(".", 1)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        row = totals.setdefault(variant, {"variant": variant, "tp": 0, "fp": 0, "fn": 0, "id_switches": 0, "gt": 0, "pred": 0})
        for key in ("tp", "fp", "fn", "id_switches", "gt", "pred"):
            row[key] += int(data[key])
    rows = []
    for row in totals.values():
        p = row["tp"] / row["pred"] if row["pred"] else 0.0
        r = row["tp"] / row["gt"] if row["gt"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        row.update(
            {
                "precision": p,
                "recall": r,
                "f1": f1,
                "mota_like": 1.0 - (row["fn"] + row["fp"] + row["id_switches"]) / row["gt"] if row["gt"] else 0.0,
                "idf1_like": 2 * row["tp"] / (2 * row["tp"] + row["fp"] + row["fn"]) if (2 * row["tp"] + row["fp"] + row["fn"]) else 0.0,
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: (item["idf1_like"], item["mota_like"], item["recall"]), reverse=True)
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--mot-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modules", type=int, nargs="+", default=list(range(23)))
    parser.add_argument("--motion-models", nargs="+", choices=["none", "constant_velocity"], default=["none", "constant_velocity"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    variants = variants_for(args.modules, args.motion_models)
    variants = [variant for i, variant in enumerate(variants) if i % args.num_shards == args.shard_index]
    seqs = sorted(path.stem for path in (args.cache_dir / "jsonl").glob("*.jsonl"))
    metrics_dir = args.output_dir / "metrics"
    jsonl_dir = args.output_dir / "jsonl"
    for variant, seq in itertools.product(variants, seqs):
        metrics = metrics_dir / f"{variant.name}.{seq}.json"
        if metrics.exists():
            continue
        jsonl = jsonl_dir / f"{variant.name}.{seq}.jsonl"
        run_cached_variant(args.cache_dir, seq, variant, jsonl)
        run(
            [
                sys.executable,
                str(REPO / "tools/evaluate_mot_jsonl.py"),
                "--jsonl",
                str(jsonl),
                "--gt",
                str(args.mot_root / "train" / seq / "gt" / "gt.txt"),
                "--output",
                str(metrics),
            ],
            args.output_dir / "logs" / f"{variant.name}.{seq}_eval.log",
        )
        summarize(metrics_dir, args.output_dir / f"summary_shard{args.shard_index}.csv")
    summarize(metrics_dir, args.output_dir / f"summary_shard{args.shard_index}.csv")


if __name__ == "__main__":
    main()
