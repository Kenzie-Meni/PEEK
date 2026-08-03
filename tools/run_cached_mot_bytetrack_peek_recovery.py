#!/usr/bin/env python3
"""Run ByteTrack plus tightly gated PEEK recovery from cached MOT artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ultralytics"))

from peek.tracking import bbox_area, bbox_center_distance, bbox_iou, bbox_overlap_fraction  # noqa: E402
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


@dataclass(frozen=True)
class RecoveryVariant:
    name: str
    modules: tuple[int, ...]
    min_hits: int
    max_missed: int
    min_peek_score: float
    min_iou: float
    max_center_frac: float
    max_area_ratio: float
    max_aspect_ratio_change: float
    output_box: str = "motion"


def cached_results(row: dict, key: str = "all_yolo_detections") -> CachedResults:
    detections = row.get(key, [])
    if not detections:
        return CachedResults(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
    return CachedResults(
        np.array([det["xyxy"] for det in detections], dtype=np.float32),
        np.array([det["score"] for det in detections], dtype=np.float32),
        np.array([0 if det.get("class_id") is None else det["class_id"] for det in detections], dtype=np.float32),
    )


def xyxy_to_dict(item: Sequence[float], source: str, origin: str) -> dict:
    x1, y1, x2, y2, tid, score, cls, *_ = [float(v) for v in item]
    return {
        "id": int(tid),
        "xyxy": [x1, y1, x2, y2],
        "score": float(score),
        "class_id": int(cls),
        "source": source,
        "origin": origin,
        "module": None,
        "modules": [],
        "age": 1,
        "hits": 1,
        "missed": 0,
    }


def det_box(det: dict) -> np.ndarray:
    return np.array(det["xyxy"], dtype=np.float32)


def box_aspect(box: np.ndarray) -> float:
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))
    return w / h


def recoverable_peek(
    peek: dict,
    pred_box: np.ndarray,
    frame_diag: float,
    variant: RecoveryVariant,
) -> tuple[bool, float]:
    if peek.get("module") not in variant.modules and not (set(peek.get("modules", [])) & set(variant.modules)):
        return False, 0.0
    if float(peek.get("score", 0.0)) < variant.min_peek_score:
        return False, 0.0
    box = det_box(peek)
    if box[2] <= box[0] or box[3] <= box[1]:
        return False, 0.0

    iou = bbox_iou(box, pred_box)
    center_distance = bbox_center_distance(box, pred_box)
    pred_diag = float(np.hypot(pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]))
    max_center = variant.max_center_frac * max(pred_diag, 0.08 * frame_diag)
    if iou < variant.min_iou and center_distance > max_center:
        return False, 0.0

    area_ratio = bbox_area(box) / max(1.0, bbox_area(pred_box))
    if area_ratio > variant.max_area_ratio or area_ratio < 1.0 / variant.max_area_ratio:
        return False, 0.0

    aspect_ratio = max(box_aspect(box) / box_aspect(pred_box), box_aspect(pred_box) / box_aspect(box))
    if aspect_ratio > variant.max_aspect_ratio_change:
        return False, 0.0

    score = float(peek.get("score", 0.0)) + 0.35 * iou - 0.15 * (center_distance / max(frame_diag, 1.0))
    return True, score


def choose_recovery(
    peeks: Sequence[dict],
    pred_box: np.ndarray,
    active_boxes: Sequence[np.ndarray],
    frame_diag: float,
    variant: RecoveryVariant,
) -> tuple[dict | None, float]:
    best = None
    best_score = -1e9
    for peek in peeks:
        box = det_box(peek)
        if any(bbox_iou(box, active) >= 0.05 or bbox_overlap_fraction(box, active) >= 0.40 for active in active_boxes):
            continue
        ok, score = recoverable_peek(peek, pred_box, frame_diag, variant)
        if ok and score > best_score:
            best = peek
            best_score = score
    return best, best_score


def run_sequence(cache_jsonl: Path, out_jsonl: Path, tracker_args: SimpleNamespace, variant: RecoveryVariant) -> Counter:
    tracker = BYTETracker(tracker_args, frame_rate=30)
    pending_hits: Counter[int] = Counter()
    stats: Counter = Counter()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with cache_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            tracks = tracker.update(cached_results(row))
            encoded = [xyxy_to_dict(item, "bytetrack", "bytetrack") for item in tracks]
            active_ids = {item["id"] for item in encoded}
            active_boxes = [det_box(item) for item in encoded]
            frame_diag = float(np.hypot(float(row["width"]), float(row["height"])))

            recovered = []
            for lost in list(tracker.lost_stracks):
                if int(tracker.frame_id - lost.end_frame) > variant.max_missed:
                    continue
                track_id = int(lost.track_id)
                if track_id in active_ids:
                    continue
                pred_box = lost.xyxy.astype(np.float32)
                chosen, score = choose_recovery(row.get("peek_detections", []), pred_box, active_boxes, frame_diag, variant)
                if chosen is None:
                    pending_hits[track_id] = 0
                    continue
                pending_hits[track_id] += 1
                stats["peek_supported_lost_tracks"] += 1
                if pending_hits[track_id] < variant.min_hits:
                    continue

                output_box = det_box(chosen) if variant.output_box == "peek" else pred_box
                recovered.append(
                    {
                        "id": track_id,
                        "xyxy": [float(v) for v in output_box],
                        "score": float(min(1.0, max(0.0, score))),
                        "class_id": int(getattr(lost, "cls", 0)),
                        "source": "peek_recovery",
                        "origin": "bytetrack",
                        "module": chosen.get("module"),
                        "modules": [int(m) for m in (chosen.get("modules") or ([chosen["module"]] if chosen.get("module") is not None else []))],
                        "age": 1,
                        "hits": int(pending_hits[track_id]),
                        "missed": int(tracker.frame_id - lost.end_frame),
                    }
                )
                stats["peek_recovery_outputs"] += 1

            for track_id in list(pending_hits):
                if track_id in active_ids:
                    pending_hits[track_id] = 0
            dst.write(json.dumps({"frame_index": int(row["frame_index"]), "tracks": encoded + recovered}) + "\n")
    return stats


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def summarize(metrics_dir: Path, output_dir: Path, stats_by_variant: dict[str, Counter]) -> list[dict]:
    totals: dict[str, Counter] = defaultdict(Counter)
    for path in metrics_dir.glob("*.json"):
        variant = path.stem.split(".", 1)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("tp", "fp", "fn", "id_switches", "gt", "pred"):
            totals[variant][key] += int(data[key])
    rows = []
    for variant, item in totals.items():
        precision = item["tp"] / item["pred"] if item["pred"] else 0.0
        recall = item["tp"] / item["gt"] if item["gt"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "variant": variant,
            "tp": int(item["tp"]),
            "fp": int(item["fp"]),
            "fn": int(item["fn"]),
            "id_switches": int(item["id_switches"]),
            "gt": int(item["gt"]),
            "pred": int(item["pred"]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mota_like": 1.0 - (item["fn"] + item["fp"] + item["id_switches"]) / item["gt"] if item["gt"] else 0.0,
            "idf1_like": 2 * item["tp"] / (2 * item["tp"] + item["fp"] + item["fn"]) if (2 * item["tp"] + item["fp"] + item["fn"]) else 0.0,
            "peek_supported_lost_tracks": int(stats_by_variant.get(variant, Counter())["peek_supported_lost_tracks"]),
            "peek_recovery_outputs": int(stats_by_variant.get(variant, Counter())["peek_recovery_outputs"]),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["mota_like"], row["f1"], row["precision"]), reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def default_variants() -> list[RecoveryVariant]:
    base = {
        "min_hits": 2,
        "max_missed": 8,
        "min_peek_score": 0.30,
        "min_iou": 0.02,
        "max_center_frac": 0.65,
        "max_area_ratio": 3.0,
        "max_aspect_ratio_change": 2.5,
    }
    strict = {**base, "min_peek_score": 0.42, "min_iou": 0.05, "max_center_frac": 0.45, "max_area_ratio": 2.0}
    variants = []
    for name, modules in {
        "m17": (17,),
        "m20": (20,),
        "m21": (21,),
        "m17_20": (17, 20),
        "m17_18": (17, 18),
        "m20_21": (20, 21),
    }.items():
        variants.append(RecoveryVariant(f"bt_peek_{name}_gated", modules, **base))
        variants.append(RecoveryVariant(f"bt_peek_{name}_strict", modules, **strict))
    return variants


def aggressive_variants() -> list[RecoveryVariant]:
    variants = []
    module_sets = {
        "m17": (17,),
        "m20": (20,),
        "m21": (21,),
        "m17_18": (17, 18),
        "m17_20": (17, 20),
        "m20_21": (20, 21),
        "m17_18_20": (17, 18, 20),
        "m17_20_21": (17, 20, 21),
        "m17_18_20_21": (17, 18, 20, 21),
    }
    settings = {
        "loose_h1": {
            "min_hits": 1,
            "max_missed": 12,
            "min_peek_score": 0.18,
            "min_iou": 0.0,
            "max_center_frac": 0.95,
            "max_area_ratio": 5.0,
            "max_aspect_ratio_change": 4.0,
        },
        "loose_h2": {
            "min_hits": 2,
            "max_missed": 12,
            "min_peek_score": 0.18,
            "min_iou": 0.0,
            "max_center_frac": 0.95,
            "max_area_ratio": 5.0,
            "max_aspect_ratio_change": 4.0,
        },
        "mid_h1": {
            "min_hits": 1,
            "max_missed": 10,
            "min_peek_score": 0.24,
            "min_iou": 0.01,
            "max_center_frac": 0.75,
            "max_area_ratio": 4.0,
            "max_aspect_ratio_change": 3.0,
        },
        "mid_h2": {
            "min_hits": 2,
            "max_missed": 10,
            "min_peek_score": 0.24,
            "min_iou": 0.01,
            "max_center_frac": 0.75,
            "max_area_ratio": 4.0,
            "max_aspect_ratio_change": 3.0,
        },
    }
    for module_name, modules in module_sets.items():
        for setting_name, kwargs in settings.items():
            for output_box in ("motion", "peek"):
                variants.append(
                    RecoveryVariant(
                        f"bt_peek_{module_name}_{setting_name}_{output_box}",
                        modules,
                        output_box=output_box,
                        **kwargs,
                    )
                )
    return variants


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
    parser.add_argument("--variant-index", type=int, default=-1, help="Run only this default variant index.")
    parser.add_argument("--num-variant-shards", type=int, default=1)
    parser.add_argument("--variant-shard-index", type=int, default=0)
    parser.add_argument("--variant-set", choices=["default", "aggressive"], default="default")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    args.mot_root = args.mot_root.resolve()
    args.output_dir = args.output_dir.resolve()

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
    stats_by_variant: dict[str, Counter] = defaultdict(Counter)
    variants = default_variants() if args.variant_set == "default" else aggressive_variants()
    if args.variant_index >= 0:
        variants = [variants[args.variant_index]]
    elif args.num_variant_shards > 1:
        variants = [variant for index, variant in enumerate(variants) if index % args.num_variant_shards == args.variant_shard_index]
    for variant in variants:
        for cache_jsonl in sorted((args.cache_dir / "jsonl").glob("*.jsonl")):
            seq = cache_jsonl.stem
            metrics = metrics_dir / f"{variant.name}.{seq}.json"
            if metrics.exists():
                continue
            out_jsonl = jsonl_dir / f"{variant.name}.{seq}.jsonl"
            stats = run_sequence(cache_jsonl, out_jsonl, tracker_args, variant)
            stats_by_variant[variant.name].update(stats)
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
                args.output_dir / "logs" / f"{variant.name}.{seq}_eval.log",
            )
        summarize(metrics_dir, args.output_dir, stats_by_variant)
    rows = summarize(metrics_dir, args.output_dir, stats_by_variant)
    print(json.dumps(rows[:12], indent=2))


if __name__ == "__main__":
    main()
