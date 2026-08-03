#!/usr/bin/env python3
"""Evaluate PEEK tracker JSONL against MOTChallenge ground truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import motmetrics as mm
except ImportError:  # pragma: no cover - fallback for bare environments.
    mm = None


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def read_gt(path: Path, include_distractors: bool = False) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            box = np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32)
            mark = int(float(parts[6])) if len(parts) > 6 else 1
            label = int(float(parts[7])) if len(parts) > 7 else 1
            visibility = float(parts[8]) if len(parts) > 8 else 1.0
            if mark == 0:
                continue
            if not include_distractors and label != 1:
                continue
            frames[frame].append({"id": track_id, "box": box, "visibility": visibility})
    return frames


def read_jsonl(path: Path, min_hits: int = 1, sources: set[str] | None = None) -> tuple[dict[int, list[dict]], set[int]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    seen_frames: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame_index"]) + 1
            seen_frames.add(frame)
            for track in row.get("tracks", []):
                if int(track.get("hits", 1)) < min_hits:
                    continue
                source = str(track.get("source", ""))
                origin = str(track.get("origin", source))
                if sources is not None and source not in sources and origin not in sources:
                    continue
                x1, y1, x2, y2 = [float(v) for v in track["xyxy"]]
                if x2 <= x1 or y2 <= y1:
                    continue
                frames[frame].append(
                    {
                        "id": int(track["id"]),
                        "box": np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32),
                        "source": source,
                        "origin": origin,
                    }
                )
    return frames, seen_frames


def evaluate(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    iou_threshold: float,
    frames_to_score: set[int] | None = None,
) -> dict:
    tp = fp = fn = idsw = 0
    pair_counts: Counter[tuple[int, int]] = Counter()
    last_match_by_gt: dict[int, int] = {}
    source_tp: Counter[str] = Counter()
    source_fp: Counter[str] = Counter()

    frame_ids = sorted(frames_to_score if frames_to_score is not None else (set(gt) | set(pred)))
    for frame in frame_ids:
        gts = gt.get(frame, [])
        preds = pred.get(frame, [])
        if not gts:
            fp += len(preds)
            for item in preds:
                source_fp[f"{item['origin']}:{item['source']}"] += 1
            continue
        if not preds:
            fn += len(gts)
            continue

        costs = np.ones((len(gts), len(preds)), dtype=np.float32)
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                costs[gi, pi] = 1.0 - iou_xywh(g["box"], p["box"])
        rows, cols = linear_sum_assignment(costs)

        matched_g: set[int] = set()
        matched_p: set[int] = set()
        for gi, pi in zip(rows, cols):
            overlap = 1.0 - float(costs[gi, pi])
            if overlap < iou_threshold:
                continue
            matched_g.add(int(gi))
            matched_p.add(int(pi))
            tp += 1
            gt_id = int(gts[gi]["id"])
            pred_id = int(preds[pi]["id"])
            pair_counts[(gt_id, pred_id)] += 1
            if gt_id in last_match_by_gt and last_match_by_gt[gt_id] != pred_id:
                idsw += 1
            last_match_by_gt[gt_id] = pred_id
            source_tp[f"{preds[pi]['origin']}:{preds[pi]['source']}"] += 1

        frame_fp = len(preds) - len(matched_p)
        frame_fn = len(gts) - len(matched_g)
        fp += frame_fp
        fn += frame_fn
        for pi, item in enumerate(preds):
            if pi not in matched_p:
                source_fp[f"{item['origin']}:{item['source']}"] += 1

    gt_total = tp + fn
    pred_total = tp + fp
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mota = 1.0 - (fn + fp + idsw) / gt_total if gt_total else 0.0
    idtp = sum(pair_counts.values())
    idfn = gt_total - idtp
    idfp = pred_total - idtp
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "id_switches": idsw,
        "gt": gt_total,
        "pred": pred_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mota_like": mota,
        "idf1_like": idf1,
        "source_tp": dict(source_tp),
        "source_fp": dict(source_fp),
    }


def evaluate_with_motmetrics(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    max_iou: float,
    frames_to_score: set[int] | None = None,
) -> dict:
    if mm is None:
        return {}

    acc = mm.MOTAccumulator(auto_id=True)
    frame_ids = sorted(frames_to_score if frames_to_score is not None else (set(gt) | set(pred)))
    for frame in frame_ids:
        gts = gt.get(frame, [])
        preds = pred.get(frame, [])
        gt_ids = [item["id"] for item in gts]
        pred_ids = [item["id"] for item in preds]
        gt_boxes = np.array([item["box"] for item in gts], dtype=np.float32)
        pred_boxes = np.array([item["box"] for item in preds], dtype=np.float32)
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=max_iou)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    metrics = [
        "num_frames",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_misses",
        "num_false_positives",
        "num_switches",
        "num_fragmentations",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "precision",
        "recall",
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
    ]
    summary = mh.compute(acc, metrics=metrics, name="overall")
    row = summary.loc["overall"].to_dict()
    out = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--sources", nargs="*", default=None, help="Optional source/origin filter, e.g. yolo peek predicted.")
    parser.add_argument("--include-distractors", action="store_true")
    parser.add_argument(
        "--score-jsonl-frames-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score only frames present in the JSONL. This keeps capped runs honest.",
    )
    args = parser.parse_args()

    gt = read_gt(args.gt, include_distractors=args.include_distractors)
    pred, seen_frames = read_jsonl(args.jsonl, min_hits=args.min_hits, sources=set(args.sources) if args.sources else None)
    frames_to_score = seen_frames if args.score_jsonl_frames_only else None
    metrics = evaluate(gt, pred, args.iou, frames_to_score=frames_to_score)
    motmetrics = evaluate_with_motmetrics(gt, pred, max_iou=args.iou, frames_to_score=frames_to_score)
    if motmetrics:
        metrics["motmetrics"] = motmetrics
    metrics.update({"jsonl": str(args.jsonl), "gt_file": str(args.gt), "iou": args.iou, "min_hits": args.min_hits})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
