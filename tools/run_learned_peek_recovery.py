#!/usr/bin/env python3
"""Train/evaluate a learned PEEK recovery selector on cached MOT artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ultralytics"))

from peek.tracking import bbox_area, bbox_center_distance, bbox_iou, bbox_overlap_fraction  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402


FEATURE_NAMES = [
    "module",
    "peek_score",
    "missed",
    "peek_pred_iou",
    "peek_pred_center_frac",
    "peek_pred_area_ratio_log",
    "peek_pred_aspect_ratio_log",
    "peek_area_frac",
    "pred_area_frac",
    "max_active_iou",
    "max_active_containment",
    "best_shadow_iou",
    "best_shadow_score",
    "best_yolo_iou",
    "best_yolo_score",
]


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


def cached_results(row: dict, key: str = "all_yolo_detections") -> CachedResults:
    detections = row.get(key, [])
    if not detections:
        return CachedResults(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
    return CachedResults(
        np.array([det["xyxy"] for det in detections], dtype=np.float32),
        np.array([det["score"] for det in detections], dtype=np.float32),
        np.array([0 if det.get("class_id") is None else det["class_id"] for det in detections], dtype=np.float32),
    )


def xyxy_to_dict(item, source: str, origin: str) -> dict:
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


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box.astype(np.float32)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_aspect(box: np.ndarray) -> float:
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))
    return w / h


def read_gt(path: Path) -> dict[int, list[np.ndarray]]:
    frames: dict[int, list[np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            frame = int(float(parts[0]))
            mark = int(float(parts[6])) if len(parts) > 6 else 1
            label = int(float(parts[7])) if len(parts) > 7 else 1
            if mark == 0 or label != 1:
                continue
            frames.setdefault(frame, []).append(
                np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32)
            )
    return frames


def match_active_to_gt(active_boxes: list[np.ndarray], gt_boxes_xywh: list[np.ndarray]) -> set[int]:
    if not active_boxes or not gt_boxes_xywh:
        return set()
    costs = np.ones((len(gt_boxes_xywh), len(active_boxes)), dtype=np.float32)
    for gi, gt in enumerate(gt_boxes_xywh):
        gt_xyxy = xywh_to_xyxy(gt)
        for ai, box in enumerate(active_boxes):
            costs[gi, ai] = 1.0 - bbox_iou(gt_xyxy, box)
    rows, cols = linear_sum_assignment(costs)
    matched = set()
    for gi, ai in zip(rows, cols):
        if 1.0 - float(costs[gi, ai]) >= 0.5:
            matched.add(int(gi))
    return matched


def best_iou_score(box: np.ndarray, detections: list[dict]) -> tuple[float, float]:
    best_iou = 0.0
    best_score = 0.0
    for det in detections:
        iou = bbox_iou(box, det_box(det))
        if iou > best_iou:
            best_iou = iou
            best_score = float(det.get("score", 0.0))
    return best_iou, best_score


def candidate_features(
    peek: dict,
    pred_box: np.ndarray,
    active_boxes: list[np.ndarray],
    row: dict,
    missed: int,
) -> list[float]:
    box = det_box(peek)
    frame_area = max(1.0, float(row["height"]) * float(row["width"]))
    frame_diag = float(np.hypot(float(row["height"]), float(row["width"])))
    pred_diag = max(1.0, float(np.hypot(pred_box[2] - pred_box[0], pred_box[3] - pred_box[1])))
    max_active_iou = max((bbox_iou(box, active) for active in active_boxes), default=0.0)
    max_active_containment = max((bbox_overlap_fraction(box, active) for active in active_boxes), default=0.0)
    shadow_iou, shadow_score = best_iou_score(box, row.get("shadow_yolo_detections", []))
    yolo_iou, yolo_score = best_iou_score(box, row.get("yolo_detections", []))
    area_ratio = bbox_area(box) / max(1.0, bbox_area(pred_box))
    aspect_ratio = max(box_aspect(box) / box_aspect(pred_box), box_aspect(pred_box) / box_aspect(box))
    return [
        float(peek.get("module", -1) if peek.get("module") is not None else -1),
        float(peek.get("score", 0.0)),
        float(missed),
        bbox_iou(box, pred_box),
        bbox_center_distance(box, pred_box) / pred_diag,
        float(np.log(max(area_ratio, 1e-6))),
        float(np.log(max(aspect_ratio, 1e-6))),
        bbox_area(box) / frame_area,
        bbox_area(pred_box) / frame_area,
        max_active_iou,
        max_active_containment,
        shadow_iou,
        shadow_score,
        yolo_iou,
        yolo_score,
    ]


def candidate_label(
    output_box: np.ndarray,
    active_boxes: list[np.ndarray],
    gt_boxes_xywh: list[np.ndarray],
) -> int:
    if not gt_boxes_xywh:
        return 0
    active_matched = match_active_to_gt(active_boxes, gt_boxes_xywh)
    for gi, gt in enumerate(gt_boxes_xywh):
        if gi in active_matched:
            continue
        if bbox_iou(output_box, xywh_to_xyxy(gt)) >= 0.5:
            return 1
    return 0


def collect_sequence_examples(
    cache_jsonl: Path,
    gt: dict[int, list[np.ndarray]],
    tracker_args: SimpleNamespace,
    modules: set[int],
    max_missed: int,
) -> tuple[list[list[float]], list[int], list[dict]]:
    tracker = BYTETracker(tracker_args, frame_rate=30)
    features: list[list[float]] = []
    labels: list[int] = []
    meta: list[dict] = []
    with cache_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tracks = tracker.update(cached_results(row))
            active_boxes = [np.array(item[:4], dtype=np.float32) for item in tracks]
            frame_id = int(row["frame_index"]) + 1
            gt_boxes = gt.get(frame_id, [])
            for lost in list(tracker.lost_stracks):
                missed = int(tracker.frame_id - lost.end_frame)
                if missed <= 0 or missed > max_missed:
                    continue
                pred_box = lost.xyxy.astype(np.float32)
                for peek in row.get("peek_detections", []):
                    module = peek.get("module")
                    if module not in modules:
                        continue
                    box = det_box(peek)
                    if max((bbox_iou(box, active) for active in active_boxes), default=0.0) >= 0.05:
                        continue
                    features.append(candidate_features(peek, pred_box, active_boxes, row, missed))
                    labels.append(candidate_label(pred_box, active_boxes, gt_boxes))
                    meta.append({"sequence": cache_jsonl.stem, "frame_index": int(row["frame_index"]), "track_id": int(lost.track_id)})
    return features, labels, meta


def train_model(kind: str, x: np.ndarray, y: np.ndarray):
    positive = max(1, int(y.sum()))
    negative = max(1, int(len(y) - y.sum()))
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=7,
        )
        return make_pipeline(SimpleImputer(), clf).fit(x, y)
    if kind == "xgb":
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negative / positive,
            n_jobs=-1,
            random_state=7,
            tree_method="hist",
        )
        return make_pipeline(SimpleImputer(), clf).fit(x, y)
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=7,
    )
    return make_pipeline(SimpleImputer(), StandardScaler(), clf).fit(x, y)


def predict_proba(model, x: list[float]) -> float:
    proba = model.predict_proba(np.array([x], dtype=np.float32))
    if proba.shape[1] == 1:
        return float(proba[0, 0])
    return float(proba[0, 1])


def replay_sequence(
    cache_jsonl: Path,
    out_jsonl: Path,
    tracker_args: SimpleNamespace,
    model,
    modules: set[int],
    threshold: float,
    max_missed: int,
    min_hits: int,
    output_box_mode: str,
) -> Counter:
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
            recovered = []
            for lost in list(tracker.lost_stracks):
                missed = int(tracker.frame_id - lost.end_frame)
                if missed <= 0 or missed > max_missed or int(lost.track_id) in active_ids:
                    continue
                pred_box = lost.xyxy.astype(np.float32)
                best_peek = None
                best_prob = -1.0
                for peek in row.get("peek_detections", []):
                    module = peek.get("module")
                    if module not in modules:
                        continue
                    box = det_box(peek)
                    if max((bbox_iou(box, active) for active in active_boxes), default=0.0) >= 0.05:
                        continue
                    prob = predict_proba(model, candidate_features(peek, pred_box, active_boxes, row, missed))
                    if prob > best_prob:
                        best_prob = prob
                        best_peek = peek
                if best_peek is None or best_prob < threshold:
                    pending_hits[int(lost.track_id)] = 0
                    continue
                pending_hits[int(lost.track_id)] += 1
                stats["peek_supported_lost_tracks"] += 1
                if pending_hits[int(lost.track_id)] < min_hits:
                    continue
                output_box = det_box(best_peek) if output_box_mode == "peek" else pred_box
                recovered.append(
                    {
                        "id": int(lost.track_id),
                        "xyxy": [float(v) for v in output_box],
                        "score": best_prob,
                        "class_id": int(getattr(lost, "cls", 0)),
                        "source": "peek_learned",
                        "origin": "bytetrack",
                        "module": best_peek.get("module"),
                        "modules": [int(best_peek["module"])] if best_peek.get("module") is not None else [],
                        "age": 1,
                        "hits": int(pending_hits[int(lost.track_id)]),
                        "missed": missed,
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


def summarize(metrics_dir: Path, stats: dict[str, Counter], output: Path) -> list[dict]:
    totals: dict[str, Counter] = {}
    for path in metrics_dir.glob("*.json"):
        variant = path.name.rsplit(".", 2)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        total = totals.setdefault(variant, Counter())
        for key in ("tp", "fp", "fn", "id_switches", "gt", "pred"):
            total[key] += int(data[key])
    rows = []
    for variant, total in totals.items():
        precision = total["tp"] / total["pred"] if total["pred"] else 0.0
        recall = total["tp"] / total["gt"] if total["gt"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "variant": variant,
                "tp": int(total["tp"]),
                "fp": int(total["fp"]),
                "fn": int(total["fn"]),
                "id_switches": int(total["id_switches"]),
                "gt": int(total["gt"]),
                "pred": int(total["pred"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mota_like": 1.0 - (total["fn"] + total["fp"] + total["id_switches"]) / total["gt"] if total["gt"] else 0.0,
                "idf1_like": 2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"]) if (2 * total["tp"] + total["fp"] + total["fn"]) else 0.0,
                "peek_supported_lost_tracks": int(stats.get(variant, Counter())["peek_supported_lost_tracks"]),
                "peek_recovery_outputs": int(stats.get(variant, Counter())["peek_recovery_outputs"]),
            }
        )
    rows.sort(key=lambda r: (r["mota_like"], r["f1"], r["precision"]), reverse=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--mot-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modules", type=int, nargs="+", default=[17, 18, 20, 21])
    parser.add_argument("--model-kind", choices=["hgb", "rf", "xgb"], default="hgb")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--max-missed", type=int, default=12)
    parser.add_argument("--min-hits", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output-boxes", nargs="+", choices=["motion", "peek"], default=["motion"])
    args = parser.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    args.mot_root = args.mot_root.resolve()
    args.output_dir = args.output_dir.resolve()

    tracker_args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    modules = set(args.modules)
    seqs = sorted(path.stem for path in (args.cache_dir / "jsonl").glob("*.jsonl"))
    gt_by_seq = {seq: read_gt(args.mot_root / "train" / seq / "gt" / "gt.txt") for seq in seqs}

    examples = {}
    all_x, all_y, all_seq = [], [], []
    for seq in seqs:
        x, y, meta = collect_sequence_examples(args.cache_dir / "jsonl" / f"{seq}.jsonl", gt_by_seq[seq], tracker_args, modules, args.max_missed)
        examples[seq] = (np.array(x, dtype=np.float32), np.array(y, dtype=np.int64), meta)
        all_x.extend(x)
        all_y.extend(y)
        all_seq.extend([seq] * len(y))
        print(f"{seq}: examples={len(y)} positives={int(np.sum(y))}")

    y_all = np.array(all_y, dtype=np.int64)
    x_all = np.array(all_x, dtype=np.float32)
    train_report = {
        "examples": int(len(y_all)),
        "positives": int(y_all.sum()),
        "positive_rate": float(y_all.mean()) if len(y_all) else 0.0,
        "feature_names": FEATURE_NAMES,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_report.json").write_text(json.dumps(train_report, indent=2) + "\n", encoding="utf-8")
    if len(y_all) == 0:
        raise RuntimeError(f"No learned PEEK recovery candidates found for modules {sorted(modules)}")
    if len(np.unique(y_all)) < 2:
        raise RuntimeError(
            f"Learned PEEK recovery candidates have only one class for modules {sorted(modules)}: "
            f"examples={len(y_all)} positives={int(y_all.sum())}"
        )

    stats: dict[str, Counter] = {}
    for heldout in seqs:
        train_seqs = [seq for seq in seqs if seq != heldout]
        x_train = np.concatenate([examples[seq][0] for seq in train_seqs if len(examples[seq][1])], axis=0)
        y_train = np.concatenate([examples[seq][1] for seq in train_seqs if len(examples[seq][1])], axis=0)
        x_test, y_test, _ = examples[heldout]
        if len(np.unique(y_train)) < 2:
            print(f"Skipping heldout={heldout}: train split has only one class")
            continue
        model = train_model(args.model_kind, x_train, y_train)
        if len(np.unique(y_test)) == 2:
            probs = model.predict_proba(x_test)[:, 1]
            print(
                f"heldout={heldout} examples={len(y_test)} positives={int(y_test.sum())} "
                f"auroc={roc_auc_score(y_test, probs):.4f} ap={average_precision_score(y_test, probs):.4f}"
            )
        joblib.dump(model, args.output_dir / "models" / f"{args.model_kind}_{heldout}.joblib")
        for threshold in args.thresholds:
            for min_hits in args.min_hits:
                for output_box in args.output_boxes:
                    variant = f"learned_{args.model_kind}_t{threshold:.2f}_h{min_hits}_{output_box}"
                    jsonl = args.output_dir / "jsonl" / f"{variant}.{heldout}.jsonl"
                    metric = args.output_dir / "metrics" / f"{variant}.{heldout}.json"
                    if metric.exists():
                        continue
                    run_stats = replay_sequence(
                        args.cache_dir / "jsonl" / f"{heldout}.jsonl",
                        jsonl,
                        tracker_args,
                        model,
                        modules,
                        threshold,
                        args.max_missed,
                        min_hits,
                        output_box,
                    )
                    stats.setdefault(variant, Counter()).update(run_stats)
                    run(
                        [
                            sys.executable,
                            str(REPO / "tools/evaluate_mot_jsonl.py"),
                            "--jsonl",
                            str(jsonl),
                            "--gt",
                            str(args.mot_root / "train" / heldout / "gt" / "gt.txt"),
                            "--output",
                            str(metric),
                        ],
                        args.output_dir / "logs" / f"{variant}.{heldout}_eval.log",
                    )
        summarize(args.output_dir / "metrics", stats, args.output_dir / "summary.csv")

    rows = summarize(args.output_dir / "metrics", stats, args.output_dir / "summary.csv")
    print(json.dumps(rows[:15], indent=2))


if __name__ == "__main__":
    main()
