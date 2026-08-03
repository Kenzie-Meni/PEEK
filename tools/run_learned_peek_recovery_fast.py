#!/usr/bin/env python3
"""Fast learned PEEK recovery sweep from cached MOT artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "third_party" / "ultralytics"))

from evaluate_mot_jsonl import evaluate, evaluate_with_motmetrics, read_gt as read_eval_gt  # noqa: E402
from run_learned_peek_recovery import (  # noqa: E402
    FEATURE_NAMES,
    cached_results,
    candidate_features,
    collect_sequence_examples,
    det_box,
    train_model,
    xyxy_to_dict,
)
from peek.tracking import bbox_iou  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def eval_item(track: dict) -> dict:
    return {
        "id": int(track["id"]),
        "box": xyxy_to_xywh(np.array(track["xyxy"], dtype=np.float32)),
        "source": str(track.get("source", "")),
        "origin": str(track.get("origin", track.get("source", ""))),
    }


def proba_vector(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    if probs.shape[1] == 1:
        return probs[:, 0].astype(np.float32)
    return probs[:, 1].astype(np.float32)


def score_sequence_once(
    cache_jsonl: Path,
    tracker_args: SimpleNamespace,
    model,
    modules: set[int],
    max_missed: int,
) -> tuple[dict[int, list[dict]], dict[int, list[dict]], set[int]]:
    tracker = BYTETracker(tracker_args, frame_rate=30)
    base_by_frame: dict[int, list[dict]] = {}
    refs: list[dict] = []
    feats: list[list[float]] = []
    seen_frames: set[int] = set()

    with cache_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame_index"]) + 1
            seen_frames.add(frame)
            tracks = tracker.update(cached_results(row))
            encoded = [xyxy_to_dict(item, "bytetrack", "bytetrack") for item in tracks]
            base_by_frame[frame] = encoded
            active_ids = {item["id"] for item in encoded}
            active_boxes = [det_box(item) for item in encoded]

            for lost in list(tracker.lost_stracks):
                track_id = int(lost.track_id)
                missed = int(tracker.frame_id - lost.end_frame)
                if missed <= 0 or missed > max_missed or track_id in active_ids:
                    continue
                pred_box = lost.xyxy.astype(np.float32)
                for peek in row.get("peek_detections", []):
                    module = peek.get("module")
                    if module not in modules:
                        continue
                    box = det_box(peek)
                    if max((bbox_iou(box, active) for active in active_boxes), default=0.0) >= 0.05:
                        continue
                    feats.append(candidate_features(peek, pred_box, active_boxes, row, missed))
                    refs.append(
                        {
                            "frame": frame,
                            "track_id": track_id,
                            "missed": missed,
                            "pred_box": pred_box,
                            "peek_box": box,
                            "peek": peek,
                            "class_id": int(getattr(lost, "cls", 0)),
                        }
                    )

    best_by_frame: dict[int, dict[int, dict]] = defaultdict(dict)
    if feats:
        probs = proba_vector(model, np.array(feats, dtype=np.float32))
        for ref, prob in zip(refs, probs):
            frame = int(ref["frame"])
            track_id = int(ref["track_id"])
            current = best_by_frame[frame].get(track_id)
            if current is None or float(prob) > current["prob"]:
                best_by_frame[frame][track_id] = {**ref, "prob": float(prob)}

    return base_by_frame, {frame: list(items.values()) for frame, items in best_by_frame.items()}, seen_frames


def build_variant_predictions(
    base_by_frame: dict[int, list[dict]],
    candidates_by_frame: dict[int, list[dict]],
    seen_frames: set[int],
    threshold: float,
    min_hits: int,
    output_box_mode: str,
) -> tuple[dict[int, list[dict]], Counter]:
    pending_hits: Counter[int] = Counter()
    stats: Counter = Counter()
    pred_by_frame: dict[int, list[dict]] = {}
    for frame in sorted(seen_frames):
        base_tracks = base_by_frame.get(frame, [])
        active_ids = {int(item["id"]) for item in base_tracks}
        tracks = list(base_tracks)
        for cand in candidates_by_frame.get(frame, []):
            track_id = int(cand["track_id"])
            if track_id in active_ids:
                continue
            if float(cand["prob"]) < threshold:
                pending_hits[track_id] = 0
                continue
            pending_hits[track_id] += 1
            stats["peek_supported_lost_tracks"] += 1
            if pending_hits[track_id] < min_hits:
                continue
            output_box = cand["peek_box"] if output_box_mode == "peek" else cand["pred_box"]
            tracks.append(
                {
                    "id": track_id,
                    "xyxy": [float(v) for v in output_box],
                    "score": float(cand["prob"]),
                    "class_id": int(cand["class_id"]),
                    "source": "peek_learned",
                    "origin": "bytetrack",
                    "module": cand["peek"].get("module"),
                    "modules": [int(cand["peek"]["module"])] if cand["peek"].get("module") is not None else [],
                    "age": 1,
                    "hits": int(pending_hits[track_id]),
                    "missed": int(cand["missed"]),
                }
            )
            stats["peek_recovery_outputs"] += 1
        for track_id in list(pending_hits):
            if track_id in active_ids:
                pending_hits[track_id] = 0
        pred_by_frame[frame] = [eval_item(item) for item in tracks]
    return pred_by_frame, stats


def write_jsonl(path: Path, base_by_frame: dict[int, list[dict]], candidates_by_frame: dict[int, list[dict]], seen_frames: set[int], threshold: float, min_hits: int, output_box_mode: str) -> None:
    pending_hits: Counter[int] = Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in sorted(seen_frames):
            base_tracks = base_by_frame.get(frame, [])
            active_ids = {int(item["id"]) for item in base_tracks}
            tracks = list(base_tracks)
            for cand in candidates_by_frame.get(frame, []):
                track_id = int(cand["track_id"])
                if track_id in active_ids:
                    continue
                if float(cand["prob"]) < threshold:
                    pending_hits[track_id] = 0
                    continue
                pending_hits[track_id] += 1
                if pending_hits[track_id] < min_hits:
                    continue
                output_box = cand["peek_box"] if output_box_mode == "peek" else cand["pred_box"]
                tracks.append(
                    {
                        "id": track_id,
                        "xyxy": [float(v) for v in output_box],
                        "score": float(cand["prob"]),
                        "class_id": int(cand["class_id"]),
                        "source": "peek_learned",
                        "origin": "bytetrack",
                        "module": cand["peek"].get("module"),
                        "modules": [int(cand["peek"]["module"])] if cand["peek"].get("module") is not None else [],
                        "age": 1,
                        "hits": int(pending_hits[track_id]),
                        "missed": int(cand["missed"]),
                    }
                )
            for track_id in list(pending_hits):
                if track_id in active_ids:
                    pending_hits[track_id] = 0
            handle.write(json.dumps({"frame_index": frame - 1, "tracks": tracks}) + "\n")


def summarize(rows: list[dict], output_dir: Path) -> list[dict]:
    totals: dict[str, Counter] = defaultdict(Counter)
    stats: dict[str, Counter] = defaultdict(Counter)
    seq_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        variant = row["variant"]
        seq_counts[variant].add(row["sequence"])
        for key in ("tp", "fp", "fn", "id_switches", "gt", "pred"):
            totals[variant][key] += int(row[key])
        stats[variant]["peek_supported_lost_tracks"] += int(row["peek_supported_lost_tracks"])
        stats[variant]["peek_recovery_outputs"] += int(row["peek_recovery_outputs"])
    out = []
    for variant, total in totals.items():
        precision = total["tp"] / total["pred"] if total["pred"] else 0.0
        recall = total["tp"] / total["gt"] if total["gt"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "variant": variant,
                "sequences": len(seq_counts[variant]),
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
                "peek_supported_lost_tracks": int(stats[variant]["peek_supported_lost_tracks"]),
                "peek_recovery_outputs": int(stats[variant]["peek_recovery_outputs"]),
            }
        )
    out.sort(key=lambda item: (item["mota_like"], item["f1"]), reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    (output_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


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
    parser.add_argument("--write-best-jsonl", action="store_true")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    args.mot_root = args.mot_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)

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
    label_gt = {}
    eval_gt = {}
    for seq in seqs:
        gt_path = args.mot_root / "train" / seq / "gt" / "gt.txt"
        from run_learned_peek_recovery import read_gt as read_label_gt

        label_gt[seq] = read_label_gt(gt_path)
        eval_gt[seq] = read_eval_gt(gt_path)

    examples = {}
    all_y = []
    for seq in seqs:
        x, y, meta = collect_sequence_examples(args.cache_dir / "jsonl" / f"{seq}.jsonl", label_gt[seq], tracker_args, modules, args.max_missed)
        examples[seq] = (np.array(x, dtype=np.float32), np.array(y, dtype=np.int64), meta)
        all_y.extend(y)
        print(f"{seq}: examples={len(y)} positives={int(np.sum(y))}", flush=True)

    y_all = np.array(all_y, dtype=np.int64)
    (args.output_dir / "candidate_report.json").write_text(
        json.dumps(
            {
                "examples": int(len(y_all)),
                "positives": int(y_all.sum()),
                "positive_rate": float(y_all.mean()) if len(y_all) else 0.0,
                "feature_names": FEATURE_NAMES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if len(np.unique(y_all)) < 2:
        raise RuntimeError("Need both positive and negative candidates to train.")

    rows = []
    for heldout in seqs:
        train_seqs = [seq for seq in seqs if seq != heldout and len(examples[seq][1])]
        x_train = np.concatenate([examples[seq][0] for seq in train_seqs], axis=0)
        y_train = np.concatenate([examples[seq][1] for seq in train_seqs], axis=0)
        x_test, y_test, _ = examples[heldout]
        if len(np.unique(y_train)) < 2:
            print(f"Skipping heldout={heldout}: train split has only one class", flush=True)
            continue
        model = train_model(args.model_kind, x_train, y_train)
        model_path = args.output_dir / "models" / f"{args.model_kind}_{heldout}.joblib"
        joblib.dump(model, model_path)
        if len(np.unique(y_test)) == 2:
            probs = proba_vector(model, x_test)
            print(
                f"heldout={heldout} examples={len(y_test)} positives={int(y_test.sum())} "
                f"auroc={roc_auc_score(y_test, probs):.4f} ap={average_precision_score(y_test, probs):.4f}",
                flush=True,
            )
        base_by_frame, candidates_by_frame, seen_frames = score_sequence_once(
            args.cache_dir / "jsonl" / f"{heldout}.jsonl",
            tracker_args,
            model,
            modules,
            args.max_missed,
        )
        print(
            f"heldout={heldout} candidate_tracks={sum(len(v) for v in candidates_by_frame.values())}",
            flush=True,
        )
        for threshold in args.thresholds:
            for min_hits in args.min_hits:
                for output_box in args.output_boxes:
                    variant = f"learned_{args.model_kind}_t{threshold:.2f}_h{min_hits}_{output_box}"
                    pred_by_frame, stats = build_variant_predictions(
                        base_by_frame,
                        candidates_by_frame,
                        seen_frames,
                        threshold,
                        min_hits,
                        output_box,
                    )
                    metrics = evaluate(eval_gt[heldout], pred_by_frame, 0.5, frames_to_score=seen_frames)
                    real_mot = evaluate_with_motmetrics(eval_gt[heldout], pred_by_frame, 0.5, frames_to_score=seen_frames)
                    metric_row = {
                        "variant": variant,
                        "sequence": heldout,
                        **{key: metrics[key] for key in ("tp", "fp", "fn", "id_switches", "gt", "pred", "precision", "recall", "f1", "mota_like", "idf1_like")},
                        "motmetrics_mota": real_mot.get("mota"),
                        "motmetrics_idf1": real_mot.get("idf1"),
                        "motmetrics_motp": real_mot.get("motp"),
                        "motmetrics_switches": real_mot.get("num_switches"),
                        "motmetrics_fragmentations": real_mot.get("num_fragmentations"),
                        "peek_supported_lost_tracks": int(stats["peek_supported_lost_tracks"]),
                        "peek_recovery_outputs": int(stats["peek_recovery_outputs"]),
                    }
                    rows.append(metric_row)
                    metric_path = args.output_dir / "metrics" / f"{variant}.{heldout}.json"
                    metric_path.parent.mkdir(parents=True, exist_ok=True)
                    metric_path.write_text(json.dumps(metric_row, indent=2) + "\n", encoding="utf-8")
        summarize(rows, args.output_dir)

    summary = summarize(rows, args.output_dir)
    if args.write_best_jsonl and summary:
        best = summary[0]["variant"]
        _, model_kind, thresh_token, hit_token, output_box = best.split("_", 4)
        threshold = float(thresh_token[1:])
        min_hits = int(hit_token[1:])
        for heldout in seqs:
            model = joblib.load(args.output_dir / "models" / f"{args.model_kind}_{heldout}.joblib")
            base_by_frame, candidates_by_frame, seen_frames = score_sequence_once(
                args.cache_dir / "jsonl" / f"{heldout}.jsonl",
                tracker_args,
                model,
                modules,
                args.max_missed,
            )
            write_jsonl(args.output_dir / "jsonl" / f"{best}.{heldout}.jsonl", base_by_frame, candidates_by_frame, seen_frames, threshold, min_hits, output_box)
    print(json.dumps(summary[:15], indent=2), flush=True)


if __name__ == "__main__":
    main()
