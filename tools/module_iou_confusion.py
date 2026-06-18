#!/usr/bin/env python3
"""Compute frame-wise IoU overlap matrices between PEEK proposal modules."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((len(values) - 1) * q / 100)
    return values[idx]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modules", nargs="+", type=int, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = [str(m) for m in args.modules]
    rows = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines()]

    totals = Counter()
    frames_present = Counter()
    best_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    matched_by_pair = Counter()

    for row in rows:
        by_module: dict[str, list[dict]] = defaultdict(list)
        for det in row.get("peek_detections", []):
            module = det.get("module")
            if module is None or int(module) not in args.modules:
                continue
            by_module[str(module)].append(det)

        for module, detections in by_module.items():
            totals[module] += len(detections)
            frames_present[module] += 1

        for src in modules:
            for det in by_module.get(src, []):
                for dst in modules:
                    if src == dst:
                        continue
                    overlaps = [iou(det["xyxy"], other["xyxy"]) for other in by_module.get(dst, [])]
                    best = max(overlaps) if overlaps else 0.0
                    best_by_pair[(src, dst)].append(best)
                    if best >= args.iou_threshold:
                        matched_by_pair[(src, dst)] += 1

    directional = {}
    mean_best_iou_matrix = {}
    match_fraction_matrix = {}
    symmetric = {}
    for src in modules:
        mean_best_iou_matrix[src] = {}
        match_fraction_matrix[src] = {}
        for dst in modules:
            if src == dst:
                mean_best_iou_matrix[src][dst] = 1.0
                match_fraction_matrix[src][dst] = 1.0
                continue
            values = best_by_pair.get((src, dst), [])
            denom = len(values)
            directional[f"{src}->{dst}"] = {
                "source_boxes": denom,
                "matched_boxes": matched_by_pair[(src, dst)],
                "match_fraction": matched_by_pair[(src, dst)] / denom if denom else 0.0,
                "mean_best_iou": mean(values),
                "p50_best_iou": pct(values, 50),
                "p90_best_iou": pct(values, 90),
            }
            mean_best_iou_matrix[src][dst] = mean(values)
            match_fraction_matrix[src][dst] = matched_by_pair[(src, dst)] / denom if denom else 0.0

    for i, a in enumerate(modules):
        for b in modules[i + 1 :]:
            ab = directional.get(f"{a}->{b}", {})
            ba = directional.get(f"{b}->{a}", {})
            symmetric[f"{a}-{b}"] = {
                "avg_match_fraction": mean([ab.get("match_fraction", 0.0), ba.get("match_fraction", 0.0)]),
                "max_match_fraction": max(ab.get("match_fraction", 0.0), ba.get("match_fraction", 0.0)),
                "avg_mean_best_iou": mean([ab.get("mean_best_iou", 0.0), ba.get("mean_best_iou", 0.0)]),
                "source_boxes": [ab.get("source_boxes", 0), ba.get("source_boxes", 0)],
            }

    out = {
        "jsonl": args.jsonl,
        "iou_threshold": args.iou_threshold,
        "modules": modules,
        "module_totals": {
            module: {"boxes": totals[module], "frames_present": frames_present[module]}
            for module in modules
        },
        "match_fraction_matrix": match_fraction_matrix,
        "mean_best_iou_matrix": mean_best_iou_matrix,
        "directional": directional,
        "symmetric_pairs_ranked": dict(
            sorted(
                symmetric.items(),
                key=lambda kv: (kv[1]["avg_match_fraction"], kv[1]["avg_mean_best_iou"]),
                reverse=True,
            )
        ),
    }
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
