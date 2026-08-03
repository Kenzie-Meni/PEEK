#!/usr/bin/env python3
"""Write JSONLs for one saved fast learned PEEK recovery variant."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib

from run_learned_peek_recovery_fast import score_sequence_once, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-kind", required=True, choices=["hgb", "rf", "xgb"])
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--min-hits", required=True, type=int)
    parser.add_argument("--output-box", required=True, choices=["motion", "peek"])
    parser.add_argument("--modules", type=int, nargs="+", default=[17, 18, 20, 21])
    parser.add_argument("--max-missed", type=int, default=12)
    args = parser.parse_args()

    tracker_args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    variant = f"learned_{args.model_kind}_t{args.threshold:.2f}_h{args.min_hits}_{args.output_box}"
    modules = set(args.modules)
    for cache_jsonl in sorted((args.cache_dir / "jsonl").glob("*.jsonl")):
        seq = cache_jsonl.stem
        model = joblib.load(args.model_dir / f"{args.model_kind}_{seq}.joblib")
        base_by_frame, candidates_by_frame, seen_frames = score_sequence_once(
            cache_jsonl,
            tracker_args,
            model,
            modules,
            args.max_missed,
        )
        out = args.output_dir / "jsonl" / f"{variant}.{seq}.jsonl"
        write_jsonl(out, base_by_frame, candidates_by_frame, seen_frames, args.threshold, args.min_hits, args.output_box)
        print(out)


if __name__ == "__main__":
    main()
