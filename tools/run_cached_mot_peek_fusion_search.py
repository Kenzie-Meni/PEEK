#!/usr/bin/env python3
"""Focused cached search over fused PEEK layer combinations."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from run_cached_mot_peek_search import Variant, run_cached_variant, run, summarize, REPO


def combo_variants(layers: list[int], motion_models: list[str], sizes: list[int]) -> list[Variant]:
    variants = []
    seen = set()
    for motion in motion_models:
        suffix = "_kf" if motion == "constant_velocity" else "_nomotion"
        for size in sizes:
            for combo in itertools.combinations(layers, size):
                for union in (False, True):
                    name = "fuse_" + "_".join(str(x) for x in combo) + ("_union" if union else "_nms") + suffix
                    if name not in seen:
                        variants.append(Variant(name, combo, motion, union_clusters=union))
                        seen.add(name)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--mot-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 18, 20, 21, 7, 16, 19, 22])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--motion-models", nargs="+", choices=["none", "constant_velocity"], default=["none", "constant_velocity"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    variants = combo_variants(args.layers, args.motion_models, args.sizes)
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
                str(Path("/home/rwhite/mambaforge/envs/peek/bin/python")),
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
