#!/usr/bin/env python3
"""Merge MOT PEEK module-search shard summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


NUMERIC = {
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
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("gpu*/summary_shard*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in NUMERIC & row.keys():
                    row[key] = float(row[key])
                rows.append(row)
    rows.sort(key=lambda row: (row.get("mot_idf1", 0.0), row.get("idf1_like", 0.0), row.get("mot_mota", 0.0)), reverse=True)
    output = args.output or args.root / "summary_all.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"rows={len(rows)} output={output}")
    for row in rows[:12]:
        print(
            f"{row['variant']:28s} mot_idf1={row['mot_idf1']:.4f} mot_mota={row['mot_mota']:.4f} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} FP={int(row['fp'])} IDS={int(row['id_switches'])}"
        )


if __name__ == "__main__":
    main()
