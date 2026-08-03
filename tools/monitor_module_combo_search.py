#!/usr/bin/env python3
"""Monitor module-combo tmux shards and notify when the search is complete."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def read_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("gpu*/summary_shard*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    for row in rows:
        for key in ("precision", "recall", "mota_like", "idf1_like"):
            row[key] = float(row[key])
        for key in ("id_switches", "tp", "fp", "fn", "pred", "gt", "frames"):
            row[key] = int(float(row[key]))
    rows.sort(key=lambda item: (item["idf1_like"], item["mota_like"], item["recall"]), reverse=True)
    return rows


def write_merged(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def notify(webhook: str, content: str) -> None:
    if not webhook:
        return
    data = json.dumps({"username": "fit-afrl", "content": content}).encode("utf-8")
    request = urllib.request.Request(webhook, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--webhook", default="")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    while any(tmux_session_exists(session) for session in args.sessions):
        time.sleep(args.poll_seconds)

    rows = read_rows(args.root)
    merged = args.root / "summary_all.csv"
    write_merged(rows, merged)

    if rows:
        best = rows[0]
        content = (
            "fit-afrl: module combo search is done. "
            f"Best={best['variant']} modules={best.get('modules', '')} "
            f"IDF1-like={best['idf1_like']:.4f}, MOTA-like={best['mota_like']:.4f}, "
            f"P={best['precision']:.4f}, R={best['recall']:.4f}, IDSW={best['id_switches']}. "
            f"Summary: {merged}"
        )
    else:
        content = f"fit-afrl: module combo search ended, but no summary rows were found under {args.root}"

    notify(args.webhook, content)


if __name__ == "__main__":
    main()
