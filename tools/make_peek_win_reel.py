#!/usr/bin/env python3
"""Concatenate PEEK win clips with a pause on each highlighted win frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def read_clip_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--max-clips", type=int, default=12)
    args = parser.parse_args()

    entries = []
    for report in args.reports:
        entries.extend(json.loads(report.read_text(encoding="utf-8")))
    entries = entries[: args.max_clips]
    if not entries:
        raise RuntimeError("No clips found in reports.")

    writer = None
    output_fps = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in entries:
        clip = Path(entry["clip"])
        if not clip.is_absolute():
            clip = Path(__file__).resolve().parents[1] / clip
        frames, fps = read_clip_frames(clip)
        if not frames:
            print(f"Skipping empty/unreadable clip: {clip}")
            continue
        if writer is None:
            output_fps = fps
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
        pause_count = max(1, int(round(args.pause_seconds * output_fps)))
        win_index = len(frames) // 2
        for i, frame in enumerate(frames):
            writer.write(frame)
            written += 1
            if i == win_index:
                for _ in range(pause_count):
                    writer.write(frame)
                    written += 1
    if writer is None:
        raise RuntimeError("No frames were written.")
    writer.release()
    print(f"wrote={args.output} clips={len(entries)} frames={written} fps={output_fps}")


if __name__ == "__main__":
    main()
