#!/usr/bin/env python3
"""Render grouped PEEK module proposal panes from tracker JSONL."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
NAMES = {0: "antenna", 1: "body", 2: "solar", 3: "thruster"}
COLOR_YOLO = (40, 255, 40)
COLOR_SHADOW_YOLO = (255, 170, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (0, 0, 0)
MODULE_COLORS = {
    0: (255, 80, 80),
    1: (255, 170, 40),
    2: (255, 255, 80),
    3: (120, 255, 0),
    4: (0, 220, 255),
    5: (0, 120, 255),
    6: (0, 180, 255),
    7: (0, 255, 255),
    10: (255, 200, 0),
    12: (255, 0, 255),
    15: (255, 0, 80),
    16: (255, 120, 0),
    19: (180, 80, 255),
    20: (0, 120, 255),
    21: (255, 255, 80),
    22: (255, 80, 80),
}


def source_images(value: str) -> list[Path]:
    path = Path(value)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(Path(p) for p in glob.glob(value) if Path(p).suffix.lower() in IMAGE_SUFFIXES)


def module_color(module: int | None) -> tuple[int, int, int]:
    return MODULE_COLORS.get(module, (255, 0, 255))


def header(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), COLOR_PANEL, -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_TEXT, 2, cv2.LINE_AA)
    return out


def draw_yolo_panel(frame: np.ndarray, row: dict) -> np.ndarray:
    out = frame.copy()
    for track in row.get("tracks", []):
        if track.get("source") != "yolo":
            continue
        x1, y1, x2, y2 = [int(v) for v in track["xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_YOLO, 2)
        label = f"{track.get('class_name') or 'obj'} {float(track.get('score', 0.0)):.2f}"
        cv2.putText(out, label, (x1, max(48, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YOLO, 1, cv2.LINE_AA)
    for det in row.get("shadow_yolo_detections", []):
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_SHADOW_YOLO, 1)
    return header(out, f"YOLO accepted + shadow low-conf ({row.get('num_shadow_yolo_detections', 0)})")


def draw_group_panel(frame: np.ndarray, row: dict, title: str, modules: set[int]) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    shown = 0
    present_modules = set()
    for det in row.get("peek_detections", []):
        module = det.get("module")
        if module is None or int(module) not in modules:
            continue
        module = int(module)
        present_modules.add(module)
        color = module_color(module)
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if "mask" in det:
            pass
        cv2.putText(
            out,
            f"m{module} {float(det.get('score', 0.0)):.2f}",
            (x1, max(48, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )
        shown += 1
    out = cv2.addWeighted(overlay, 0.08, out, 0.92, 0)
    mods = ",".join(f"m{m}" for m in sorted(present_modules)) if present_modules else "none"
    return header(out, f"{title}: {shown} boxes | {mods}")


def parse_groups(values: list[str]) -> list[tuple[str, set[int]]]:
    groups = []
    for value in values:
        name, mods = value.split(":", 1)
        groups.append((name, {int(m) for m in mods.split(",") if m}))
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="Group in Name:module,module format. Can be repeated.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = source_images(args.source)
    rows = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines()]
    if args.max_frames:
        rows = rows[: args.max_frames]
        images = images[: args.max_frames]
    group_values = args.group or ["Backbone:6,7,10", "Neck:12,15,20", "Head-adjacent:16,19,21,22"]
    groups = parse_groups(group_values)
    if len(images) < len(rows):
        raise ValueError(f"Not enough source frames: {len(images)} images for {len(rows)} rows")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise FileNotFoundError(images[0])
    h, w = first.shape[:2]
    panel_w = w // 2
    panel_h = h // 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (panel_w * 2, panel_h * 2))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output}")
    try:
        for img_path, row in zip(images, rows):
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise FileNotFoundError(img_path)
            panels = [draw_yolo_panel(frame, row)]
            for title, modules in groups[:3]:
                panels.append(draw_group_panel(frame, row, title, modules))
            while len(panels) < 4:
                panels.append(header(frame, ""))
            panels = [cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA) for panel in panels[:4]]
            canvas = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4])])
            writer.write(canvas)
    finally:
        writer.release()
    print(f"video={output}")
    print(f"frames={len(rows)}")


if __name__ == "__main__":
    main()
