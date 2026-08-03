#!/usr/bin/env python3
"""Run YOLO26 + PEEK-assisted tracking on a video or camera source."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
import time
from typing import Iterator

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from peek.tracking import YOLOPEEKTracker, draw_tracks  # noqa: E402


NAMES = {0: "antenna", 1: "body", 2: "solar", 3: "thruster"}
COLOR_YOLO = (40, 255, 40)
COLOR_PEEK = (255, 0, 255)
COLOR_PREDICTED = (255, 255, 0)
COLOR_SHADOW_YOLO = (255, 170, 40)
COLOR_PEEK_PROPOSAL = (0, 215, 255)
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
    13: (255, 80, 160),
    15: (255, 0, 80),
    16: (255, 120, 0),
    18: (120, 255, 0),
    19: (180, 80, 255),
    20: (0, 120, 255),
    21: (255, 255, 80),
    22: (255, 80, 80),
}


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_video_source(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def source_images(value: str) -> list[Path]:
    path = Path(value)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    matches = sorted(Path(p) for p in glob.glob(value))
    return [p for p in matches if p.suffix.lower() in IMAGE_SUFFIXES]


def iter_video_frames(source: str | int) -> Iterator[tuple[int, object]]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def iter_image_frames(paths: list[Path]) -> Iterator[tuple[int, object]]:
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {path}")
        yield index, frame


def encode_track(track) -> dict:
    return {
        "id": int(track.track_id),
        "xyxy": [float(x) for x in track.xyxy],
        "score": float(track.score),
        "class_id": None if track.cls is None else int(track.cls),
        "class_name": None if track.cls is None else NAMES.get(int(track.cls), str(track.cls)),
        "source": track.source,
        "origin": getattr(track, "origin", track.source),
        "module": None if getattr(track, "module", None) is None else int(track.module),
        "modules": [int(m) for m in getattr(track, "modules", ())],
        "age": int(track.age),
        "hits": int(track.hits),
        "missed": int(track.missed),
    }


def encode_detection(det) -> dict:
    return {
        "xyxy": [float(x) for x in det.xyxy],
        "score": float(det.score),
        "class_id": None if det.cls is None else int(det.cls),
        "class_name": None if det.cls is None else NAMES.get(int(det.cls), str(det.cls)),
        "source": det.source,
        "module": None if det.module is None else int(det.module),
        "modules": [int(m) for m in getattr(det, "modules", ())],
    }


def draw_header(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), COLOR_PANEL, -1)
    cv2.putText(out, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_TEXT, 2, cv2.LINE_AA)
    return out


def is_border_proposal(det, shape: tuple[int, int], margin: int = 14, max_area_frac: float = 0.45) -> bool:
    h, w = shape
    x1, y1, x2, y2 = det.xyxy.astype(float)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    touches_border = x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >= (h - margin)
    too_large = area >= max_area_frac * w * h
    return bool(touches_border or too_large)


def draw_source_legend(frame: np.ndarray, items: list[tuple[str, tuple[int, int, int]]], origin: tuple[int, int]) -> None:
    x, y = origin
    row_h = 26
    box_w = 420
    box_h = row_h * len(items) + 18
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), COLOR_PANEL, -1)
    for i, (label, color) in enumerate(items):
        yy = y + 14 + i * row_h
        cv2.rectangle(frame, (x + 14, yy - 10), (x + 42, yy + 8), color, -1)
        cv2.rectangle(frame, (x + 14, yy - 10), (x + 42, yy + 8), COLOR_TEXT, 1)
        cv2.putText(frame, label, (x + 52, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)


def module_color(module: int | None) -> tuple[int, int, int]:
    return MODULE_COLORS.get(module, COLOR_PEEK)


def module_set_label(obj) -> str:
    modules = getattr(obj, "modules", ())
    if modules:
        return "m" + "+".join(str(int(module)) for module in modules)
    module = getattr(obj, "module", None)
    return "m?" if module is None else f"m{int(module)}"


def module_set_color(obj) -> tuple[int, int, int]:
    modules = getattr(obj, "modules", ())
    module = int(modules[0]) if modules else getattr(obj, "module", None)
    return module_color(module)


def draw_final_panel(frame: np.ndarray, tracks, peek_graduate_hits: int = 3) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    hidden_peek_candidates = 0
    for track in tracks:
        if track.source == "peek":
            color = module_set_color(track)
        elif track.source == "predicted":
            color = COLOR_PREDICTED
        else:
            color = COLOR_YOLO

        if getattr(track, "origin", track.source) == "peek" and track.hits < peek_graduate_hits:
            hidden_peek_candidates += 1
            continue

        x1, y1, x2, y2 = track.xyxy.astype(int)
        if track.mask is not None:
            overlay[track.mask > 0] = color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cls_name = NAMES.get(track.cls, str(track.cls)) if track.cls is not None else "obj"
        module = getattr(track, "module", None)
        module_part = f":{module_set_label(track)}" if track.source == "peek" or getattr(track, "origin", track.source) == "peek" else ""
        label = f"{track.track_id}:{cls_name}:{track.source}{module_part}"
        cv2.putText(out, label, (x1, max(52, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
    out = draw_header(out, f"Final tracks | hidden tentative PEEK: {hidden_peek_candidates}")
    draw_source_legend(
        out,
        [
            ("YOLO detection updated this track", COLOR_YOLO),
            (f"PEEK track/recovery, color names module after {peek_graduate_hits} hits", COLOR_PEEK),
            ("No match: predicted/bridged track", COLOR_PREDICTED),
        ],
        (12, out.shape[0] - 102),
    )
    return out


def draw_peek_panel(frame: np.ndarray, peek_detections, tracks, shadow_yolo_detections=(), border_margin: int = 14) -> np.ndarray:
    """Visualize the raw PEEK proposals separately from accepted tracks."""
    out = frame.copy()
    overlay = out.copy()
    visible = []
    suppressed = 0

    for det in peek_detections:
        if is_border_proposal(det, frame.shape[:2], margin=border_margin):
            suppressed += 1
            continue
        visible.append(det)
        color = module_set_color(det)
        if det.mask is not None:
            overlay[det.mask > 0] = color
        x1, y1, x2, y2 = det.xyxy.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{module_set_label(det)} {det.score:.2f}",
            (x1, max(52, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )

    for det in shadow_yolo_detections:
        x1, y1, x2, y2 = det.xyxy.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_SHADOW_YOLO, 1)
        cv2.putText(
            out,
            f"lowYOLO {det.score:.2f}",
            (x1, max(52, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_SHADOW_YOLO,
            1,
            cv2.LINE_AA,
        )

    out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)

    peek_track_count = sum(1 for t in tracks if t.source == "peek")
    predicted_count = sum(1 for t in tracks if t.source == "predicted")
    text = (
        f"PEEK proposals shown: {len(visible)} | edge haze hidden: {suppressed} | "
        f"shadow YOLO: {len(shadow_yolo_detections)} | PEEK-held tracks: {peek_track_count} | predicted: {predicted_count}"
    )
    out = draw_header(out, text)

    module_items = []
    for module in sorted(
        {
            int(module)
            for det in visible
            for module in (getattr(det, "modules", ()) or (() if det.module is None else (det.module,)))
        }
    ):
        module_items.append((f"PEEK contributor m{module}", module_color(module)))
    draw_source_legend(
        out,
        module_items[:5]
        + [
            ("Low-confidence YOLO support only", COLOR_SHADOW_YOLO),
            ("Whole-frame/edge haze ignored; DoG localizes regions", (120, 120, 120)),
        ],
        (12, out.shape[0] - 102),
    )
    return out


def draw_explanation_frame(frame: np.ndarray, result, peek_graduate_hits: int = 3) -> np.ndarray:
    left = draw_final_panel(frame, result.tracks, peek_graduate_hits=peek_graduate_hits)
    right = draw_peek_panel(frame, result.peek_detections, result.tracks, result.shadow_yolo_detections)
    return np.concatenate([left, right], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video path, image directory/glob, or camera index.")
    parser.add_argument(
        "--weights",
        default="weights/yolo26s_peek_bbox_best.pt",
        help="YOLO26 weights. Defaults to the current best bbox checkpoint.",
    )
    parser.add_argument("--output", default="runs/track/peek_yolo26_tracking.mp4", help="Annotated output video path.")
    parser.add_argument("--jsonl", default="runs/track/peek_yolo26_tracking.jsonl", help="Per-frame track JSONL path.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--peek-modules", type=int, nargs="+", default=[12, 16, 19, 21])
    parser.add_argument("--peek-z", type=float, default=1.0)
    parser.add_argument("--peek-min-area", type=int, default=120)
    parser.add_argument("--peek-max-regions-per-module", type=int, default=4)
    parser.add_argument(
        "--peek-max-area-frac",
        type=float,
        default=0.35,
        help="Ignore PEEK proposals whose bounding box covers this fraction of the frame.",
    )
    parser.add_argument(
        "--peek-dog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a difference-of-Gaussians filter to PEEK maps before thresholding.",
    )
    parser.add_argument("--peek-dog-small", type=float, default=2.0)
    parser.add_argument("--peek-dog-large", type=float, default=9.0)
    parser.add_argument("--peek-min-extent", type=float, default=0.08)
    parser.add_argument("--peek-max-aspect-ratio", type=float, default=4.0)
    parser.add_argument(
        "--peek-min-short-side",
        type=int,
        default=12,
        help="Reject PEEK contour boxes with a very small width or height.",
    )
    parser.add_argument("--peek-border-margin", type=int, default=8)
    parser.add_argument(
        "--peek-focus-z",
        type=float,
        default=0.55,
        help="Lower global z threshold used inside expected locations for tracks YOLO just missed.",
    )
    parser.add_argument(
        "--peek-focus-local-z",
        type=float,
        default=0.95,
        help="Local z threshold inside expected track regions, relative to that local crop.",
    )
    parser.add_argument(
        "--peek-focus-padding",
        type=float,
        default=0.30,
        help="Expand lost-track boxes by this fraction before local PEEK recovery.",
    )
    parser.add_argument(
        "--peek-focus-min-area-frac",
        type=float,
        default=0.40,
        help="Minimum focused contour area as a fraction of --peek-min-area.",
    )
    parser.add_argument(
        "--peek-focus-max-regions-per-track",
        type=int,
        default=1,
        help="Maximum weak local PEEK proposals emitted per lost track and module.",
    )
    parser.add_argument(
        "--peek-graduate-hits",
        type=int,
        default=3,
        help="Show PEEK-origin tracks on the final panel after this many matched PEEK hits.",
    )
    parser.add_argument(
        "--peek-union-gate",
        action="store_true",
        help="Restrict PEEK proposals to the expanded YOLO/object-track union.",
    )
    parser.add_argument(
        "--peek-union-pad",
        type=float,
        default=0.06,
        help="Expand YOLO/object-track union by this frame fraction before accepting PEEK proposals.",
    )
    parser.add_argument(
        "--peek-union-min-iou",
        type=float,
        default=0.01,
        help="Minimum overlap with expanded YOLO/object-track union for accepting PEEK proposals.",
    )
    parser.add_argument(
        "--peek-proximity-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep PEEK proposals only near current detections or recently YOLO-supported tracks.",
    )
    parser.add_argument(
        "--peek-anchor-max-distance-frac",
        type=float,
        default=0.12,
        help="Frame-diagonal fraction used as extra distance allowance for PEEK proximity gating.",
    )
    parser.add_argument(
        "--peek-focus-max-tracks",
        type=int,
        default=4,
        help="Maximum recently YOLO-supported lost tracks that get weak local PEEK search windows.",
    )
    parser.add_argument(
        "--peek-focus-max-missed",
        type=int,
        default=3,
        help="Do not create weak local PEEK search windows for tracks missed longer than this.",
    )
    parser.add_argument(
        "--peek-nms-iou",
        type=float,
        default=0.35,
        help="Suppress duplicate PEEK proposals above this IoU before tracking.",
    )
    parser.add_argument(
        "--peek-nms-max-candidates",
        type=int,
        default=50,
        help="Keep only the top-scoring PEEK candidates before pairwise NMS.",
    )
    parser.add_argument(
        "--peek-union-clusters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Union clustered PEEK boxes instead of suppressing them with NMS.",
    )
    parser.add_argument(
        "--peek-cluster-iou",
        type=float,
        default=0.10,
        help="PEEK boxes with at least this IoU are clustered before unioning.",
    )
    parser.add_argument(
        "--peek-cluster-center-frac",
        type=float,
        default=0.35,
        help="Also cluster PEEK boxes whose centers are within this fraction of the larger box diagonal.",
    )
    parser.add_argument(
        "--peek-cluster-min-modules",
        type=int,
        default=1,
        help="Reject union clusters supported by fewer than this many distinct modules.",
    )
    parser.add_argument(
        "--peek-cluster-min-area",
        type=int,
        default=180,
        help="Reject individual/unioned PEEK boxes smaller than this area.",
    )
    parser.add_argument(
        "--peek-cluster-min-short-side",
        type=int,
        default=12,
        help="Reject individual/unioned PEEK boxes whose shorter side is below this many pixels.",
    )
    parser.add_argument(
        "--peek-cluster-max-area-frac",
        type=float,
        default=0.12,
        help="Reject unioned PEEK boxes larger than this fraction of the frame.",
    )
    parser.add_argument(
        "--shadow-yolo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run YOLO at a lower confidence and expose subthreshold boxes as support for PEEK.",
    )
    parser.add_argument(
        "--shadow-yolo-conf",
        type=float,
        default=0.004,
        help="Confidence floor for low-confidence YOLO boxes used as PEEK support.",
    )
    parser.add_argument(
        "--shadow-yolo-as-peek-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use low-confidence YOLO boxes as proximity anchors for PEEK proposals.",
    )
    parser.add_argument(
        "--suppress-peek-yolo-iou",
        type=float,
        default=0.10,
        help="Drop PEEK proposals that overlap a current YOLO detection by this IoU.",
    )
    parser.add_argument(
        "--suppress-peek-yolo-containment",
        type=float,
        default=0.55,
        help="Drop PEEK proposals whose area is mostly covered by a current YOLO detection.",
    )
    parser.add_argument(
        "--motion-model",
        choices=["none", "constant_velocity"],
        default="none",
        help="Per-track motion model used before association.",
    )
    parser.add_argument(
        "--motion-process-noise",
        type=float,
        default=1.0,
        help="Process noise for motion models that use uncertainty.",
    )
    parser.add_argument(
        "--motion-measurement-noise",
        type=float,
        default=10.0,
        help="Measurement noise for motion models that use uncertainty.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--no-video", action="store_true", help="Only write JSONL, no annotated video.")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Write a side-by-side video: final tracks on the left, raw PEEK proposals on the right.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    jsonl_path = Path(args.jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = source_images(args.source)
    if image_paths:
        frames = iter_image_frames(image_paths)
        first_index, first_frame = next(frames)
        fps = 30.0
    else:
        video_source = parse_video_source(args.source)
        frames = iter_video_frames(video_source)
        first_index, first_frame = next(frames)
        cap_probe = cv2.VideoCapture(video_source)
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
        cap_probe.release()

    height, width = first_frame.shape[:2]
    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_width = width * 2 if args.explain else width
        writer = cv2.VideoWriter(str(output), fourcc, fps, (writer_width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output}")

    from peek.tracking import PEEKRegionProposer  # noqa: WPS433

    proposer = PEEKRegionProposer(
        modules=args.peek_modules,
        z_threshold=args.peek_z,
        min_area=args.peek_min_area,
        max_area_fraction=args.peek_max_area_frac,
        use_dog=args.peek_dog,
        dog_sigma_small=args.peek_dog_small,
        dog_sigma_large=args.peek_dog_large,
        min_extent=args.peek_min_extent,
        max_aspect_ratio=args.peek_max_aspect_ratio,
        min_short_side=args.peek_min_short_side,
        border_margin=args.peek_border_margin,
        max_regions_per_module=args.peek_max_regions_per_module,
        focus_z_threshold=args.peek_focus_z,
        focus_local_z_threshold=args.peek_focus_local_z,
        focus_padding=args.peek_focus_padding,
        focus_min_area_fraction=args.peek_focus_min_area_frac,
        focus_max_regions_per_track=args.peek_focus_max_regions_per_track,
    )

    frame_count = 0
    start = time.perf_counter()
    with YOLOPEEKTracker(
        weights=args.weights,
        peek_modules=args.peek_modules,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        proposer=proposer,
        gate_peek_to_yolo_union=args.peek_union_gate,
        peek_gate_padding=args.peek_union_pad,
        peek_gate_min_iou=args.peek_union_min_iou,
        gate_peek_by_anchor_distance=args.peek_proximity_gate,
        peek_anchor_max_distance_frac=args.peek_anchor_max_distance_frac,
        peek_focus_max_tracks=args.peek_focus_max_tracks,
        peek_focus_max_missed=args.peek_focus_max_missed,
        peek_nms_iou=args.peek_nms_iou,
        peek_nms_max_candidates=args.peek_nms_max_candidates,
        union_cluster_peek=args.peek_union_clusters,
        peek_cluster_iou=args.peek_cluster_iou,
        peek_cluster_center_frac=args.peek_cluster_center_frac,
        peek_cluster_min_modules=args.peek_cluster_min_modules,
        peek_cluster_min_area=args.peek_cluster_min_area,
        peek_cluster_min_short_side=args.peek_cluster_min_short_side,
        peek_cluster_max_area_fraction=args.peek_cluster_max_area_frac,
        shadow_yolo_conf=args.shadow_yolo_conf,
        use_shadow_yolo=args.shadow_yolo,
        use_shadow_yolo_as_peek_anchor=args.shadow_yolo_as_peek_anchor,
        suppress_peek_yolo_iou=args.suppress_peek_yolo_iou,
        suppress_peek_yolo_containment=args.suppress_peek_yolo_containment,
        motion_model=args.motion_model,
        motion_process_noise=args.motion_process_noise,
        motion_measurement_noise=args.motion_measurement_noise,
    ) as tracker, jsonl_path.open("w", encoding="utf-8") as handle:
        pending = [(first_index, first_frame)]
        for _, frame in pending:
            result = tracker.process_frame(frame)
            handle.write(
                json.dumps(
                    {
                        "frame_index": result.frame_index,
                        "latency_ms": result.latency_ms,
                        "num_yolo_detections": len(result.yolo_detections),
                        "num_shadow_yolo_detections": len(result.shadow_yolo_detections),
                        "num_peek_detections": len(result.peek_detections),
                        "peek_detections": [
                            encode_detection(det) for det in result.peek_detections
                        ],
                        "shadow_yolo_detections": [
                            encode_detection(det) for det in result.shadow_yolo_detections
                        ],
                        "tracks": [encode_track(track) for track in result.tracks],
                    }
                )
                + "\n"
            )
            if writer is not None:
                rendered = (
                    draw_explanation_frame(frame, result, peek_graduate_hits=args.peek_graduate_hits)
                    if args.explain
                    else draw_tracks(frame, result.tracks, names=NAMES)
                )
                writer.write(rendered)
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
        if not args.max_frames or frame_count < args.max_frames:
            for _, frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                result = tracker.process_frame(frame)
                handle.write(
                    json.dumps(
                        {
                            "frame_index": result.frame_index,
                            "latency_ms": result.latency_ms,
                            "num_yolo_detections": len(result.yolo_detections),
                            "num_shadow_yolo_detections": len(result.shadow_yolo_detections),
                            "num_peek_detections": len(result.peek_detections),
                            "peek_detections": [
                                encode_detection(det) for det in result.peek_detections
                            ],
                            "shadow_yolo_detections": [
                                encode_detection(det) for det in result.shadow_yolo_detections
                            ],
                            "tracks": [encode_track(track) for track in result.tracks],
                        }
                    )
                    + "\n"
                )
                if writer is not None:
                    rendered = (
                        draw_explanation_frame(frame, result, peek_graduate_hits=args.peek_graduate_hits)
                        if args.explain
                        else draw_tracks(frame, result.tracks, names=NAMES)
                    )
                    writer.write(rendered)
                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    break

    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - start
    fps_eff = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"tracked_frames={frame_count} elapsed_s={elapsed:.2f} effective_fps={fps_eff:.2f}")
    print(f"jsonl={jsonl_path}")
    if writer is not None:
        print(f"video={output}")


if __name__ == "__main__":
    main()
