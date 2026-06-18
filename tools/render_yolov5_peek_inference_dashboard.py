#!/usr/bin/env python3
"""Render YOLOv5 inference with PEEK module maps on GSFC frames."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.special import entr
import torch


REPO = Path(__file__).resolve().parents[1]
Y5_ROOT = REPO / "third_party" / "yolov5"
if str(Y5_ROOT) not in sys.path:
    sys.path.insert(0, str(Y5_ROOT))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_boxes  # noqa: E402
from utils.plots import Annotator, colors  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


BLUE = (190, 115, 70)
WHITE = (245, 248, 255)
BLACK = (0, 0, 0)
YELLOW = (40, 230, 255)
CYAN = (230, 190, 40)
GRAY = (90, 90, 90)


def source_paths(pattern_or_dir: str) -> list[Path]:
    path = Path(pattern_or_dir)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return sorted(Path(p) for p in glob.glob(pattern_or_dir))


def peek_map(feature_tensor: torch.Tensor, out_hw: tuple[int, int]) -> np.ndarray:
    arr = feature_tensor[0].detach().float().cpu().numpy()
    arr = np.moveaxis(arr, 0, -1)
    arr = arr + abs(float(arr.min()))
    entropy_map = -np.sum(entr(arr), axis=-1)
    entropy_map = cv2.resize(entropy_map, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    entropy_map = np.nan_to_num(entropy_map, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(entropy_map, [2, 98])
    if hi <= lo:
        return np.zeros(out_hw, dtype=np.float32)
    return np.clip((entropy_map - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def heatmap_panel(frame: np.ndarray, heat: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    base = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    heat_small = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((heat_small * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(base, 0.35, color, 0.65, 0)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), WHITE, 2)
    cv2.rectangle(out, (0, 0), (w, 24), BLACK, -1)
    cv2.putText(out, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)
    return out


def architecture_panel(size: tuple[int, int], modules: list[int]) -> np.ndarray:
    w, h = size
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), WHITE, 2)
    cv2.putText(panel, "YOLOv5 modules", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)

    cols = 5
    cell_w = (w - 64) // cols
    cell_h = 56
    start_x = 32
    start_y = 72
    selected = set(modules)
    for module in range(25):
        row = module // cols
        col = module % cols
        x1 = start_x + col * cell_w
        y1 = start_y + row * cell_h
        x2 = x1 + cell_w - 12
        y2 = y1 + 42
        is_selected = module in selected
        fill = YELLOW if is_selected else BLACK
        border = YELLOW if is_selected else GRAY
        cv2.rectangle(panel, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), border, 2)
        text = f"{module}"
        color = BLACK if is_selected else WHITE
        cv2.putText(panel, text, (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    cv2.putText(panel, "marked modules feed PEEK maps", (24, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)
    return panel


def paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    h, w = image.shape[:2]
    canvas[y : y + h, x : x + w] = image


def draw_predictions(
    frame: np.ndarray,
    det: torch.Tensor,
    input_shape: tuple[int, int],
    names,
) -> np.ndarray:
    out = frame.copy()
    annotator = Annotator(out, line_width=2)
    if det is not None and len(det):
        det[:, :4] = scale_boxes(input_shape, det[:, :4], frame.shape).round()
        for *xyxy, score, cls in det:
            c = int(cls)
            label = f"{names[c] if c in names else c} {float(score):.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))
    return annotator.result()


def render_dashboard(
    raw_frame: np.ndarray,
    pred_frame: np.ndarray,
    maps: dict[int, np.ndarray],
    modules: list[int],
    output_size: tuple[int, int],
) -> np.ndarray:
    out_w, out_h = output_size
    canvas = np.full((out_h, out_w, 3), BLUE, dtype=np.uint8)
    cv2.rectangle(canvas, (22, 54), (out_w - 23, out_h - 23), WHITE, -1)
    cv2.putText(canvas, "PEEK Inference", (out_w // 2 - 150, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 3, cv2.LINE_AA)

    margin = 38
    top_y = 72
    main_w, main_h = 720, 405
    arch_w, arch_h = 400, 405
    panel_w, panel_h = 218, 140
    bottom_y = top_y + main_h + 24

    pred_panel = cv2.resize(pred_frame, (main_w, main_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(pred_panel, (0, 0), (main_w - 1, main_h - 1), BLACK, 3)
    cv2.rectangle(pred_panel, (0, 0), (main_w, 24), BLACK, -1)
    cv2.putText(pred_panel, "Predictions", (main_w // 2 - 48, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    paste(canvas, pred_panel, margin, top_y)

    arch = architecture_panel((arch_w, arch_h), modules)
    paste(canvas, arch, margin + main_w + 30, top_y)

    for i, module in enumerate(modules):
        panel = heatmap_panel(raw_frame, maps[module], f"Module {module}", (panel_w, panel_h))
        paste(canvas, panel, margin + i * (panel_w + 14), bottom_y)

    cv2.putText(
        canvas,
        "YOLOv5 baseline inference with PEEK maps from selected modules",
        (margin, out_h - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/home/rwhite/NFS/All_Team/Ryan/datasets/GSFC")
    parser.add_argument("--weights", default="/home/rwhite/NFS/All_Team/Ryan/PEEK/weights/yolov5-satellite-components-det-baseline.pt")
    parser.add_argument("--output", default="/home/rwhite/NFS/All_Team/Ryan/PEEK/runs/track/yolov5_baseline_peek_gsfc_inference.mp4")
    parser.add_argument("--modules", nargs="+", type=int, default=[1, 7, 16, 19, 23])
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    paths = source_paths(args.source)
    if args.max_frames:
        paths = paths[: args.max_frames]
    if not paths:
        raise FileNotFoundError(args.source)

    device = select_device(args.device)
    backend = DetectMultiBackend(args.weights, device=device, fuse=True)
    stride = int(backend.stride)
    imgsz = check_img_size(args.imgsz, s=stride)
    backend.warmup(imgsz=(1, 3, imgsz, imgsz))
    names = backend.names if hasattr(backend, "names") else {}

    cache: dict[int, torch.Tensor] = {}
    hooks = []
    sequential = backend.model.model
    for module in args.modules:
        hooks.append(sequential[module].register_forward_hook(lambda _m, _inp, out, module=module: cache.__setitem__(module, out.detach())))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output}")

    count = 0
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(path)

            cache.clear()
            im = letterbox(frame, new_shape=imgsz, stride=stride, auto=True)[0]
            im = im[:, :, ::-1].transpose(2, 0, 1).copy()
            im_tensor = torch.from_numpy(im).to(device).float() / 255.0
            im_tensor = im_tensor.unsqueeze(0)

            pred = backend(im_tensor)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            det = non_max_suppression(pred, conf_thres=args.conf, iou_thres=args.iou, max_det=300)[0]

            pred_frame = draw_predictions(frame, det, im_tensor.shape[2:], names)
            maps = {module: peek_map(cache[module], frame.shape[:2]) for module in args.modules}
            writer.write(render_dashboard(frame, pred_frame, maps, args.modules, (args.width, args.height)))
            count += 1
    finally:
        writer.release()
        for hook in hooks:
            hook.remove()

    print(f"video={output}")
    print(f"frames={count}")


if __name__ == "__main__":
    main()
