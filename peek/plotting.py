"""
PEEK plotting.

Overlays PEEK heatmaps on input images using per-image latent pickles saved by
extractors.

Per-image pickle format:
    {module_index: torch.Tensor}

Supported latent tensor shapes:
    - (B, C, H, W)  -> uses batch index 0
    - (C, H, W)
    - (H, W, C)

Optionally generates prediction images and shows them as a 3rd column:
    run_path="auto"
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from peek.utils.paths import configure_ultralytics_dir, repo_path, resolve_weights


# -----------------------
# Image helpers
# -----------------------
def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(image_dir: Path, limit: int = 0) -> List[Path]:
    paths = [p for p in sorted(image_dir.iterdir()) if p.is_file() and _is_image(p)]
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


# -----------------------
# PEEK math (original)
# -----------------------
def _peek_entropy_map_hwc(feature_maps_hwc: np.ndarray) -> np.ndarray:
    """
    Original PEEK math:
      positivized_maps = x + abs(min(x))
      entropy_map = -sum(entr(positivized_maps), axis=-1)
    """
    from scipy.special import entr  # local import

    x = feature_maps_hwc.astype(np.float32, copy=False)

    # Shift so the minimum becomes zero (same behavior as your original)
    x = x + float(np.abs(np.min(x)))

    # Elementwise entr(x) = -x * log(x) with entr(0) = 0
    h = -np.sum(entr(x), axis=-1)
    return h.astype(np.float32, copy=False)


def _resize_map_hw(peek_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Resize (H,W) -> (out_h,out_w).
    OpenCV expects size=(width,height).
    """
    import cv2  # local import

    return cv2.resize(peek_hw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _tensor_to_hwc(t: torch.Tensor) -> Optional[np.ndarray]:
    """
    Convert a tensor to HWC numpy for PEEK.

    Accepts:
      - (B,C,H,W) -> use batch 0 -> HWC
      - (C,H,W)   -> HWC
      - (H,W,C)   -> passthrough
    """
    if not isinstance(t, torch.Tensor):
        return None

    if t.ndim == 4:
        t0 = t[0]
        return t0.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()

    if t.ndim == 3:
        # Heuristic: if first dim looks like channels, treat as CHW
        c, h, w = t.shape
        if c <= 4096 and h >= 2 and w >= 2:
            return t.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()

        # Otherwise assume already HWC
        arr = t.detach().float().cpu().contiguous().numpy()
        return arr

    return None


# -----------------------
# Prediction generation
# -----------------------
def _looks_like_yolov5_weights(weights: Union[str, Path]) -> bool:
    """
    Keep this simple and predictable:
    - If filename contains 'yolov5' -> use vendored yolov5 inference/annotator.
    """
    name = str(weights).lower()
    return "yolov5" in Path(name).name


def _add_yolov5_to_syspath() -> Path:
    """
    Adds the vendored YOLOv5 repo to sys.path.
    """
    root = repo_path(".")
    y5_root = (root / "third_party" / "yolov5").resolve()
    if not y5_root.exists():
        raise FileNotFoundError(f"Missing vendored YOLOv5 repo at: {y5_root}")

    if str(y5_root) not in sys.path:
        sys.path.insert(0, str(y5_root))

    return y5_root


@torch.no_grad()
def _yolov5_generate_predictions(
    image_dir: Path,
    weights_path: Path,
    out_dir: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    verbose: bool,
) -> Path:
    """
    Runs vendored YOLOv5 and writes annotated images into out_dir.
    """
    _add_yolov5_to_syspath()

    import cv2  # local import
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.plots import Annotator, colors
    from utils.torch_utils import select_device

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    dev = select_device(device)
    backend = DetectMultiBackend(str(weights_path), device=dev)

    stride = int(backend.stride)
    imgsz_checked = check_img_size(imgsz, s=stride)
    backend.warmup(imgsz=(1, 3, imgsz_checked, imgsz_checked))

    names = backend.names if hasattr(backend, "names") else None

    for p in paths:
        im0 = cv2.imread(str(p))
        if im0 is None:
            if verbose:
                print(f"[plotting] skip unreadable: {p}")
            continue

        im = letterbox(im0, new_shape=imgsz_checked, stride=stride, auto=True)[0]
        im = im[:, :, ::-1].transpose(2, 0, 1).copy()  # contiguous for torch
        im = torch.from_numpy(im).to(dev)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = backend(im)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det)

        annotator = Annotator(im0.copy(), line_width=2)
        det = pred[0]

        if det is not None and len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, score, cls in det:
                c = int(cls)
                label = f"{c} {float(score):.2f}"
                if names and 0 <= c < len(names):
                    label = f"{names[c]} {float(score):.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))

        out_img = annotator.result()
        out_file = out_dir / p.name
        cv2.imwrite(str(out_file), out_img)

        if verbose:
            print(f"[plotting] saved prediction: {out_file}")

    return out_dir


@torch.no_grad()
def _ultralytics_generate_predictions(
    image_dir: Path,
    weights: Union[str, Path],
    out_dir: Path,
    imgsz: int,
    device: Union[str, int],
    conf: float,
    iou: float,
    max_det: int,
    verbose: bool,
) -> Path:
    """
    Runs Ultralytics YOLO and saves annotated predictions into out_dir.
    """
    # Force Ultralytics cache/download/runs under repo root (must happen before import)
    configure_ultralytics_dir()

    from ultralytics import YOLO  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    # Force downloads into <repo>/weights/ when given a bare name
    weights_arg = resolve_weights(weights)

    model = YOLO(weights_arg)

    # Ultralytics writes into project/name
    project = str(out_dir.parent)
    name = out_dir.name

    _ = model.predict(
        source=str(image_dir),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        save=True,
        save_txt=False,
        save_conf=False,
        project=project,
        name=name,
        exist_ok=True,
        verbose=False,  # avoid version-dependent fuse(verbose=...) paths
    )

    if verbose:
        print(f"[plotting] saved predictions to: {out_dir}")

    return out_dir


# -----------------------
# Public API
# -----------------------
def plot_PEEK(
    modules: List[int],
    image_dir: Union[str, Path],
    feature_folder: Union[str, Path],
    save_path: Union[bool, str, Path] = False,
    run_path: Union[bool, str, Path] = False,
    weights_path: Optional[Union[str, Path]] = None,
    imgsz: int = 640,
    device: Union[str, int] = "",
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    limit: int = 0,
    verbose: bool = False,
):
    """
    Plot PEEK overlays for each image in image_dir.

    Args:
        modules        module indices to plot (rows)
        image_dir      folder containing input images
        feature_folder folder containing {stem}.pkl latents
        save_path      False shows figures, otherwise saves PNGs into folder
        run_path       False: no preds; "auto": generate preds; else: folder of preds
        weights_path   required when run_path="auto"
        limit          0 = all images
    """
    import matplotlib.pyplot as plt  # local import

    image_dir = repo_path(image_dir)
    feature_folder = repo_path(feature_folder)

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image_dir: {image_dir}")
    if not feature_folder.exists():
        raise FileNotFoundError(f"Missing feature_folder: {feature_folder}")

    # Resolve output folder for figures
    save_dir: Optional[Path] = None
    if save_path:
        save_dir = repo_path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve / generate predictions folder
    pred_dir: Optional[Path] = None
    if run_path:
        if isinstance(run_path, str) and run_path.lower() == "auto":
            if weights_path is None:
                raise ValueError("weights_path is required when run_path='auto'")

            w = weights_path
            out_base = repo_path(".") / "runs" / "peek_detect" / image_dir.name

            if _looks_like_yolov5_weights(w):
                wp = repo_path(w)
                if not wp.exists():
                    raise FileNotFoundError(f"Missing YOLOv5 weights file: {wp}")

                pred_dir = _yolov5_generate_predictions(
                    image_dir=image_dir,
                    weights_path=wp,
                    out_dir=out_base,
                    imgsz=imgsz,
                    device=str(device),
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    verbose=verbose,
                )
            else:
                pred_dir = _ultralytics_generate_predictions(
                    image_dir=image_dir,
                    weights=w,
                    out_dir=out_base,
                    imgsz=imgsz,
                    device=device,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    verbose=verbose,
                )
        else:
            pred_dir = repo_path(run_path)
            if not pred_dir.exists():
                raise FileNotFoundError(f"Missing run_path folder: {pred_dir}")

    # Collect images
    img_paths = _iter_images(image_dir, limit=limit)
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    cols = 3 if pred_dir is not None else 2

    for img_path in img_paths:
        pkl_path = feature_folder / f"{img_path.stem}.pkl"
        if not pkl_path.exists():
            if verbose:
                print(f"[plotting] missing latents: {pkl_path}")
            continue

        # Read input image
        image = plt.imread(str(img_path))
        if image.ndim < 3:
            if verbose:
                print(f"[plotting] skip non-RGB: {img_path}")
            continue

        h, w = int(image.shape[0]), int(image.shape[1])

        # Load feature dict
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            if verbose:
                print(f"[plotting] unexpected pickle type: {pkl_path}")
            continue

        fig, axes = plt.subplots(
            len(modules),
            cols,
            figsize=(4.5 * cols, 3.0 * max(len(modules), 1)),
        )

        # Ensure 2D axes array
        if len(modules) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, layer_idx in enumerate(modules):
            # Titles
            if i == 0:
                axes[i, 0].set_title("Input")
                axes[i, 1].set_title("PEEK")
                if pred_dir is not None:
                    axes[i, 2].set_title("Predictions")

            # Input
            axes[i, 0].imshow(image)
            axes[i, 0].set_ylabel(f"Module {layer_idx}")

            # PEEK overlay
            axes[i, 1].imshow(image)

            t = data.get(layer_idx, None)
            if t is None:
                axes[i, 1].text(0.02, 0.95, "missing", transform=axes[i, 1].transAxes)
            else:
                hwc = _tensor_to_hwc(t)
                if hwc is None:
                    axes[i, 1].text(0.02, 0.95, "unsupported", transform=axes[i, 1].transAxes)
                else:
                    peek_hw = _peek_entropy_map_hwc(hwc)
                    peek_hw = _resize_map_hw(peek_hw, out_h=h, out_w=w)
                    axes[i, 1].imshow(peek_hw, alpha=0.7, cmap="jet")

            # Predictions
            if pred_dir is not None:
                pred_img_path = pred_dir / img_path.name
                if pred_img_path.exists():
                    pred_img = plt.imread(str(pred_img_path))
                    axes[i, 2].imshow(pred_img)
                else:
                    axes[i, 2].imshow(image)
                    axes[i, 2].text(0.02, 0.95, "missing pred", transform=axes[i, 2].transAxes)

        # Hide ticks
        for i in range(len(modules)):
            for j in range(cols):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        fig.tight_layout()

        if save_dir is not None:
            out_file = save_dir / f"{img_path.stem}.png"
            fig.savefig(str(out_file), dpi=150)
            if verbose:
                print(f"[plotting] saved: {out_file}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
