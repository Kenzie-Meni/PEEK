"""
PEEK plotting.

Overlays PEEK heatmaps on input images using per-image latent pickles saved by the
latent extractor.

Per-image pickle format:
    {module_index: torch.Tensor}

Supported latent tensor shapes for PEEK overlay:
    - (B, C, H, W)  -> uses batch index 0
    - (C, H, W)
    - (H, W, C)

Optionally, prediction images can be generated (YOLOv5) and shown as a 3rd column
with run_path="auto".
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch


# -----------------------
# Repo-relative paths
# -----------------------
def _find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Finds the repo root by walking upward from this file until a marker is found.

    Markers:
      - third_party/
      - .git/
      - pyproject.toml
      - README.md
    """
    if start is None:
        start = Path(__file__).resolve()

    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "third_party").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / "README.md").exists():
            return parent

    return Path.cwd().resolve()


def _repo_path(p: Union[str, Path]) -> Path:
    """
    Resolves a path relative to repo root (unless already absolute).
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return (_find_repo_root() / p).resolve()


def _is_image(p: Path) -> bool:
    """
    Checks common image extensions.
    """
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(image_dir: Path, limit: int = 0) -> List[Path]:
    """
    Returns sorted image paths in a directory.
    """
    paths = [p for p in sorted(image_dir.iterdir()) if p.is_file() and _is_image(p)]
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


# -----------------------
# PEEK math (ORIGINAL)
# -----------------------
def _compute_PEEK_original(feature_maps_hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Original YOLOv5-era PEEK math (matches your old notebook code).

    Inputs:
      feature_maps_hwc: (H, W, C) float array

    Steps:
      1) x_pos = x + abs(min(x))
      2) entropy_map = -sum_c entr(x_pos)_c
      3) resize to (out_h, out_w)
    """
    import cv2  # local import
    from scipy.special import entr

    x = feature_maps_hwc.astype(np.float32, copy=False)

    # Make feature map positive (global min over H,W,C)
    x_pos = x + np.abs(np.min(x))

    # Compute pseudo-entropy over channels
    entropy_map = -np.sum(entr(x_pos), axis=-1)

    # Resize to image size
    peek_map = cv2.resize(entropy_map, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return peek_map.astype(np.float32, copy=False)


def _tensor_to_hwc(t: torch.Tensor) -> Optional[np.ndarray]:
    """
    Converts a torch tensor into a NumPy array in HWC format.
    Returns None if tensor shape is unsupported.
    """
    if not isinstance(t, torch.Tensor):
        return None

    # (B,C,H,W) -> (H,W,C)
    if t.ndim == 4:
        t0 = t[0]
        if t0.ndim != 3:
            return None
        return t0.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()

    # (C,H,W) -> (H,W,C)
    if t.ndim == 3:
        c, h, w = t.shape

        # Treat as CHW if first dim is channel-like
        if c <= 4096 and h >= 2 and w >= 2:
            return t.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()

        # Otherwise assume already HWC
        arr = t.detach().float().cpu().contiguous().numpy()
        if arr.shape[-1] <= 4096:
            return arr
        return None

    return None


# -----------------------
# YOLOv5 predictions (optional)
# -----------------------
@torch.no_grad()
def _yolov5_generate_predictions(
    image_dir: Path,
    weights_path: Path,
    out_dir: Path,
    imgsz: int = 640,
    device: str = "",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    verbose: bool = False,
) -> Path:
    """
    Runs YOLOv5 and writes annotated images into out_dir (same filenames).
    """
    repo_root = _find_repo_root()
    y5_root = (repo_root / "third_party" / "yolov5").resolve()
    if not y5_root.exists():
        raise FileNotFoundError(f"Missing YOLOv5 submodule at: {y5_root}")

    if str(y5_root) not in sys.path:
        sys.path.insert(0, str(y5_root))

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

        # Important: copy() avoids negative strides from ::-1 slicing
        im = im[:, :, ::-1].transpose(2, 0, 1).copy()

        im = torch.from_numpy(im).to(dev)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = backend(im)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred = non_max_suppression(
            pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
        )

        annotator = Annotator(im0.copy(), line_width=2)
        det = pred[0]

        if det is not None and len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                c = int(cls)
                if names and 0 <= c < len(names):
                    label = f"{names[c]} {conf:.2f}"
                else:
                    label = f"{c} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))

        out_img = annotator.result()
        out_file = out_dir / p.name
        cv2.imwrite(str(out_file), out_img)

        if verbose:
            print(f"[plotting] saved prediction: {out_file}")

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
    verbose: bool = False,
    weights_path: Optional[Union[str, Path]] = None,
    imgsz: int = 640,
    device: str = "",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    limit: int = 0,
):
    """
    Plots PEEK overlays for a set of modules for each image in a folder.

    Args:
        modules         List of module indices to plot
        image_dir       Folder of images (repo-relative OK)
        feature_folder  Folder of per-image latent pickles (repo-relative OK)
        save_path       False shows figures; otherwise saves PNGs to folder
        run_path        False hides preds; "auto" generates preds; otherwise uses folder
        weights_path    Required when run_path="auto"
        limit           0 = all images; otherwise caps count
    """
    import matplotlib.pyplot as plt  # local import

    repo_root = _find_repo_root()

    image_dir = _repo_path(image_dir)
    feature_folder = _repo_path(feature_folder)

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image_dir: {image_dir}")
    if not feature_folder.exists():
        raise FileNotFoundError(f"Missing feature_folder: {feature_folder}")

    # Resolve save folder
    save_dir: Optional[Path] = None
    if save_path:
        save_dir = _repo_path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve / generate prediction folder
    pred_dir: Optional[Path] = None
    if run_path:
        if isinstance(run_path, str) and run_path.lower() == "auto":
            if weights_path is None:
                raise ValueError("weights_path is required when run_path='auto'")
            weights_path = _repo_path(weights_path)

            pred_dir = repo_root / "runs" / "peek_detect" / image_dir.name
            pred_dir = _yolov5_generate_predictions(
                image_dir=image_dir,
                weights_path=weights_path,
                out_dir=pred_dir,
                imgsz=imgsz,
                device=device,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                verbose=verbose,
            )
        else:
            pred_dir = _repo_path(run_path)
            if not pred_dir.exists():
                raise FileNotFoundError(f"Missing run_path: {pred_dir}")

    # Collect images
    img_paths = _iter_images(image_dir, limit=limit)
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    cols = 3 if pred_dir is not None else 2

    for img_path in img_paths:
        stem = img_path.stem
        pkl_path = feature_folder / f"{stem}.pkl"

        if not pkl_path.exists():
            if verbose:
                print(f"[plotting] missing latents: {pkl_path}")
            continue

        # Read image (RGB)
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

        # Make axes always 2D
        if len(modules) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, layer_idx in enumerate(modules):
            # Column titles
            if i == 0:
                axes[i, 0].set_title("Input")
                axes[i, 1].set_title("PEEK")
                if pred_dir is not None:
                    axes[i, 2].set_title("Predictions")

            # Input image
            axes[i, 0].imshow(image)
            axes[i, 0].set_ylabel(f"Module {layer_idx}")

            # PEEK overlay
            t = data.get(layer_idx, None)
            axes[i, 1].imshow(image)

            if t is None:
                if verbose:
                    print(f"[plotting] missing module {layer_idx} in {pkl_path.name}")
                axes[i, 1].text(0.02, 0.95, "missing", transform=axes[i, 1].transAxes)
            else:
                hwc = _tensor_to_hwc(t)
                if hwc is None:
                    if verbose:
                        print(f"[plotting] unsupported shape for module {layer_idx}: {tuple(t.shape)}")
                    axes[i, 1].text(0.02, 0.95, "unsupported", transform=axes[i, 1].transAxes)
                else:
                    # ORIGINAL PEEK math + resize (matches old code)
                    peek_hw = _compute_PEEK_original(hwc, out_h=h, out_w=w)
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
            out_file = save_dir / f"{stem}.png"
            fig.savefig(str(out_file), dpi=150)
            if verbose:
                print(f"[plotting] saved: {out_file}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
