"""
YOLOv5 latent extraction.

Extracts intermediate activations ("latents after a module") from the vendored YOLOv5
repo using forward hooks.

Vendor path:
    third_party/yolov5

Latents are saved per image as pickles:
    {module_index: torch.Tensor}
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from peek.extractors.hooks import LatentExtractor
from peek.utils.paths import repo_path


def _add_yolov5_to_syspath() -> Path:
    """
    Adds the vendored YOLOv5 repo to sys.path so its internal imports work.
    """
    root = repo_path(".")
    y5_root = (root / "third_party" / "yolov5").resolve()
    if not y5_root.exists():
        raise FileNotFoundError(f"Missing vendored YOLOv5 repo at: {y5_root}")

    if str(y5_root) not in sys.path:
        sys.path.insert(0, str(y5_root))

    return y5_root


def _resolve_glob(pattern: str) -> str:
    """
    Resolves an images glob.

    - If absolute: use as-is.
    - If relative: interpret relative to repo root.
    """
    if os.path.isabs(pattern):
        return pattern
    return str((repo_path(".") / pattern).resolve())


@torch.no_grad()
def extract_yolov5_latents(
    weights_path: Union[str, Path],
    images_glob: str,
    out_dir: Union[str, Path],
    device: str = "",
    imgsz: int = 640,
    modules: Optional[List[int]] = None,
    fp16: bool = False,
    to_cpu: bool = True,
    limit: int = 0,
    warmup: bool = True,
    return_first: bool = False,
    verbose: bool = False,
) -> Optional[Dict[int, torch.Tensor]]:
    """
    Run YOLOv5 inference and save latents collected via forward hooks.

    Args:
        weights_path   Path to YOLOv5 .pt weights (repo-relative OK)
        images_glob    Glob for input images (repo-relative OK; interpreted from repo root)
        out_dir        Output folder for .pkl files (repo-relative OK)
        device         YOLOv5 device selector string ("", "cpu", "0", etc.)
        imgsz          Inference resolution (YOLOv5 will stride-align)
        modules        None for all modules, or list of YOLOv5 module indices
        fp16           Store captured latents as float16
        to_cpu         Move captured latents to CPU before saving
        limit          Max number of images (0 = all)
        warmup         Run YOLOv5 warmup once before loop
        return_first   Return first cache dict for inspection
        verbose        Print progress lines

    Returns:
        None, or first cache dict if return_first=True.
    """
    # Ensure vendored yolov5 can be imported
    _add_yolov5_to_syspath()

    # Resolve IO paths
    weights_path_p = repo_path(weights_path)
    out_dir_p = repo_path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if not weights_path_p.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path_p}")

    # YOLOv5 internals (require yolov5 root on sys.path)
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import check_img_size
    from utils.torch_utils import select_device

    import cv2  # local import

    # Collect input images
    g = _resolve_glob(images_glob)
    paths = sorted(glob.glob(g))
    if limit and limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No images matched: {images_glob} (resolved: {g})")

    # Load model backend
    device_obj = select_device(device)
    backend = DetectMultiBackend(str(weights_path_p), device=device_obj)

    stride = int(backend.stride)
    imgsz_checked = check_img_size(imgsz, s=stride)

    if not hasattr(backend, "model"):
        raise ValueError("DetectMultiBackend missing .model (expected PyTorch backend)")
    torch_model = backend.model

    # Attach hooks (LatentExtractor knows how to index a Sequential-like model)
    extractor = LatentExtractor(torch_model, modules=modules, to_cpu=to_cpu, fp16=fp16)

    if warmup:
        backend.warmup(imgsz=(1, 3, imgsz_checked, imgsz_checked))

    extractor.start()
    first_cache = None

    for p in paths:
        im0 = cv2.imread(p)
        if im0 is None:
            if verbose:
                print(f"[yolov5] skip unreadable: {p}")
            continue

        # Letterbox to stride-aligned shape
        im = letterbox(im0, new_shape=imgsz_checked, stride=stride, auto=True)[0]

        # BGR->RGB and HWC->CHW
        # .copy() avoids negative-stride numpy views (torch.from_numpy can't handle them)
        im = im[:, :, ::-1].transpose(2, 0, 1).copy()
        im = torch.from_numpy(im).contiguous()

        # Normalize to [0,1], move to device, add batch dim
        im = im.to(device_obj).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        # Forward pass populates extractor.cache
        _ = backend(im)

        if return_first and first_cache is None:
            first_cache = dict(extractor.cache)

        # Save per-image latents
        stem = Path(p).stem
        out_pkl = out_dir_p / f"{stem}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(dict(extractor.cache), f)

        extractor.clear()

        if verbose:
            print(f"[yolov5] saved: {out_pkl}")

    extractor.stop()
    return first_cache if return_first else None
