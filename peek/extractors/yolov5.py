"""
YOLOv5 latent extraction.

This module extracts intermediate activations ("latents after a module")
from a vendored YOLOv5 repository using forward hooks. The YOLOv5 repo is
expected at: third_party/yolov5

Latents are saved per image as pickles containing:
    {module_index: torch.Tensor}
"""

import glob
import os
import sys
from pathlib import Path

import torch

from peek.extractors.hooks import LatentExtractor


@torch.no_grad()
def extract_yolov5_latents(
    weights_path: str,
    images_glob: str,
    out_dir: str,
    device: str = "",
    imgsz: int = 640,
    modules=None,
    fp16: bool = False,
    to_cpu: bool = True,
    limit: int = 0,
    warmup: bool = True,
    return_first: bool = False,
):
    """
    Run YOLOv5 inference and save latents collected via forward hooks.

    Args:
        weights_path   Path to YOLOv5 .pt weights
        images_glob    Glob for input images
        out_dir        Output folder for .pkl files
        device         YOLOv5 device selector string
        imgsz          Square inference resolution
        modules        None for all modules, or list of YOLOv5 module indices
        fp16           Store captured latents as float16
        to_cpu         Move captured latents to CPU before saving
        limit          Max number of images (0 = all)
        warmup         Run YOLOv5 warmup once before loop
        return_first   Return first cache dict for inspection

    Returns:
        None by default, or the first cache dict if return_first=True.
    """
    # Resolve YOLOv5 submodule path and add it to sys.path
    y5_root = Path("third_party/yolov5").resolve()
    if not y5_root.exists():
        raise FileNotFoundError(f"Missing YOLOv5 at {y5_root}")
    sys.path.insert(0, str(y5_root))

    # Import YOLOv5 internals (requires yolov5 root on sys.path)
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import check_img_size
    from utils.torch_utils import select_device

    # Local import to avoid forcing OpenCV unless this function is used
    import cv2

    # Collect input images
    paths = sorted(glob.glob(images_glob))
    if limit and limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No images matched: {images_glob}")

    # Create output folder
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model backend
    device_obj = select_device(device)
    backend = DetectMultiBackend(weights_path, device=device_obj)

    # Stride-aligned image size
    stride = int(backend.stride)
    imgsz_checked = check_img_size(imgsz, s=stride)

    # Underlying PyTorch model (this is where hooks attach)
    if not hasattr(backend, "model"):
        raise ValueError("DetectMultiBackend missing .model (expected PyTorch backend)")
    torch_model = backend.model

    # Hook extractor attaches to torch_model.model[i] (YOLOv5 module indexing)
    extractor = LatentExtractor(torch_model, modules=modules, to_cpu=to_cpu, fp16=fp16)

    extractor.start()
    first_cache = None

    for p in paths:
        # Read image (BGR uint8)
        im0 = cv2.imread(p)
        if im0 is None:
            continue

        # Letterbox to model resolution
        im = letterbox(im0, new_shape=imgsz_checked, stride=stride, auto=True)[0]

        # BGR -> RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # HWC -> CHW
        im = im.transpose(2, 0, 1)

        # To torch
        im = torch.from_numpy(im).contiguous()
        im = im.to(device_obj).float() / 255.0

        # Add batch dim if needed
        if im.ndim == 3:
            im = im.unsqueeze(0)

        # Forward pass populates extractor.cache via hooks
        _ = backend(im)

        # Optionally keep the first cache in memory
        if return_first and first_cache is None:
            first_cache = dict(extractor.cache)

        # Save one pickle per image, named by filename stem
        stem = os.path.splitext(os.path.basename(p))[0]
        extractor.save(out_path / f"{stem}.pkl")
        extractor.clear()

    extractor.stop()

    if return_first:
        return first_cache
    return None
