"""
Ultralytics (YOLO26) latent extraction.

This module extracts intermediate activations ("latents after a module")
from a vendored Ultralytics repository using forward hooks.

Expected vendor path:
    third_party/ultralytics

Weights:
    You can pass a local .pt path, or a model name like "yolo26s.pt".
    Ultralytics will download named weights on first use if not present.

Latents are saved per image as pickles containing:
    {module_index: torch.Tensor}

Notes:
    - The "module_index" here refers to indices inside the underlying model's
      top-level sequential/list of layers (when available).
    - Some Ultralytics models wrap the actual layer list under .model.model.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from peek.extractors.hooks import LatentExtractor


# -----------------------
# Repo-relative paths
# -----------------------
def _find_repo_root(start: Optional[Path] = None) -> Path:
    # Finds repo root by walking upward until a marker is found.
    if start is None:
        start = Path(__file__).resolve()

    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "third_party").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "README.md").exists():
            return parent

    return Path.cwd().resolve()


def _repo_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (_find_repo_root() / p).resolve()


def _add_ultralytics_to_syspath() -> Path:
    repo_root = _find_repo_root()
    ulta_root = (repo_root / "third_party" / "ultralytics").resolve()
    if not ulta_root.exists():
        raise FileNotFoundError(f"Missing Ultralytics vendor at: {ulta_root}")
    if str(ulta_root) not in sys.path:
        sys.path.insert(0, str(ulta_root))
    return ulta_root


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(image_dir: Path, limit: int = 0) -> List[Path]:
    paths = [p for p in sorted(image_dir.iterdir()) if p.is_file() and _is_image(p)]
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


# -----------------------
# Model plumbing
# -----------------------
def _unwrap_torch_model(yolo_obj) -> torch.nn.Module:
    """
    Ultralytics YOLO() object usually exposes a torch module at:
        yolo_obj.model

    But the "list of layers" is often at:
        yolo_obj.model.model
    (where the inner .model is a Sequential / list-like of blocks)

    This returns the torch module we should forward through (outer),
    and the module we should index hooks against (inner, if present).
    """
    if not hasattr(yolo_obj, "model"):
        raise ValueError("Ultralytics YOLO object missing .model")

    torch_model = yolo_obj.model
    return torch_model


def _hook_target(torch_model: torch.nn.Module) -> torch.nn.Module:
    """
    Returns the module whose children correspond to stable "indices" we want.

    Preference order:
      1) torch_model.model   (common in Ultralytics DetectionModel)
      2) torch_model         (fallback)
    """
    inner = getattr(torch_model, "model", None)
    if isinstance(inner, torch.nn.Module):
        return inner
    return torch_model


@torch.no_grad()
def extract_ultralytics_latents(
    weights: str,
    image_dir: str,
    out_dir: str,
    device: Union[str, int] = "",
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
    Run Ultralytics YOLO26 inference and save latents collected via forward hooks.

    Args:
        weights        "yolo26s.pt" (auto-download) OR a local path to weights
        image_dir      Folder of images (repo-relative OK)
        out_dir        Output folder for .pkl files (repo-relative OK)
        device         "" / "cpu" / 0 / "0" etc (passed to predict)
        imgsz          Inference resolution
        modules        None for all hookable top-level modules, or a list of indices
        fp16           Store captured latents as float16
        to_cpu         Move captured latents to CPU before saving
        limit          Max number of images (0 = all)
        warmup         Run one dummy predict before loop
        return_first   Return first cache dict for inspection
        verbose        Print progress lines

    Output format:
        One pickle per image: {module_idx: torch.Tensor}
    """
    _add_ultralytics_to_syspath()

    # Import Ultralytics AFTER sys.path modification
    from ultralytics import YOLO  # type: ignore

    repo_root = _find_repo_root()
    image_dir_p = _repo_path(image_dir)
    out_dir_p = _repo_path(out_dir)

    if not image_dir_p.exists():
        raise FileNotFoundError(f"Missing image_dir: {image_dir_p}")

    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Load YOLO model (named weights auto-download if needed)
    model = YOLO(weights)

    torch_model = _unwrap_torch_model(model)
    target = _hook_target(torch_model)

    # Attach hooks to the target (indexed module list, when possible)
    extractor = LatentExtractor(target, modules=modules, to_cpu=to_cpu, fp16=fp16)
    extractor.start()

    paths = _iter_images(image_dir_p, limit=limit)
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir_p}")

    # Warmup: run one prediction to trigger any lazy init
    if warmup:
        _ = model.predict(source=str(paths[0]), imgsz=imgsz, device=device, verbose=False)

    first_cache = None

    for p in paths:
        if verbose:
            print(f"[ultralytics] predict: {p.name}")

        # Running predict triggers forward pass -> hooks populate extractor.cache
        _ = model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)

        if return_first and first_cache is None:
            first_cache = dict(extractor.cache)

        # Save per-image latents
        out_pkl = out_dir_p / f"{p.stem}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(dict(extractor.cache), f)

        extractor.clear()

    extractor.stop()

    if verbose:
        print(f"[ultralytics] saved latents to: {out_dir_p}")

    return first_cache if return_first else None
