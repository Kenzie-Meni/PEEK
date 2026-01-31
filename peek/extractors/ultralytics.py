"""
Ultralytics latent extraction.

Extracts intermediate activations ("latents after a module") from an Ultralytics
YOLO model using forward hooks.

Vendor path:
    third_party/ultralytics

Weights:
    - Pass a local path like "weights/yolo26s.pt" (preferred)
    - Or pass a bare name like "yolo26s.pt" and it will download into <repo>/weights/
      via resolve_weights().

Output:
    One pickle per image:
        {module_index: torch.Tensor}
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from peek.extractors.hooks import LatentExtractor
from peek.utils.paths import configure_ultralytics_dir, repo_path, resolve_weights


# -----------------------
# Ultralytics vendor path
# -----------------------
def _add_ultralytics_to_syspath() -> Path:
    """
    Adds the vendored Ultralytics repo to sys.path so imports resolve to the submodule.
    """
    root = repo_path(".")
    ulta_root = (root / "third_party" / "ultralytics").resolve()
    if not ulta_root.exists():
        raise FileNotFoundError(f"Missing vendored Ultralytics repo at: {ulta_root}")

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


def _hook_target(torch_model: torch.nn.Module) -> torch.nn.Module:
    """
    Chooses a stable "indexed module list" to hook.

    Ultralytics models often have:
        torch_model.model  (a Sequential / list-like of blocks)

    If present, hooks attach there; otherwise hooks attach to torch_model directly.
    """
    inner = getattr(torch_model, "model", None)
    if isinstance(inner, torch.nn.Module):
        return inner
    return torch_model


@torch.no_grad()
def extract_ultralytics_latents(
    weights: Union[str, Path],
    image_dir: Union[str, Path],
    out_dir: Union[str, Path],
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
    Run Ultralytics inference and save latents collected via forward hooks.

    Args:
        weights       "weights/yolo26s.pt" or "yolo26s.pt" (downloads into <repo>/weights/)
        image_dir     folder of images (repo-relative OK)
        out_dir       folder for per-image .pkl (repo-relative OK)
        device        "" / "cpu" / 0 / "0" etc
        imgsz         inference resolution
        modules       None for all hookable modules, or list of indices
        fp16          save latents as float16
        to_cpu        move latents to CPU before saving
        limit         0 = all images
        warmup        run one predict call before loop
        return_first  return first cache dict (for quick inspection)
        verbose       print progress lines

    Returns:
        None, or first cache dict if return_first=True.
    """
    # Force Ultralytics cache/download/runs under repo root (must happen before import)
    configure_ultralytics_dir()

    # Ensure vendored ultralytics is importable
    _add_ultralytics_to_syspath()

    # Import after sys.path + env config
    from ultralytics import YOLO  # type: ignore

    image_dir_p = repo_path(image_dir)
    out_dir_p = repo_path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if not image_dir_p.exists():
        raise FileNotFoundError(f"Missing image_dir: {image_dir_p}")

    # Resolve weights so bare "yolo26s.pt" downloads into <repo>/weights/
    weights_arg = resolve_weights(weights)

    # Load model
    model = YOLO(weights_arg)

    # Underlying torch module (this is what actually runs forward)
    if not hasattr(model, "model"):
        raise ValueError("Ultralytics YOLO object missing .model")

    torch_model = model.model
    target = _hook_target(torch_model)

    # Attach hooks
    extractor = LatentExtractor(target, modules=modules, to_cpu=to_cpu, fp16=fp16)
    extractor.start()

    paths = _iter_images(image_dir_p, limit=limit)
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir_p}")

    # Warmup: triggers any lazy init and ensures hooks have executed once
    if warmup:
        _ = model.predict(source=str(paths[0]), imgsz=imgsz, device=device, verbose=False)

    first_cache = None

    for p in paths:
        if verbose:
            print(f"[ultralytics] predict: {p.name}")

        # Running predict triggers forward hooks
        _ = model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)

        if return_first and first_cache is None:
            first_cache = dict(extractor.cache)

        out_pkl = out_dir_p / f"{p.stem}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(dict(extractor.cache), f)

        extractor.clear()

    extractor.stop()

    if verbose:
        print(f"[ultralytics] saved latents to: {out_dir_p}")

    return first_cache if return_first else None
