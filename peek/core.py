"""
Core PEEK math and tensor helpers.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from scipy.special import entr


class PEEK:
    """
    PEEK map computation using the original YOLOv5-era math.

    Given feature maps x with shape (H, W, C):

      1) Positivity shift:
           x_pos = x + abs(min(x))

      2) "Pseudo-entropy" over channels:
           peek = -sum_c entr(x_pos)_c
                = -sum_c ( -x_pos_c * log(x_pos_c) )
                =  sum_c x_pos_c * log(x_pos_c)

    Notes
    - This is intentionally not Shannon entropy: it does not normalize channels.
    - The global shift uses the global minimum over all H, W, C.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def __call__(self, feature_maps_hwc: np.ndarray) -> np.ndarray:
        x = feature_maps_hwc.astype(np.float32, copy=False)
        x_pos = x + float(np.abs(np.min(x)))
        x_pos = x_pos + self.eps
        peek_map = -np.sum(entr(x_pos), axis=-1)
        return peek_map.astype(np.float32, copy=False)


def tensor_to_hwc(t: torch.Tensor) -> Optional[np.ndarray]:
    """
    Convert a latent tensor to HWC numpy for PEEK.

    Accepts:
      - (B, C, H, W) -> use batch 0 -> HWC
      - (C, H, W)    -> HWC
      - (H, W, C)    -> passthrough
      - (B, H, W, C) -> use batch 0 -> HWC
    """
    if not isinstance(t, torch.Tensor):
        return None

    if t.ndim == 4:
        _, d1, d2, d3 = t.shape
        channel_counts = {1, 3, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096}

        if d3 in channel_counts and d1 >= 2 and d2 >= 2:
            return t[0].detach().float().cpu().contiguous().numpy()

        return t[0].detach().float().cpu().permute(1, 2, 0).contiguous().numpy()

    if t.ndim == 3:
        c, h, w = t.shape
        if c <= 4096 and h >= 2 and w >= 2:
            return t.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()

        return t.detach().float().cpu().contiguous().numpy()

    return None


def peek_mean_variance(peek_map_hw: np.ndarray) -> tuple[float, float]:
    """
    Compute the mean and variance of a PEEK map.
    """
    p = np.asarray(peek_map_hw, dtype=np.float64)
    mean = float(np.mean(p))
    variance = float(np.mean((p - mean) ** 2))
    return mean, variance


def peek_stats_from_feature_maps(
    feature_maps_hwc: np.ndarray,
    *,
    peek: Optional[PEEK] = None,
) -> dict[str, float]:
    """
    Compute a PEEK map and its mean/variance from HWC feature maps.
    """
    peek_fn = peek or PEEK()
    peek_map = peek_fn(feature_maps_hwc)
    mean, variance = peek_mean_variance(peek_map)
    return {"peek_mean": mean, "peek_variance": variance}


def peek_stats_from_tensor(
    t: torch.Tensor,
    *,
    peek: Optional[PEEK] = None,
) -> Optional[dict[str, float]]:
    """
    Compute PEEK mean/variance from a latent tensor.
    """
    hwc = tensor_to_hwc(t)
    if hwc is None:
        return None
    return peek_stats_from_feature_maps(hwc, peek=peek)


def relative_variance_contribution(
    variances_by_module: Mapping[int, float],
    *,
    zero_total_value: float = 0.0,
) -> dict[int, float]:
    """
    Compute layer-wise Relative Variance Contribution (RVC).
    """
    total_variance = float(sum(float(v) for v in variances_by_module.values()))
    if total_variance <= 0.0:
        return {int(module): float(zero_total_value) for module in variances_by_module}

    return {
        int(module): float(float(variance) / total_variance)
        for module, variance in variances_by_module.items()
    }
