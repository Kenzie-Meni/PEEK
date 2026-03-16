"""
Plotting helpers for dense-head PEEK analytic studies.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from .analytic import rb_overlay
from .utils.paths import repo_path


def centered_norm(arr: np.ndarray) -> TwoSlopeNorm:
    vmax = np.nanmax(np.abs(arr)) + 1e-12
    return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)


def load_geometry_npz(npz_path: Union[str, Path]) -> dict[str, Any]:
    with np.load(repo_path(npz_path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def plot_dense_head_alignment_maps(
    image_path: Union[str, Path],
    npz_path: Union[str, Path],
    *,
    variant: str = "trained",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot input, PEEK, T1/T2/T3 sign overlays, and continuous heatmaps.
    """
    image = Image.open(repo_path(image_path)).convert("RGB")
    data = load_geometry_npz(npz_path)

    t1_key = "T1_up_train" if variant == "trained" else "T1_up_rand"
    prod_key = "prod_sign_train" if variant == "trained" else "prod_sign_rand"
    T1_up = np.asarray(data[t1_key], dtype=np.float32)
    T2_up = np.asarray(data["T2_up"], dtype=np.float32)
    T3_up = np.asarray(data["T3_up"], dtype=np.float32)
    peek_map = np.asarray(data["PEEK_img"], dtype=np.float32)
    prod_sign = np.asarray(data[prod_key]).astype(bool)

    H, W = peek_map.shape
    ov_T1 = rb_overlay(T1_up > 0, H, W)
    ov_T2 = rb_overlay(T2_up > 0, H, W)
    ov_T3 = rb_overlay(T3_up > 0, H, W)
    ov_prod = rb_overlay(prod_sign, H, W)

    product_map = T1_up * T2_up * T3_up

    fig, axes = plt.subplots(2, 6, figsize=(20, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(peek_map, alpha=0.7, cmap="jet")
    axes[0, 1].set_title("PEEK")
    axes[0, 2].imshow(image)
    axes[0, 2].imshow(ov_T1, alpha=0.5)
    axes[0, 2].set_title("T1 sign")
    axes[0, 3].imshow(image)
    axes[0, 3].imshow(ov_T2, alpha=0.5)
    axes[0, 3].set_title("T2 sign")
    axes[0, 4].imshow(image)
    axes[0, 4].imshow(ov_T3, alpha=0.5)
    axes[0, 4].set_title("T3 sign")
    axes[0, 5].imshow(image)
    axes[0, 5].imshow(ov_prod, alpha=0.5)
    axes[0, 5].set_title("Product sign")

    heatmaps = [
        (T1_up, "T1 heatmap"),
        (T2_up, "T2 heatmap"),
        (T3_up, "T3 heatmap"),
        (product_map, "T1*T2*T3"),
    ]

    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    for offset, (arr, title) in enumerate(heatmaps, start=2):
        im = axes[1, offset].imshow(arr, cmap="coolwarm", norm=centered_norm(arr))
        axes[1, offset].set_title(title)
        fig.colorbar(im, ax=axes[1, offset], fraction=0.046, pad=0.02)

    for row in axes:
        for ax in row:
            ax.axis("off")

    fig.suptitle(f"{Path(image_path).name} — {variant}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        out = repo_path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def render_geometry_directory(
    npz_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    variants: tuple[str, ...] = ("trained", "random"),
) -> list[Path]:
    """
    Render all saved geometry NPZ files in a directory to PNGs.
    """
    npz_dir_p = repo_path(npz_dir)
    out_dir_p = repo_path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    for npz_path in sorted(npz_dir_p.glob("*.npz")):
        data = load_geometry_npz(npz_path)
        image_path = str(data["image_path"])
        stem = npz_path.stem
        for variant in variants:
            out_path = out_dir_p / f"{stem}_{variant}.png"
            plot_dense_head_alignment_maps(
                image_path=image_path,
                npz_path=npz_path,
                variant=variant,
                save_path=out_path,
            )
            rendered.append(out_path)
    return rendered


def _read_numeric_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def compare_geometry_csvs(
    train_csv: Union[str, Path],
    random_csv: Union[str, Path],
) -> list[dict[str, float]]:
    """
    Aggregate trained-vs-random metric comparisons.
    """
    train_rows = _read_numeric_csv(repo_path(train_csv))
    random_rows = _read_numeric_csv(repo_path(random_csv))

    train_by_image = {row["image"]: row for row in train_rows}
    random_by_image = {row["image"]: row for row in random_rows}
    shared_images = sorted(set(train_by_image) & set(random_by_image))
    if not shared_images:
        return []

    metrics: list[str] = []
    for key in train_by_image[shared_images[0]]:
        if key in {"image", "true", "pred", "pred_synset", "image_path"}:
            continue
        try:
            float(train_by_image[shared_images[0]][key])
            float(random_by_image[shared_images[0]][key])
            metrics.append(key)
        except (TypeError, ValueError):
            continue

    rows: list[dict[str, float]] = []
    for metric in metrics:
        a = np.asarray([float(train_by_image[image][metric]) for image in shared_images], dtype=np.float64)
        b = np.asarray([float(random_by_image[image][metric]) for image in shared_images], dtype=np.float64)
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
        rows.append(
            {
                "metric": metric,
                "mean_train": float(np.mean(a)),
                "mean_random": float(np.mean(b)),
                "delta_mean": float(np.mean(a - b)),
                "corr_train_random": corr,
            }
        )
    return rows
