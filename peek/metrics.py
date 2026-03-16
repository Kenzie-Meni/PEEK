"""
PEEK downstream metrics.

Computes per-image PEEK mean/variance and layer-wise Relative Variance
Contribution (RVC) from saved latent pickle files.
"""

from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .core import PEEK, peek_stats_from_tensor, relative_variance_contribution
from .utils.paths import repo_path


def _iter_feature_pickles(feature_folder: Path, limit: int = 0) -> list[Path]:
    paths = sorted(p for p in feature_folder.iterdir() if p.is_file() and p.suffix.lower() == ".pkl")
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_feature_folder_metrics(
    feature_folder: Union[str, Path],
    modules: Optional[Sequence[int]] = None,
    out_csv: Optional[Union[str, Path]] = None,
    summary_csv: Optional[Union[str, Path]] = None,
    limit: int = 0,
    verbose: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    Compute PEEK mean, PEEK variance, and per-image RVC from saved latent pickles.

    Args:
        feature_folder: Folder containing per-image latent pickles.
        modules: Optional subset of module indices to score.
        out_csv: Optional repo-root-relative CSV path for per-image metrics.
        summary_csv: Optional repo-root-relative CSV path for aggregated per-module stats.
        limit: Max number of pickle files to process (0 = all).

    Returns:
        A dict with:
          - "per_image": one row per image/module
          - "summary": one row per module with mean/std aggregates
    """
    feature_folder_p = repo_path(feature_folder)
    if not feature_folder_p.exists():
        raise FileNotFoundError(f"Missing feature_folder: {feature_folder_p}")

    paths = _iter_feature_pickles(feature_folder_p, limit=limit)
    if not paths:
        raise FileNotFoundError(f"No pickle files found in: {feature_folder_p}")

    selected_modules = None if modules is None else {int(module) for module in modules}
    peek = PEEK()
    per_image_rows: list[dict[str, Any]] = []
    by_module: dict[int, list[dict[str, float]]] = defaultdict(list)

    for pkl_path in paths:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            if verbose:
                print(f"[metrics] skip non-dict pickle: {pkl_path}")
            continue

        image_name = pkl_path.stem
        module_stats: dict[int, dict[str, float]] = {}

        for module, value in data.items():
            module_idx = int(module)
            if selected_modules is not None and module_idx not in selected_modules:
                continue

            stats = peek_stats_from_tensor(value, peek=peek)
            if stats is None:
                if verbose:
                    print(f"[metrics] skip unsupported latent for module {module_idx}: {pkl_path.name}")
                continue

            module_stats[module_idx] = stats

        if not module_stats:
            if verbose:
                print(f"[metrics] no usable modules in: {pkl_path}")
            continue

        rvc_by_module = relative_variance_contribution(
            {module: stats["peek_variance"] for module, stats in module_stats.items()}
        )

        for module_idx in sorted(module_stats):
            stats = module_stats[module_idx]
            row = {
                "image": image_name,
                "module": module_idx,
                "peek_mean": stats["peek_mean"],
                "peek_variance": stats["peek_variance"],
                "rvc": rvc_by_module[module_idx],
            }
            per_image_rows.append(row)
            by_module[module_idx].append(
                {
                    "peek_mean": stats["peek_mean"],
                    "peek_variance": stats["peek_variance"],
                    "rvc": rvc_by_module[module_idx],
                }
            )

        if verbose:
            print(f"[metrics] scored: {pkl_path.name}")

    if not per_image_rows:
        raise ValueError(f"No PEEK metrics could be computed from: {feature_folder_p}")

    summary_rows: list[dict[str, Any]] = []
    for module_idx in sorted(by_module):
        values = by_module[module_idx]
        means = np.asarray([row["peek_mean"] for row in values], dtype=np.float64)
        variances = np.asarray([row["peek_variance"] for row in values], dtype=np.float64)
        rvcs = np.asarray([row["rvc"] for row in values], dtype=np.float64)
        summary_rows.append(
            {
                "module": module_idx,
                "n_images": int(len(values)),
                "mean_peek_mean": float(np.mean(means)),
                "std_peek_mean": float(np.std(means)),
                "mean_peek_variance": float(np.mean(variances)),
                "std_peek_variance": float(np.std(variances)),
                "mean_rvc": float(np.mean(rvcs)),
                "std_rvc": float(np.std(rvcs)),
            }
        )

    if out_csv is not None:
        _write_csv(
            repo_path(out_csv),
            per_image_rows,
            ["image", "module", "peek_mean", "peek_variance", "rvc"],
        )

    if summary_csv is not None:
        _write_csv(
            repo_path(summary_csv),
            summary_rows,
            [
                "module",
                "n_images",
                "mean_peek_mean",
                "std_peek_mean",
                "mean_peek_variance",
                "std_peek_variance",
                "mean_rvc",
                "std_rvc",
            ],
        )

    return {"per_image": per_image_rows, "summary": summary_rows}
