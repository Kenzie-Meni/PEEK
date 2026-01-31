"""
Weights management.

Centralizes weight-file policy for this repo:

- Canonical location: <repo_root>/weights/
- If a local .pt is provided elsewhere, copy it into weights/ (once).
- If a model name like "yolo11s.pt" is provided and missing, trigger
  Ultralytics auto-download, then copy the downloaded .pt into weights/.

This keeps extractors + plotting consistent regardless of notebook cwd.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union


# -----------------------
# Repo root + paths
# -----------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Find repo root by walking upward until a marker is found.

    Markers:
      - third_party/
      - .git/
      - pyproject.toml
      - README.md / readme.md
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
        if (parent / "README.md").exists() or (parent / "readme.md").exists():
            return parent

    return Path.cwd().resolve()


def repo_path(p: Union[str, Path]) -> Path:
    """
    Resolve a path relative to repo root unless already absolute.
    """
    p = Path(p)
    if p.is_absolute():
        return p.resolve()
    return (find_repo_root() / p).resolve()


def weights_dir() -> Path:
    """
    Return <repo_root>/weights (create if missing).
    """
    d = find_repo_root() / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


# -----------------------
# Ultralytics weights
# -----------------------
def _search_for_downloaded_pt(filename: str) -> Optional[Path]:
    """
    Best-effort search for a downloaded .pt after Ultralytics auto-download.
    """
    candidates = []

    # Common Ultralytics cache/config locations
    candidates.append(Path.home() / ".cache" / "ultralytics")
    candidates.append(Path.home() / ".config" / "ultralytics")

    # Repo + cwd (sometimes users download into repo root or cwd)
    rr = find_repo_root()
    candidates.append(rr)
    candidates.append(Path.cwd())

    # Also search weights dir itself
    candidates.append(weights_dir())

    for base in candidates:
        try:
            if base.exists():
                hit = next(base.rglob(filename), None)
                if hit and hit.exists():
                    return hit.resolve()
        except Exception:
            continue

    return None


def ensure_ultralytics_weights(weights: Union[str, Path]) -> Path:
    """
    Ensure weights exist in <repo_root>/weights/<name>.pt and return that path.

    Args:
      weights:
        - A local file path to a .pt
        - Or a model name like "yolo11s.pt", "yolo26s.pt" (Ultralytics will download)

    Returns:
      Path to the canonical weights file under weights/.
    """
    w = Path(weights)
    dest = (weights_dir() / w.name).resolve()

    # If user passed a path that exists, copy it into weights/ once.
    if w.exists():
        w = w.resolve()
        if w == dest:
            return dest
        if not dest.exists():
            shutil.copy2(w, dest)
        return dest

    # If it already exists in weights/, done.
    if dest.exists():
        return dest

    # Trigger Ultralytics auto-download for named weights
    # and then locate where it downloaded.
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise ImportError(
            "Ultralytics is required to auto-download weights. "
            "Install ultralytics or vendor it correctly."
        ) from e

    # Constructing YOLO(weights) usually triggers download if missing.
    _ = YOLO(str(weights))

    found = _search_for_downloaded_pt(w.name)
    if found is None or not found.exists():
        raise FileNotFoundError(
            f"Requested weights '{weights}' were not found locally and could not be located after download."
        )

    # Copy the located file into weights/
    if found.resolve() != dest:
        shutil.copy2(found, dest)

    return dest
