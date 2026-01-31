"""
Path utilities for the PEEK repo.

Goals:
- Make repo-relative paths work from anywhere (including notebooks/).
- Enforce a single weights directory: <repo_root>/weights
- Force Ultralytics to keep ALL caches, downloads, and runs under repo root
  by setting ULTRALYTICS_DIR before importing ultralytics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


# -----------------------
# Repo root resolution
# -----------------------
def find_repo_root(start: Optional[Union[str, Path]] = None) -> Path:
    """
    Walk upward until a repo marker is found.

    Markers:
      - third_party/
      - .git/
      - pyproject.toml
      - README.md
    """
    if start is None:
        p = Path(__file__).resolve()
    else:
        p = Path(start).resolve()

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


def repo_path(p: Union[str, Path], start: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path relative to repo root unless absolute.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return (find_repo_root(start) / p).resolve()


# -----------------------
# Weights handling
# -----------------------
def weights_dir(start: Optional[Union[str, Path]] = None) -> Path:
    """
    Return <repo_root>/weights and ensure it exists.
    """
    d = find_repo_root(start) / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_weights(weights: Union[str, Path], start: Optional[Union[str, Path]] = None) -> str:
    """
    Resolve a weights argument for Ultralytics.

    Rules:
    - If a path exists (absolute or repo-relative), return its absolute path.
    - If given a bare filename like "yolo26s.pt":
        force it to <repo_root>/weights/yolo26s.pt
      so Ultralytics downloads there.
    - Otherwise, return as-is (Ultralytics hub names).
    """
    w = str(weights)

    # Case 1: path-like (contains / or \)
    if "/" in w or "\\" in w:
        wp = repo_path(w, start)
        if wp.exists():
            return str(wp)
        return str(wp)  # allow Ultralytics to download to this path

    # Case 2: bare filename ending in .pt → pin to weights/
    if w.lower().endswith(".pt"):
        return str((weights_dir(start) / w).resolve())

    # Case 3: model stem or hub name
    return w


# -----------------------
# Ultralytics configuration
# -----------------------
def configure_ultralytics_dir(start: Optional[Union[str, Path]] = None) -> Path:
    """
    Force Ultralytics to store *everything* under repo root.

    Must be called BEFORE importing ultralytics.
    """
    root = find_repo_root(start)
    os.environ.setdefault("ULTRALYTICS_DIR", str(root))
    return root
