#!/usr/bin/env python3
"""Stage common MOT-style tracking benchmarks and pretrained YOLO weights."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadItem:
    name: str
    url: str
    filename: str


BENCHMARKS = {
    "mot17": DownloadItem(
        name="mot17",
        url="https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip",
        filename="MOT17.zip",
    ),
    "mot16": DownloadItem(
        name="mot16",
        url="https://bj.bcebos.com/v1/paddledet/data/mot/MOT16.zip",
        filename="MOT16.zip",
    ),
    "image_lists": DownloadItem(
        name="image_lists",
        url="https://bj.bcebos.com/v1/paddledet/data/mot/image_lists.zip",
        filename="image_lists.zip",
    ),
}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest}", flush=True)
        return
    partial = dest.with_suffix(dest.suffix + ".aria2")
    if partial.exists():
        print(f"resuming: {dest}", flush=True)
    run(["aria2c", "-c", "-x", "8", "-s", "8", "-k", "1M", "-o", dest.name, "-d", str(dest.parent), url])


def unzip(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / f".{zip_path.stem}.unzipped"
    if marker.exists():
        print(f"already unzipped: {zip_path}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"unzipping: {zip_path} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def download_yolo_weights(names: list[str], weights_dir: Path) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ultralytics"))
    from ultralytics import YOLO  # type: ignore

    for name in names:
        target = weights_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"exists: {target}", flush=True)
            continue
        print(f"downloading YOLO weight via Ultralytics: {name}", flush=True)
        model = YOLO(name)
        src = Path(model.ckpt_path or name)
        if src.exists() and src.resolve() != target.resolve():
            target.write_bytes(src.read_bytes())
        print(f"ready: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/tracking_benchmarks"))
    parser.add_argument("--benchmarks", nargs="+", default=["mot17", "mot16", "image_lists"])
    parser.add_argument("--weights-dir", type=Path, default=Path("PEEK/weights"))
    parser.add_argument("--yolo-weights", nargs="+", default=["yolo11s.pt", "yolo11x.pt"])
    parser.add_argument("--no-unzip", action="store_true")
    args = parser.parse_args()

    archives = args.root / "archives"
    extracted = args.root / "extracted"
    args.root.mkdir(parents=True, exist_ok=True)

    for key in args.benchmarks:
        item = BENCHMARKS[key]
        dest = archives / item.filename
        download(item.url, dest)
        if not args.no_unzip:
            unzip(dest, extracted)

    download_yolo_weights(args.yolo_weights, args.weights_dir)
    print(f"benchmark root: {args.root.resolve()}", flush=True)
    print(f"weights dir: {args.weights_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
