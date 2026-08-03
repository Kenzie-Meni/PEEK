#!/usr/bin/env python3
"""Cache YOLO predictions and per-module PEEK proposals for MOT sequences."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from peek.core import PEEK  # noqa: E402
from peek.tracking import PEEKRegionProposer, YOLOPEEKTracker  # noqa: E402
from peek.tracking import tensor_to_hwc  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def source_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def encode_detection(det) -> dict:
    return {
        "xyxy": [float(v) for v in det.xyxy],
        "score": float(det.score),
        "class_id": None if det.cls is None else int(det.cls),
        "source": str(det.source),
        "module": None if det.module is None else int(det.module),
        "modules": [int(m) for m in getattr(det, "modules", ())],
    }


def tensor_summary(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }


def save_latents_npz(path: Path, cache: dict[int, torch.Tensor], fp16: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for module, tensor in cache.items():
        arr = tensor.detach().cpu().numpy()
        if fp16 and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float16)
        arrays[f"m{int(module)}"] = arr
    np.savez_compressed(path, **arrays)


def compute_peek_maps(
    cache: dict[int, torch.Tensor],
    modules: list[int],
    include_dog: bool = True,
    dog_sigma_small: float = 2.0,
    dog_sigma_large: float = 9.0,
) -> dict[str, np.ndarray]:
    peek = PEEK()
    arrays: dict[str, np.ndarray] = {}
    for module in modules:
        tensor = cache.get(module)
        if tensor is None:
            continue
        hwc = tensor_to_hwc(tensor)
        if hwc is None:
            continue
        peek_map = peek(hwc)
        arrays[f"m{int(module)}_raw"] = peek_map.astype(np.float16)
        if include_dog:
            sigma_small = max(0.1, dog_sigma_small)
            sigma_large = max(sigma_small + 0.1, dog_sigma_large)
            local = cv2.GaussianBlur(peek_map, (0, 0), sigma_small)
            background = cv2.GaussianBlur(peek_map, (0, 0), sigma_large)
            arrays[f"m{int(module)}_dog"] = (local - background).astype(np.float16)
    return arrays


def save_peek_maps_npz(path: Path, maps: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **maps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-root", required=True, type=Path)
    parser.add_argument("--weights", default="weights/yolo26s.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs/track/mot17_yolo_peek_cache")
    parser.add_argument("--device", default="0")
    parser.add_argument("--modules", type=int, nargs="+", default=list(range(23)))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--shadow-yolo-conf", type=float, default=0.004)
    parser.add_argument("--classes", type=int, nargs="*", default=[0], help="YOLO classes to keep. Default is MOT person class.")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Also save raw hooked tensors per frame as compressed NPZ. This can be very large.",
    )
    parser.add_argument("--latent-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-peek-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save native-resolution raw PEEK maps per module as compressed NPZ.",
    )
    parser.add_argument(
        "--save-dog-peek-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save unstandardized DoG-filtered PEEK maps for ROI/proposal extraction.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = args.output_dir / "jsonl"
    latent_dir = args.output_dir / "latents"
    peek_map_dir = args.output_dir / "peek_maps"
    proposer = PEEKRegionProposer(
        modules=args.modules,
        z_threshold=1.0,
        min_area=120,
        max_area_fraction=0.35,
        use_dog=True,
        dog_sigma_small=2.0,
        dog_sigma_large=9.0,
        min_extent=0.08,
        max_aspect_ratio=4.0,
        min_short_side=12,
        border_margin=8,
        max_regions_per_module=4,
        focus_z_threshold=0.55,
        focus_local_z_threshold=0.95,
        focus_padding=0.30,
        focus_min_area_fraction=0.40,
        focus_max_regions_per_track=1,
    )
    keep_classes = None if args.classes is None else set(args.classes)
    seq_dirs = [p for p in sorted((args.mot_root / "train").glob("*")) if (p / "img1").exists()]
    seq_dirs = [seq for index, seq in enumerate(seq_dirs) if index % args.num_shards == args.shard_index]

    manifest = {
        "mot_root": str(args.mot_root),
        "weights": str(args.weights),
        "modules": args.modules,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "shadow_yolo_conf": args.shadow_yolo_conf,
        "classes": sorted(keep_classes) if keep_classes is not None else None,
        "max_frames": args.max_frames,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "save_latents": args.save_latents,
        "save_peek_maps": args.save_peek_maps,
        "save_dog_peek_maps": args.save_dog_peek_maps,
        "proposal_extraction_uses_dog": True,
        "sequences": [],
    }

    start_all = time.perf_counter()
    with YOLOPEEKTracker(
        weights=args.weights,
        peek_modules=args.modules,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        proposer=proposer,
        use_shadow_yolo=True,
        shadow_yolo_conf=args.shadow_yolo_conf,
    ) as tracker:
        for seq_dir in seq_dirs:
            paths = source_images(seq_dir / "img1")
            if args.max_frames:
                paths = paths[: args.max_frames]
            out_path = jsonl_dir / f"{seq_dir.name}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            start_seq = time.perf_counter()
            frames = 0
            with out_path.open("w", encoding="utf-8") as handle:
                for frame_index, image_path in enumerate(paths):
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        raise FileNotFoundError(image_path)
                    h, w = frame.shape[:2]
                    tracker.extractor.clear()
                    results = tracker.model.predict(
                        source=frame,
                        imgsz=args.imgsz,
                        device=args.device,
                        conf=min(args.conf, args.shadow_yolo_conf),
                        iou=args.iou,
                        verbose=False,
                    )
                    all_yolo = tracker._detections_from_result(results[0], h, w)
                    if keep_classes is not None:
                        all_yolo = [det for det in all_yolo if det.cls in keep_classes]
                    yolo = [det for det in all_yolo if det.score >= args.conf]
                    shadow = [det for det in all_yolo if args.shadow_yolo_conf <= det.score < args.conf]
                    latents = dict(tracker.extractor.cache)
                    peek = proposer.propose(latents, (h, w), focus_regions=None)

                    latent_path = None
                    if args.save_latents:
                        latent_path = latent_dir / seq_dir.name / f"{image_path.stem}.npz"
                        save_latents_npz(latent_path, latents, fp16=args.latent_fp16)
                    peek_map_path = None
                    if args.save_peek_maps:
                        peek_map_path = peek_map_dir / seq_dir.name / f"{image_path.stem}.npz"
                        save_peek_maps_npz(
                            peek_map_path,
                            compute_peek_maps(
                                latents,
                                args.modules,
                                include_dog=args.save_dog_peek_maps,
                                dog_sigma_small=2.0,
                                dog_sigma_large=9.0,
                            ),
                        )

                    handle.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "image": str(image_path),
                                "height": h,
                                "width": w,
                                "all_yolo_detections": [encode_detection(det) for det in all_yolo],
                                "yolo_detections": [encode_detection(det) for det in yolo],
                                "shadow_yolo_detections": [encode_detection(det) for det in shadow],
                                "peek_detections": [encode_detection(det) for det in peek],
                                "latent_summary": {str(k): tensor_summary(v) for k, v in latents.items()},
                                "latent_path": None if latent_path is None else str(latent_path),
                                "peek_map_path": None if peek_map_path is None else str(peek_map_path),
                            }
                        )
                        + "\n"
                    )
                    frames += 1
            elapsed = time.perf_counter() - start_seq
            manifest["sequences"].append(
                {
                    "name": seq_dir.name,
                    "frames": frames,
                    "jsonl": str(out_path),
                    "elapsed_s": elapsed,
                    "fps": frames / elapsed if elapsed > 0 else 0.0,
                }
            )
            print(f"{seq_dir.name}: cached_frames={frames} elapsed_s={elapsed:.2f} fps={frames / elapsed if elapsed else 0:.2f}")

    manifest["elapsed_s"] = time.perf_counter() - start_all
    (args.output_dir / f"manifest_shard{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"cache_dir={args.output_dir}")


if __name__ == "__main__":
    main()
