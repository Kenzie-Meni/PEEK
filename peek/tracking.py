"""
PEEK-assisted real-time tracking.

This module keeps the tracking mechanism explicit for research use:

1. Run an Ultralytics YOLO model on each frame.
2. Capture selected latent tensors with the hook-based LatentExtractor.
3. Convert selected PEEK maps into recovery regions.
4. Associate YOLO detections and PEEK recovery regions into lightweight tracks.

The intended use is online video tracking, especially cases where YOLO detects an
object in some frames and PEEK can help bridge short detector dropouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Iterable, Optional, Sequence, Union

import cv2
import numpy as np
import torch

from peek.core import PEEK
from peek.extractors.hooks import LatentExtractor
from peek.utils.paths import configure_ultralytics_dir, repo_path, resolve_weights


ArrayLikeFrame = Union[str, Path, np.ndarray]


@dataclass
class TrackedDetection:
    """One detection or PEEK recovery candidate in image coordinates."""

    xyxy: np.ndarray
    score: float
    cls: Optional[int] = None
    mask: Optional[np.ndarray] = None
    source: str = "yolo"
    module: Optional[int] = None
    modules: tuple[int, ...] = field(default_factory=tuple)

    def clipped(self, height: int, width: int) -> "TrackedDetection":
        xyxy = self.xyxy.astype(np.float32, copy=True)
        xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, width - 1)
        xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, height - 1)
        return TrackedDetection(
            xyxy=xyxy,
            score=float(self.score),
            cls=self.cls,
            mask=self.mask,
            source=self.source,
            module=self.module,
            modules=self.modules,
        )


@dataclass
class TrackState:
    """Online state for one tracked object or component."""

    track_id: int
    xyxy: np.ndarray
    score: float
    cls: Optional[int] = None
    mask: Optional[np.ndarray] = None
    source: str = "yolo"
    origin: str = "yolo"
    module: Optional[int] = None
    modules: tuple[int, ...] = field(default_factory=tuple)
    age: int = 1
    hits: int = 1
    missed: int = 0
    history: list[np.ndarray] = field(default_factory=list)

    def update(self, det: TrackedDetection) -> None:
        self.history.append(self.xyxy.copy())
        self.xyxy = det.xyxy.astype(np.float32, copy=True)
        self.score = float(det.score)
        self.cls = det.cls if det.cls is not None else self.cls
        if det.source == "yolo":
            self.mask = det.mask
            self.module = None
            self.modules = ()
        elif det.mask is not None:
            self.mask = det.mask
            self.module = det.module if det.module is not None else self.module
            self.modules = det.modules or (() if det.module is None else (det.module,))
        self.source = det.source
        self.age += 1
        self.hits += 1
        self.missed = 0

    def mark_missed(self) -> None:
        self.history.append(self.xyxy.copy())
        self.age += 1
        self.missed += 1
        self.source = "predicted"
        self.mask = None

    @property
    def confirmed(self) -> bool:
        return self.hits > 1 and self.missed == 0


@dataclass
class FrameTrackingResult:
    """Tracking output for one frame."""

    frame_index: int
    tracks: list[TrackState]
    yolo_detections: list[TrackedDetection]
    shadow_yolo_detections: list[TrackedDetection]
    peek_detections: list[TrackedDetection]
    latency_ms: float


def _add_ultralytics_to_syspath() -> Path:
    root = repo_path(".")
    ulta_root = (root / "third_party" / "ultralytics").resolve()
    if not ulta_root.exists():
        raise FileNotFoundError(f"Missing vendored Ultralytics repo at: {ulta_root}")
    if str(ulta_root) not in sys.path:
        sys.path.insert(0, str(ulta_root))
    return ulta_root


def _hook_target(torch_model: torch.nn.Module) -> torch.nn.Module:
    inner = getattr(torch_model, "model", None)
    if isinstance(inner, torch.nn.Module):
        return inner
    return torch_model


def _read_frame(frame: ArrayLikeFrame) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        return frame
    img = cv2.imread(str(frame))
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {frame}")
    return img


def _as_numpy(x) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def bbox_area(box: np.ndarray) -> float:
    """Area of an xyxy box."""
    x1, y1, x2, y2 = box.astype(np.float32)
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def bbox_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def bbox_overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box a covered by box b."""
    area = bbox_area(a)
    return float(bbox_intersection(a, b) / area) if area > 0 else 0.0


def bbox_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Center distance between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)
    return float(np.hypot(((ax1 + ax2) - (bx1 + bx2)) / 2.0, ((ay1 + ay2) - (by1 + by2)) / 2.0))


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def tensor_to_hwc(t: torch.Tensor) -> Optional[np.ndarray]:
    """Convert captured CHW/BCHW/HWC tensors to HWC numpy for PEEK."""
    if not torch.is_tensor(t):
        return None
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        return None
    if t.shape[0] <= 4096 and t.shape[1] >= 2 and t.shape[2] >= 2:
        return t.detach().float().cpu().permute(1, 2, 0).contiguous().numpy()
    return t.detach().float().cpu().contiguous().numpy()


class PEEKRegionProposer:
    """
    Convert captured latent tensors into PEEK recovery regions.

    The proposal rule is intentionally simple and online-friendly: compute a PEEK
    map per requested module, standardize it, threshold high-activation regions,
    clean the binary mask, and emit contour boxes.
    """

    def __init__(
        self,
        modules: Sequence[int],
        z_threshold: float = 1.0,
        min_area: int = 200,
        max_area_fraction: float = 0.45,
        use_dog: bool = False,
        dog_sigma_small: float = 2.0,
        dog_sigma_large: float = 9.0,
        min_extent: float = 0.08,
        max_aspect_ratio: float = 8.0,
        min_short_side: int = 8,
        border_margin: int = 8,
        kernel_size: int = 5,
        max_regions_per_module: int = 8,
        focus_z_threshold: float = 0.35,
        focus_local_z_threshold: float = 0.75,
        focus_padding: float = 0.35,
        focus_min_area_fraction: float = 0.30,
        focus_max_regions_per_track: int = 1,
    ):
        self.modules = list(modules)
        self.z_threshold = float(z_threshold)
        self.min_area = int(min_area)
        self.max_area_fraction = float(max_area_fraction)
        self.use_dog = bool(use_dog)
        self.dog_sigma_small = float(dog_sigma_small)
        self.dog_sigma_large = float(dog_sigma_large)
        self.min_extent = float(min_extent)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_short_side = int(min_short_side)
        self.border_margin = int(border_margin)
        self.kernel_size = int(kernel_size)
        self.max_regions_per_module = int(max_regions_per_module)
        self.focus_z_threshold = float(focus_z_threshold)
        self.focus_local_z_threshold = float(focus_local_z_threshold)
        self.focus_padding = float(focus_padding)
        self.focus_min_area_fraction = float(focus_min_area_fraction)
        self.focus_max_regions_per_track = int(focus_max_regions_per_track)
        self.peek = PEEK()

    def propose(
        self,
        latents: dict[int, torch.Tensor],
        frame_shape: tuple[int, int],
        focus_regions: Optional[Sequence[np.ndarray]] = None,
    ) -> list[TrackedDetection]:
        h, w = frame_shape
        detections: list[TrackedDetection] = []
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        for module in self.modules:
            tensor = latents.get(module)
            if tensor is None:
                continue
            hwc = tensor_to_hwc(tensor)
            if hwc is None:
                continue

            peek_map = self.peek(hwc)
            peek_map = cv2.resize(peek_map, (w, h), interpolation=cv2.INTER_LINEAR)
            if self.use_dog:
                sigma_small = max(0.1, self.dog_sigma_small)
                sigma_large = max(sigma_small + 0.1, self.dog_sigma_large)
                local = cv2.GaussianBlur(peek_map, (0, 0), sigma_small)
                background = cv2.GaussianBlur(peek_map, (0, 0), sigma_large)
                peek_map = local - background
            std = float(np.std(peek_map))
            if std <= 1e-12:
                continue

            z = (peek_map - float(np.mean(peek_map))) / std
            binary = (z > self.z_threshold).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Suzuki-Abe border following, exposed through OpenCV findContours.
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, rw, rh = cv2.boundingRect(contour)
                if rw <= 1 or rh <= 1:
                    continue
                if min(rw, rh) < self.min_short_side:
                    continue
                bbox_area = float(rw * rh)
                frame_area = float(w * h)
                covers_frame = rw >= 0.90 * w and rh >= 0.90 * h
                if bbox_area >= self.max_area_fraction * frame_area or covers_frame:
                    continue
                if (
                    x <= self.border_margin
                    or y <= self.border_margin
                    or x + rw >= w - self.border_margin
                    or y + rh >= h - self.border_margin
                ):
                    continue
                extent = float(area / bbox_area) if bbox_area else 0.0
                if extent < self.min_extent:
                    continue
                aspect_ratio = max(float(rw) / float(rh), float(rh) / float(rw))
                if aspect_ratio > self.max_aspect_ratio:
                    continue
                roi_z = z[y : y + rh, x : x + rw]
                contour_mask = np.zeros((rh, rw), dtype=np.uint8)
                shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
                cv2.drawContours(contour_mask, [shifted], -1, 1, thickness=-1)
                local_support = float(roi_z[contour_mask > 0].mean()) if np.any(contour_mask) else 0.0
                candidates.append((local_support * np.sqrt(area), local_support, contour, x, y, rw, rh))

            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
            for _, local_support, contour, x, y, rw, rh in ranked[: self.max_regions_per_module]:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
                xyxy = np.array([x, y, x + rw, y + rh], dtype=np.float32)
                score = min(1.0, max(0.0, local_support / (self.z_threshold + 3.0)))
                detections.append(
                    TrackedDetection(
                        xyxy=xyxy,
                        score=score,
                        mask=mask,
                        source="peek",
                        module=module,
                    ).clipped(h, w)
                )

            if focus_regions:
                detections.extend(self._propose_focus_regions(z, focus_regions, module, (h, w), kernel))

        return detections

    def _propose_focus_regions(
        self,
        z: np.ndarray,
        focus_regions: Sequence[np.ndarray],
        module: int,
        frame_shape: tuple[int, int],
        kernel: np.ndarray,
    ) -> list[TrackedDetection]:
        """Recover weak-but-local PEEK support around tracks YOLO just lost."""
        h, w = frame_shape
        detections: list[TrackedDetection] = []
        min_focus_area = max(16.0, self.min_area * self.focus_min_area_fraction)

        for region in focus_regions:
            x1, y1, x2, y2 = region.astype(np.float32)
            bw = max(2.0, x2 - x1)
            bh = max(2.0, y2 - y1)
            pad_x = self.focus_padding * bw
            pad_y = self.focus_padding * bh
            rx1 = max(0, int(np.floor(x1 - pad_x)))
            ry1 = max(0, int(np.floor(y1 - pad_y)))
            rx2 = min(w, int(np.ceil(x2 + pad_x)))
            ry2 = min(h, int(np.ceil(y2 + pad_y)))
            if rx2 - rx1 < self.min_short_side or ry2 - ry1 < self.min_short_side:
                continue

            crop = z[ry1:ry2, rx1:rx2]
            local_std = float(np.std(crop))
            if local_std <= 1e-12:
                continue
            local_z = (crop - float(np.mean(crop))) / local_std
            weak_global = crop > self.focus_z_threshold
            weak_local = local_z > self.focus_local_z_threshold
            binary = np.logical_or(weak_global, weak_local).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_focus_area:
                    continue
                x, y, rw, rh = cv2.boundingRect(contour)
                if min(rw, rh) < self.min_short_side:
                    continue
                aspect_ratio = max(float(rw) / float(rh), float(rh) / float(rw))
                if aspect_ratio > self.max_aspect_ratio:
                    continue
                bbox_area = float(rw * rh)
                extent = float(area / bbox_area) if bbox_area else 0.0
                if extent < self.min_extent:
                    continue

                contour_mask = np.zeros((rh, rw), dtype=np.uint8)
                shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
                cv2.drawContours(contour_mask, [shifted], -1, 1, thickness=-1)
                roi_global = crop[y : y + rh, x : x + rw]
                roi_local = local_z[y : y + rh, x : x + rw]
                valid = contour_mask > 0
                global_support = float(roi_global[valid].mean()) if np.any(valid) else 0.0
                local_support = float(roi_local[valid].mean()) if np.any(valid) else 0.0

                abs_x = rx1 + x
                abs_y = ry1 + y
                cx = abs_x + rw / 2.0
                cy = abs_y + rh / 2.0
                expected_cx = (x1 + x2) / 2.0
                expected_cy = (y1 + y2) / 2.0
                dist = float(np.hypot(cx - expected_cx, cy - expected_cy))
                expected_diag = float(np.hypot(bw, bh))
                distance_penalty = dist / max(expected_diag, 1.0)
                score = global_support + 0.65 * local_support - 0.25 * distance_penalty
                candidates.append((score, global_support, local_support, contour, abs_x, abs_y, rw, rh))

            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
            for score, global_support, local_support, contour, abs_x, abs_y, rw, rh in ranked[
                : self.focus_max_regions_per_track
            ]:
                if score <= 0:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                shifted = contour + np.array([[[rx1, ry1]]], dtype=contour.dtype)
                cv2.drawContours(mask, [shifted], -1, 1, thickness=-1)
                xyxy = np.array([abs_x, abs_y, abs_x + rw, abs_y + rh], dtype=np.float32)
                det_score = min(1.0, max(0.0, 0.12 + (global_support + local_support) / 8.0))
                detections.append(
                    TrackedDetection(
                        xyxy=xyxy,
                        score=det_score,
                        mask=mask,
                        source="peek",
                        module=module,
                    ).clipped(h, w)
                )

        return detections


class PEEKAssistedTracker:
    """Lightweight online tracker with YOLO-first and PEEK-recovery association."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        peek_iou_threshold: float = 0.15,
        max_missed: int = 8,
        min_yolo_conf: float = 0.25,
        spawn_peek_tracks: bool = True,
        min_peek_score: float = 0.12,
    ):
        self.iou_threshold = float(iou_threshold)
        self.peek_iou_threshold = float(peek_iou_threshold)
        self.max_missed = int(max_missed)
        self.min_yolo_conf = float(min_yolo_conf)
        self.spawn_peek_tracks = bool(spawn_peek_tracks)
        self.min_peek_score = float(min_peek_score)
        self.tracks: list[TrackState] = []
        self.next_id = 1

    def update(
        self,
        yolo_detections: Sequence[TrackedDetection],
        peek_detections: Sequence[TrackedDetection],
    ) -> list[TrackState]:
        active_tracks = list(self.tracks)
        unmatched_tracks = set(range(len(active_tracks)))
        unmatched_yolo = set(range(len(yolo_detections)))

        # First associate model detections. These carry the strongest class/mask evidence.
        for ti, di in self._greedy_matches(active_tracks, yolo_detections, unmatched_tracks, unmatched_yolo, self.iou_threshold):
            active_tracks[ti].update(yolo_detections[di])
            unmatched_tracks.discard(ti)
            unmatched_yolo.discard(di)

        # Then let PEEK recover tracks that YOLO missed.
        unmatched_peek = set(range(len(peek_detections)))
        for ti, di in self._greedy_matches(active_tracks, peek_detections, unmatched_tracks, unmatched_peek, self.peek_iou_threshold):
            active_tracks[ti].update(peek_detections[di])
            unmatched_tracks.discard(ti)
            unmatched_peek.discard(di)

        for ti in sorted(unmatched_tracks):
            active_tracks[ti].mark_missed()

        for di in sorted(unmatched_yolo):
            det = yolo_detections[di]
            if det.score < self.min_yolo_conf:
                continue
            active_tracks.append(
                TrackState(
                    track_id=self.next_id,
                    xyxy=det.xyxy.astype(np.float32, copy=True),
                    score=float(det.score),
                    cls=det.cls,
                    mask=det.mask,
                    source=det.source,
                    origin="yolo",
                    module=det.module,
                    modules=det.modules,
                )
            )
            self.next_id += 1

        if self.spawn_peek_tracks:
            for di in sorted(unmatched_peek):
                det = peek_detections[di]
                if det.score < self.min_peek_score:
                    continue
                active_tracks.append(
                    TrackState(
                        track_id=self.next_id,
                        xyxy=det.xyxy.astype(np.float32, copy=True),
                        score=float(det.score),
                        cls=det.cls,
                        mask=det.mask,
                        source=det.source,
                        origin="peek",
                        module=det.module,
                        modules=det.modules or (() if det.module is None else (det.module,)),
                    )
                )
                self.next_id += 1

        self.tracks = [t for t in active_tracks if t.missed <= self.max_missed]
        return list(self.tracks)

    def _greedy_matches(
        self,
        tracks: Sequence[TrackState],
        detections: Sequence[TrackedDetection],
        track_indices: set[int],
        detection_indices: set[int],
        threshold: float,
    ) -> list[tuple[int, int]]:
        pairs: list[tuple[float, int, int]] = []
        for ti in track_indices:
            for di in detection_indices:
                track = tracks[ti]
                det = detections[di]
                if track.cls is not None and det.cls is not None and track.cls != det.cls:
                    continue
                iou = bbox_iou(track.xyxy, det.xyxy)
                if iou >= threshold:
                    pairs.append((iou, ti, di))

        pairs.sort(reverse=True, key=lambda p: p[0])
        matches = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)
        return matches


class YOLOPEEKTracker:
    """
    Real-time YOLO + PEEK tracking wrapper.

    Example:
        tracker = YOLOPEEKTracker("weights/yolo26s.pt", peek_modules=[16, 19, 22])
        for result in tracker.track_video("input.mp4"):
            ...
    """

    def __init__(
        self,
        weights: Union[str, Path] = "yolo26s.pt",
        peek_modules: Sequence[int] = (16, 19, 22),
        device: Union[str, int] = "",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        fp16_latents: bool = False,
        proposer: Optional[PEEKRegionProposer] = None,
        tracker: Optional[PEEKAssistedTracker] = None,
        gate_peek_to_yolo_union: bool = True,
        peek_gate_padding: float = 0.06,
        peek_gate_min_iou: float = 0.01,
        gate_peek_by_anchor_distance: bool = True,
        peek_anchor_max_distance_frac: float = 0.12,
        peek_focus_max_tracks: int = 8,
        peek_focus_max_missed: int = 4,
        peek_nms_iou: float = 0.35,
        peek_nms_max_candidates: int = 80,
        union_cluster_peek: bool = False,
        peek_cluster_iou: float = 0.10,
        peek_cluster_center_frac: float = 0.35,
        peek_cluster_min_modules: int = 1,
        peek_cluster_min_area: int = 180,
        peek_cluster_min_short_side: int = 12,
        peek_cluster_max_area_fraction: float = 0.12,
        shadow_yolo_conf: float = 0.06,
        use_shadow_yolo: bool = True,
        use_shadow_yolo_as_peek_anchor: bool = True,
        suppress_peek_yolo_iou: float = 0.10,
        suppress_peek_yolo_containment: float = 0.55,
    ):
        configure_ultralytics_dir()
        _add_ultralytics_to_syspath()

        from ultralytics import YOLO  # type: ignore

        self.weights = resolve_weights(weights)
        self.model = YOLO(self.weights)
        if not hasattr(self.model, "model"):
            raise ValueError("Ultralytics YOLO object missing .model")

        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.gate_peek_to_yolo_union = bool(gate_peek_to_yolo_union)
        self.peek_gate_padding = float(peek_gate_padding)
        self.peek_gate_min_iou = float(peek_gate_min_iou)
        self.gate_peek_by_anchor_distance = bool(gate_peek_by_anchor_distance)
        self.peek_anchor_max_distance_frac = float(peek_anchor_max_distance_frac)
        self.peek_focus_max_tracks = int(peek_focus_max_tracks)
        self.peek_focus_max_missed = int(peek_focus_max_missed)
        self.peek_nms_iou = float(peek_nms_iou)
        self.peek_nms_max_candidates = int(peek_nms_max_candidates)
        self.union_cluster_peek = bool(union_cluster_peek)
        self.peek_cluster_iou = float(peek_cluster_iou)
        self.peek_cluster_center_frac = float(peek_cluster_center_frac)
        self.peek_cluster_min_modules = int(peek_cluster_min_modules)
        self.peek_cluster_min_area = int(peek_cluster_min_area)
        self.peek_cluster_min_short_side = int(peek_cluster_min_short_side)
        self.peek_cluster_max_area_fraction = float(peek_cluster_max_area_fraction)
        self.shadow_yolo_conf = float(shadow_yolo_conf)
        self.use_shadow_yolo = bool(use_shadow_yolo)
        self.use_shadow_yolo_as_peek_anchor = bool(use_shadow_yolo_as_peek_anchor)
        self.suppress_peek_yolo_iou = float(suppress_peek_yolo_iou)
        self.suppress_peek_yolo_containment = float(suppress_peek_yolo_containment)

        target = _hook_target(self.model.model)
        self.extractor = LatentExtractor(
            target,
            modules=list(peek_modules),
            to_cpu=True,
            fp16=fp16_latents,
        )
        self.extractor.start()

        self.proposer = proposer or PEEKRegionProposer(peek_modules)
        self.tracker = tracker or PEEKAssistedTracker(min_yolo_conf=conf)
        self.frame_index = 0

    def close(self) -> None:
        self.extractor.stop()

    def __enter__(self) -> "YOLOPEEKTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @torch.no_grad()
    def process_frame(self, frame: ArrayLikeFrame) -> FrameTrackingResult:
        start = time.perf_counter()
        im = _read_frame(frame)
        h, w = im.shape[:2]

        self.extractor.clear()
        predict_conf = min(self.conf, self.shadow_yolo_conf) if self.use_shadow_yolo else self.conf
        results = self.model.predict(
            source=im,
            imgsz=self.imgsz,
            device=self.device,
            conf=predict_conf,
            iou=self.iou,
            verbose=False,
        )
        result = results[0]
        all_yolo_dets = self._detections_from_result(result, h, w)
        yolo_dets = [det for det in all_yolo_dets if det.score >= self.conf]
        shadow_yolo_dets = [
            det
            for det in all_yolo_dets
            if self.use_shadow_yolo and self.shadow_yolo_conf <= det.score < self.conf
        ]
        focus_regions = self._peek_focus_regions(yolo_dets)
        peek_dets = self.proposer.propose(dict(self.extractor.cache), (h, w), focus_regions=focus_regions)
        if self.union_cluster_peek:
            peek_dets = self._union_clustered_peek_detections(peek_dets, h, w)
        else:
            peek_dets = self._nms_peek_detections(peek_dets)
        peek_dets = self._suppress_peek_explained_by_yolo(peek_dets, yolo_dets)
        if self.use_shadow_yolo:
            peek_dets = self._apply_shadow_yolo_support(peek_dets, shadow_yolo_dets)
        if self.gate_peek_by_anchor_distance:
            support_dets = shadow_yolo_dets if self.use_shadow_yolo_as_peek_anchor else ()
            peek_dets = self._filter_peek_by_anchor_distance(peek_dets, yolo_dets, h, w, support_dets)
        if self.gate_peek_to_yolo_union:
            support_dets = shadow_yolo_dets if self.use_shadow_yolo_as_peek_anchor else ()
            peek_dets = self._filter_peek_to_yolo_union(peek_dets, yolo_dets, h, w, support_dets)
        tracks = self.tracker.update(yolo_dets, peek_dets)

        latency_ms = (time.perf_counter() - start) * 1000.0
        out = FrameTrackingResult(
            frame_index=self.frame_index,
            tracks=tracks,
            yolo_detections=yolo_dets,
            shadow_yolo_detections=shadow_yolo_dets,
            peek_detections=peek_dets,
            latency_ms=latency_ms,
        )
        self.frame_index += 1
        return out

    def track_video(self, source: Union[str, Path, int]) -> Iterable[FrameTrackingResult]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.process_frame(frame)
        finally:
            cap.release()

    def _detections_from_result(self, result, height: int, width: int) -> list[TrackedDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _as_numpy(boxes.xyxy).astype(np.float32)
        conf = _as_numpy(boxes.conf).astype(np.float32)
        cls = _as_numpy(boxes.cls).astype(np.float32)

        masks_obj = getattr(result, "masks", None)
        masks = None
        if masks_obj is not None:
            masks = _as_numpy(masks_obj.data).astype(np.uint8)
            if masks.ndim == 2:
                masks = masks[None, :, :]
            if masks.shape[-2:] != (height, width):
                resized = []
                for m in masks:
                    resized.append(cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST))
                masks = np.stack(resized, axis=0)

        detections: list[TrackedDetection] = []
        for i, box in enumerate(xyxy):
            mask = masks[i] if masks is not None and i < len(masks) else None
            detections.append(
                TrackedDetection(
                    xyxy=box,
                    score=float(conf[i]),
                    cls=int(cls[i]),
                    mask=mask,
                    source="yolo",
                ).clipped(height, width)
            )
        return detections

    def _peek_focus_regions(self, yolo_detections: Sequence[TrackedDetection]) -> list[np.ndarray]:
        """Tracks not matched by YOLO become local weak-evidence PEEK search regions."""
        candidates: list[tuple[int, int, np.ndarray]] = []
        for track in self.tracker.tracks:
            if getattr(track, "origin", track.source) != "yolo":
                continue
            if track.hits < 2 or track.missed > self.peek_focus_max_missed:
                continue
            best_iou = 0.0
            for det in yolo_detections:
                if track.cls is not None and det.cls is not None and track.cls != det.cls:
                    continue
                best_iou = max(best_iou, bbox_iou(track.xyxy, det.xyxy))
            if best_iou < self.tracker.iou_threshold:
                candidates.append((track.missed, -track.hits, track.xyxy.astype(np.float32, copy=True)))
        candidates.sort(key=lambda item: item[:2])
        return [box for _, _, box in candidates[: self.peek_focus_max_tracks]]

    def _nms_peek_detections(self, peek_detections: Sequence[TrackedDetection]) -> list[TrackedDetection]:
        """Suppress stacked PEEK boxes from multiple modules or nearby contours."""
        if self.peek_nms_iou <= 0 or len(peek_detections) <= 1:
            return list(peek_detections)
        ranked = sorted(peek_detections, key=lambda det: (det.score, bbox_area(det.xyxy)), reverse=True)
        if self.peek_nms_max_candidates > 0:
            ranked = ranked[: self.peek_nms_max_candidates]
        kept: list[TrackedDetection] = []
        for det in ranked:
            if all(bbox_iou(det.xyxy, existing.xyxy) < self.peek_nms_iou for existing in kept):
                kept.append(det)
        return kept

    def _union_clustered_peek_detections(
        self,
        peek_detections: Sequence[TrackedDetection],
        height: int,
        width: int,
    ) -> list[TrackedDetection]:
        """Union PEEK boxes that describe the same local recovery region."""
        candidates = [
            det
            for det in peek_detections
            if bbox_area(det.xyxy) >= self.peek_cluster_min_area
            and min(float(det.xyxy[2] - det.xyxy[0]), float(det.xyxy[3] - det.xyxy[1]))
            >= self.peek_cluster_min_short_side
        ]
        if len(candidates) <= 1:
            return candidates

        ranked = sorted(candidates, key=lambda det: (det.score, bbox_area(det.xyxy)), reverse=True)
        if self.peek_nms_max_candidates > 0:
            ranked = ranked[: self.peek_nms_max_candidates]

        parent = list(range(len(ranked)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, det in enumerate(ranked):
            for j in range(i + 1, len(ranked)):
                other = ranked[j]
                overlap = bbox_iou(det.xyxy, other.xyxy)
                if overlap >= self.peek_cluster_iou:
                    union(i, j)
                    continue

                dx1, dy1, dx2, dy2 = det.xyxy.astype(np.float32)
                ox1, oy1, ox2, oy2 = other.xyxy.astype(np.float32)
                det_diag = float(np.hypot(dx2 - dx1, dy2 - dy1))
                other_diag = float(np.hypot(ox2 - ox1, oy2 - oy1))
                center_limit = self.peek_cluster_center_frac * max(det_diag, other_diag)
                if center_limit > 0 and bbox_center_distance(det.xyxy, other.xyxy) <= center_limit:
                    union(i, j)

        groups: dict[int, list[TrackedDetection]] = {}
        for index, det in enumerate(ranked):
            groups.setdefault(find(index), []).append(det)

        merged: list[TrackedDetection] = []
        for group in groups.values():
            modules = sorted(
                {
                    int(module)
                    for det in group
                    for module in (det.modules or (() if det.module is None else (det.module,)))
                    if module is not None
                }
            )
            if len(modules) < self.peek_cluster_min_modules:
                continue

            boxes = np.stack([det.xyxy.astype(np.float32) for det in group], axis=0)
            xyxy = np.array(
                [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()],
                dtype=np.float32,
            )
            if bbox_area(xyxy) < self.peek_cluster_min_area:
                continue
            if bbox_area(xyxy) > self.peek_cluster_max_area_fraction * float(height * width):
                continue
            if min(float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])) < self.peek_cluster_min_short_side:
                continue

            mask = None
            masks = [det.mask for det in group if det.mask is not None]
            if masks:
                mask = np.zeros((height, width), dtype=np.uint8)
                for item in masks:
                    mask[item > 0] = 1

            best = max(group, key=lambda det: det.score)
            merged.append(
                TrackedDetection(
                    xyxy=xyxy,
                    score=max(float(det.score) for det in group),
                    cls=best.cls,
                    mask=mask,
                    source="peek",
                    module=modules[0] if len(modules) == 1 else None,
                    modules=tuple(modules),
                ).clipped(height, width)
            )

        merged.sort(key=lambda det: (det.score, bbox_area(det.xyxy)), reverse=True)
        return merged

    def _suppress_peek_explained_by_yolo(
        self,
        peek_detections: Sequence[TrackedDetection],
        yolo_detections: Sequence[TrackedDetection],
    ) -> list[TrackedDetection]:
        """Drop PEEK proposals that duplicate an already-visible YOLO detection."""
        if not yolo_detections:
            return list(peek_detections)

        kept: list[TrackedDetection] = []
        for det in peek_detections:
            redundant = False
            for yolo in yolo_detections:
                iou = bbox_iou(det.xyxy, yolo.xyxy)
                containment = bbox_overlap_fraction(det.xyxy, yolo.xyxy)
                if iou >= self.suppress_peek_yolo_iou or containment >= self.suppress_peek_yolo_containment:
                    redundant = True
                    break
            if not redundant:
                kept.append(det)
        return kept

    def _apply_shadow_yolo_support(
        self,
        peek_detections: Sequence[TrackedDetection],
        shadow_yolo_detections: Sequence[TrackedDetection],
    ) -> list[TrackedDetection]:
        """Let subthreshold YOLO boxes provide class/score hints to nearby PEEK proposals."""
        if not shadow_yolo_detections:
            return list(peek_detections)

        supported: list[TrackedDetection] = []
        for det in peek_detections:
            best: Optional[TrackedDetection] = None
            best_support = 0.0
            for shadow in shadow_yolo_detections:
                iou = bbox_iou(det.xyxy, shadow.xyxy)
                containment = max(
                    bbox_overlap_fraction(det.xyxy, shadow.xyxy),
                    bbox_overlap_fraction(shadow.xyxy, det.xyxy),
                )
                support = max(iou, 0.5 * containment)
                if support > best_support:
                    best = shadow
                    best_support = support

            if best is None or best_support <= 0:
                supported.append(det)
                continue

            boosted = TrackedDetection(
                xyxy=det.xyxy,
                score=max(float(det.score), min(1.0, 0.5 * float(det.score) + 0.5 * float(best.score))),
                cls=best.cls if det.cls is None else det.cls,
                mask=det.mask,
                source=det.source,
                module=det.module,
                modules=det.modules,
            )
            supported.append(boosted)
        return supported

    def _filter_peek_by_anchor_distance(
        self,
        peek_detections: Sequence[TrackedDetection],
        yolo_detections: Sequence[TrackedDetection],
        height: int,
        width: int,
        support_detections: Sequence[TrackedDetection] = (),
    ) -> list[TrackedDetection]:
        """
        Keep PEEK boxes near current detections or recently YOLO-supported tracks.

        This preserves PEEK-only recovery when YOLO misses the component, while
        rejecting isolated PEEK contours far from the object neighborhood.
        """
        anchors: list[np.ndarray] = [det.xyxy for det in yolo_detections]
        anchors.extend(det.xyxy for det in support_detections)
        for track in self.tracker.tracks:
            if getattr(track, "origin", track.source) == "yolo" and track.missed <= self.tracker.max_missed:
                anchors.append(track.xyxy)
        if not anchors:
            return []

        frame_diag = float(np.hypot(width, height))
        base_distance = self.peek_anchor_max_distance_frac * frame_diag
        kept: list[TrackedDetection] = []
        for det in peek_detections:
            dx1, dy1, dx2, dy2 = det.xyxy.astype(np.float32)
            dcx, dcy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            det_diag = float(np.hypot(dx2 - dx1, dy2 - dy1))
            for anchor in anchors:
                ax1, ay1, ax2, ay2 = anchor.astype(np.float32)
                acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
                anchor_diag = float(np.hypot(ax2 - ax1, ay2 - ay1))
                allowed = base_distance + 0.5 * (det_diag + anchor_diag)
                if float(np.hypot(dcx - acx, dcy - acy)) <= allowed:
                    kept.append(det)
                    break
        return kept

    def _filter_peek_to_yolo_union(
        self,
        peek_detections: Sequence[TrackedDetection],
        yolo_detections: Sequence[TrackedDetection],
        height: int,
        width: int,
        support_detections: Sequence[TrackedDetection] = (),
    ) -> list[TrackedDetection]:
        """Keep PEEK proposals only inside/near the union of YOLO-supported objects."""
        gates: list[np.ndarray] = [det.xyxy for det in yolo_detections]
        gates.extend(det.xyxy for det in support_detections)
        for track in self.tracker.tracks:
            if getattr(track, "origin", track.source) == "yolo" and track.missed <= self.tracker.max_missed:
                gates.append(track.xyxy)
        if not gates:
            return []

        pad_x = self.peek_gate_padding * width
        pad_y = self.peek_gate_padding * height
        expanded = []
        for box in gates:
            x1, y1, x2, y2 = box.astype(np.float32)
            expanded.append(
                np.array(
                    [
                        max(0.0, x1 - pad_x),
                        max(0.0, y1 - pad_y),
                        min(float(width - 1), x2 + pad_x),
                        min(float(height - 1), y2 + pad_y),
                    ],
                    dtype=np.float32,
                )
            )

        kept: list[TrackedDetection] = []
        for det in peek_detections:
            x1, y1, x2, y2 = det.xyxy.astype(np.float32)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for gate in expanded:
                gx1, gy1, gx2, gy2 = gate
                center_inside = gx1 <= cx <= gx2 and gy1 <= cy <= gy2
                if center_inside or bbox_iou(det.xyxy, gate) >= self.peek_gate_min_iou:
                    kept.append(det)
                    break
        return kept


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackState],
    names: Optional[dict[int, str]] = None,
    draw_masks: bool = True,
) -> np.ndarray:
    """Draw bbox and optional mask tracks for quick demos/video export."""
    out = frame.copy()
    overlay = out.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.xyxy.astype(int)
        color = (40, 220, 40) if track.source == "yolo" else (30, 140, 255)
        if track.source == "predicted":
            color = (160, 160, 160)
        if draw_masks and track.mask is not None:
            overlay[track.mask > 0] = color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cls_name = names.get(track.cls, str(track.cls)) if names and track.cls is not None else "obj"
        label = f"{track.track_id}:{cls_name}:{track.source}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if draw_masks:
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    return out


__all__ = [
    "FrameTrackingResult",
    "PEEKAssistedTracker",
    "PEEKRegionProposer",
    "TrackState",
    "TrackedDetection",
    "YOLOPEEKTracker",
    "bbox_area",
    "bbox_intersection",
    "bbox_iou",
    "bbox_overlap_fraction",
    "draw_tracks",
    "mask_to_bbox",
]
