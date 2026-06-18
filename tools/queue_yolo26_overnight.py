#!/usr/bin/env python3
"""Run a two-GPU overnight YOLO26 experiment queue with Discord updates."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/rwhite/mambaforge/envs/peek/bin/python")
DATA = Path("/home/rwhite/NFS/All_Team/Ryan/datasets/peek_yolo26_bbox/data.yaml")
RUN_ROOT = REPO / "runs" / "detect"
STATE_PATH = RUN_ROOT / "overnight_yolo26_queue_state.json"
LOCK_PATH = RUN_ROOT / "overnight_yolo26_queue_state.lock"
EVENT_LOG = RUN_ROOT / "overnight_yolo26_queue_events.log"
BASELINE_BEST = {
    "run": "peek_yolo26s_bbox_mapmax_640_fresh_tmux",
    "epoch": 88,
    "map50": 0.62209,
    "map5095": 0.43970,
    "precision": 0.73082,
    "recall": 0.56298,
}


@dataclass(frozen=True)
class Job:
    key: str
    label: str
    device: int
    weights: str
    epochs: int
    patience: int
    args: tuple[str, ...]

    @property
    def run_dir(self) -> Path:
        return RUN_ROOT / self.key

    def command(self) -> list[str]:
        return [
            str(PYTHON),
            "-u",
            "tools/train_yolo26_small.py",
            "--data",
            str(DATA),
            "--weights",
            self.weights,
            "--epochs",
            str(self.epochs),
            "--imgsz",
            "640",
            "--device",
            str(self.device),
            "--batch",
            "16",
            "--name",
            self.key,
            "--patience",
            str(self.patience),
            "--save-period",
            "10",
            *self.args,
        ]


JOBS = [
    Job(
        key="peek_yolo26s_bbox_fresh_seed31_default_640_tmux",
        label="fresh seed31 defaultish 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=220,
        patience=90,
        args=(
            "--optimizer", "auto", "--cos-lr", "--close-mosaic", "20",
            "--seed", "31", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_fresh_seed43_lowerase_640_tmux",
        label="fresh seed43 low-erasing 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=220,
        patience=90,
        args=(
            "--optimizer", "auto", "--cos-lr", "--close-mosaic", "20",
            "--scale", "0.35", "--translate", "0.08", "--mosaic", "0.90",
            "--erasing", "0.05", "--hsv-h", "0.010", "--hsv-s", "0.55",
            "--hsv-v", "0.30", "--seed", "43", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_fresh_seed59_nomosaiclate_640_tmux",
        label="fresh seed59 longer-mosaic 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=90,
        args=(
            "--optimizer", "auto", "--cos-lr", "--close-mosaic", "35",
            "--scale", "0.45", "--translate", "0.10", "--mosaic", "1.0",
            "--erasing", "0.15", "--seed", "59", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_fresh_seed71_sgd_640_tmux",
        label="fresh seed71 SGD mAP 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=220,
        patience=90,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.40",
            "--translate", "0.08", "--mosaic", "0.85", "--erasing", "0.05",
            "--hsv-h", "0.010", "--hsv-s", "0.50", "--hsv-v", "0.28",
            "--seed", "71", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_best_refine_defaultlr_640_tmux",
        label="best checkpoint defaultish refine 640",
        device=0,
        weights="runs/detect/peek_yolo26s_bbox_mapmax_640_fresh_tmux/weights/best.pt",
        epochs=90,
        patience=35,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "15",
            "--lr0", "0.0010", "--lrf", "0.05", "--scale", "0.25",
            "--translate", "0.05", "--mosaic", "0.35", "--erasing", "0.0",
            "--hsv-h", "0.008", "--hsv-s", "0.35", "--hsv-v", "0.20",
            "--seed", "83", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_smallobj_to_map_slow_640_tmux",
        label="small-object checkpoint slow mAP recovery 640",
        device=1,
        weights="runs/detect/peek_yolo26s_bbox_smallobj_recall_640_tmux/weights/best.pt",
        epochs=90,
        patience=35,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "15",
            "--lr0", "0.0008", "--lrf", "0.08", "--scale", "0.10",
            "--translate", "0.02", "--mosaic", "0.10", "--erasing", "0.0",
            "--box", "7.5", "--cls", "0.45", "--dfl", "1.5",
            "--hsv-h", "0.004", "--hsv-s", "0.20", "--hsv-v", "0.12",
            "--seed", "97", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_fresh_seed101_lowcolor_640_tmux",
        label="fresh seed101 low-color 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=220,
        patience=85,
        args=(
            "--optimizer", "auto", "--cos-lr", "--close-mosaic", "20",
            "--scale", "0.30", "--translate", "0.06", "--mosaic", "0.75",
            "--erasing", "0.0", "--hsv-h", "0.006", "--hsv-s", "0.30",
            "--hsv-v", "0.18", "--seed", "101", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_fresh_seed113_highrecall_640_tmux",
        label="fresh seed113 recall-leaning 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=220,
        patience=85,
        args=(
            "--optimizer", "auto", "--cos-lr", "--close-mosaic", "25",
            "--scale", "0.50", "--translate", "0.10", "--mosaic", "1.0",
            "--erasing", "0.0", "--box", "8.5", "--cls", "0.55",
            "--hsv-h", "0.010", "--hsv-s", "0.45", "--hsv-v", "0.25",
            "--seed", "113", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_best_micro_seed127_640_tmux",
        label="best checkpoint micro seed127 640",
        device=0,
        weights="runs/detect/peek_yolo26s_bbox_mapmax_640_fresh_tmux/weights/best.pt",
        epochs=50,
        patience=18,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "5",
            "--lr0", "0.00025", "--lrf", "0.10", "--scale", "0.04",
            "--translate", "0.01", "--mosaic", "0.0", "--erasing", "0.0",
            "--hsv-h", "0.004", "--hsv-s", "0.18", "--hsv-v", "0.10",
            "--seed", "127", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_fresh_seed139_default_640_tmux",
        label="fresh seed139 defaultish 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=220,
        patience=85,
        args=(
            "--optimizer", "auto", "--cos-lr", "--close-mosaic", "20",
            "--seed", "139", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_sgd_seed151_close25_640_tmux",
        label="iter2 SGD seed151 close25 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.40",
            "--translate", "0.08", "--mosaic", "0.85", "--erasing", "0.05",
            "--hsv-h", "0.010", "--hsv-s", "0.50", "--hsv-v", "0.28",
            "--seed", "151", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_sgd_seed163_close35_640_tmux",
        label="iter2 SGD seed163 close35 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "35",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.45",
            "--translate", "0.10", "--mosaic", "1.0", "--erasing", "0.05",
            "--hsv-h", "0.010", "--hsv-s", "0.50", "--hsv-v", "0.28",
            "--seed", "163", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_sgd_seed173_lowaug_640_tmux",
        label="iter2 SGD seed173 low-aug 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.005", "--lrf", "0.05", "--scale", "0.30",
            "--translate", "0.06", "--mosaic", "0.65", "--erasing", "0.0",
            "--hsv-h", "0.008", "--hsv-s", "0.40", "--hsv-v", "0.22",
            "--seed", "173", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_sgd_seed181_thrusterlean_640_tmux",
        label="iter2 SGD seed181 thruster-leaning 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "30",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.40",
            "--translate", "0.08", "--mosaic", "0.85", "--erasing", "0.0",
            "--box", "8.5", "--cls", "0.55", "--dfl", "1.5",
            "--hsv-h", "0.010", "--hsv-s", "0.45", "--hsv-v", "0.25",
            "--seed", "181", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_best71_refine_mosaic20_640_tmux",
        label="iter2 seed71 best refine mosaic20 640",
        device=0,
        weights="runs/detect/peek_yolo26s_bbox_fresh_seed71_sgd_640_tmux/weights/best.pt",
        epochs=100,
        patience=40,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "15",
            "--lr0", "0.0009", "--lrf", "0.05", "--scale", "0.20",
            "--translate", "0.04", "--mosaic", "0.20", "--erasing", "0.0",
            "--hsv-h", "0.006", "--hsv-s", "0.30", "--hsv-v", "0.18",
            "--seed", "191", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_best71_micro_640_tmux",
        label="iter2 seed71 best micro-polish 640",
        device=1,
        weights="runs/detect/peek_yolo26s_bbox_fresh_seed71_sgd_640_tmux/weights/best.pt",
        epochs=60,
        patience=22,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "5",
            "--lr0", "0.00025", "--lrf", "0.10", "--scale", "0.04",
            "--translate", "0.01", "--mosaic", "0.0", "--erasing", "0.0",
            "--hsv-h", "0.004", "--hsv-s", "0.18", "--hsv-v", "0.10",
            "--seed", "193", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_sgd_seed197_highbox_640_tmux",
        label="iter2 SGD seed197 high-box 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.40",
            "--translate", "0.08", "--mosaic", "0.85", "--erasing", "0.05",
            "--box", "9.0", "--cls", "0.50", "--dfl", "1.5",
            "--hsv-h", "0.010", "--hsv-s", "0.50", "--hsv-v", "0.28",
            "--seed", "197", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter2_sgd_seed211_default_640_tmux",
        label="iter2 SGD seed211 default recipe 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.40",
            "--translate", "0.08", "--mosaic", "0.85", "--erasing", "0.05",
            "--hsv-h", "0.010", "--hsv-s", "0.50", "--hsv-v", "0.28",
            "--seed", "211", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_sgd_seed223_generalize_640_tmux",
        label="iter3 fresh SGD seed223 generalization-preserving 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.35",
            "--translate", "0.06", "--mosaic", "0.80", "--erasing", "0.0",
            "--hsv-h", "0.008", "--hsv-s", "0.40", "--hsv-v", "0.22",
            "--seed", "223", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_sgd_seed233_confusion_640_tmux",
        label="iter3 fresh SGD seed233 antenna-solar separation 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.0055", "--lrf", "0.04", "--scale", "0.35",
            "--translate", "0.06", "--mosaic", "0.75", "--erasing", "0.0",
            "--box", "7.5", "--cls", "0.65", "--dfl", "1.5",
            "--hsv-h", "0.008", "--hsv-s", "0.38", "--hsv-v", "0.20",
            "--seed", "233", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_seed71_refine_lowaug_640_tmux",
        label="iter3 seed71 low-augmentation transfer refine 640",
        device=0,
        weights="runs/detect/peek_yolo26s_bbox_fresh_seed71_sgd_640_tmux/weights/best.pt",
        epochs=100,
        patience=40,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "10",
            "--lr0", "0.00055", "--lrf", "0.08", "--scale", "0.08",
            "--translate", "0.02", "--mosaic", "0.05", "--erasing", "0.0",
            "--hsv-h", "0.004", "--hsv-s", "0.18", "--hsv-v", "0.10",
            "--seed", "227", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_current_lowlr_generalize_640_tmux",
        label="iter3 current best low-LR generalization polish 640",
        device=1,
        weights="runs/detect/peek_yolo26s_bbox_iter2_best71_refine_mosaic20_640_tmux/weights/best.pt",
        epochs=80,
        patience=35,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "8",
            "--lr0", "0.00035", "--lrf", "0.08", "--scale", "0.04",
            "--translate", "0.01", "--mosaic", "0.0", "--erasing", "0.0",
            "--box", "7.5", "--cls", "0.55", "--dfl", "1.5",
            "--hsv-h", "0.003", "--hsv-s", "0.14", "--hsv-v", "0.08",
            "--seed", "241", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_current_thruster_refine_640_tmux",
        label="iter3 current best tiny-thruster refine 640",
        device=0,
        weights="runs/detect/peek_yolo26s_bbox_iter2_best71_refine_mosaic20_640_tmux/weights/best.pt",
        epochs=90,
        patience=35,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "10",
            "--lr0", "0.00045", "--lrf", "0.08", "--scale", "0.08",
            "--translate", "0.02", "--mosaic", "0.05", "--erasing", "0.0",
            "--box", "8.5", "--cls", "0.60", "--dfl", "1.5",
            "--hsv-h", "0.004", "--hsv-s", "0.18", "--hsv-v", "0.10",
            "--seed", "229", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_sgd_seed251_thruster_640_tmux",
        label="iter3 fresh SGD seed251 tiny-thruster pressure 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "30",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.45",
            "--translate", "0.08", "--mosaic", "0.90", "--erasing", "0.0",
            "--box", "8.5", "--cls", "0.55", "--dfl", "1.5",
            "--hsv-h", "0.010", "--hsv-s", "0.45", "--hsv-v", "0.25",
            "--seed", "251", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_sgd_seed239_box8_640_tmux",
        label="iter3 fresh SGD seed239 balanced high-box 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=240,
        patience=95,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "30",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.40",
            "--translate", "0.08", "--mosaic", "0.85", "--erasing", "0.02",
            "--box", "8.0", "--cls", "0.55", "--dfl", "1.5",
            "--hsv-h", "0.010", "--hsv-s", "0.45", "--hsv-v", "0.25",
            "--seed", "239", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter3_seed71_thruster_refine_640_tmux",
        label="iter3 seed71 tiny-thruster transfer refine 640",
        device=1,
        weights="runs/detect/peek_yolo26s_bbox_fresh_seed71_sgd_640_tmux/weights/best.pt",
        epochs=100,
        patience=40,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "12",
            "--lr0", "0.0007", "--lrf", "0.08", "--scale", "0.10",
            "--translate", "0.02", "--mosaic", "0.10", "--erasing", "0.0",
            "--box", "8.25", "--cls", "0.55", "--dfl", "1.5",
            "--hsv-h", "0.004", "--hsv-s", "0.20", "--hsv-v", "0.12",
            "--seed", "257", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_sgd_seed263_mildbox_640_tmux",
        label="iter4 fresh SGD seed263 mild-box seed223-style 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=250,
        patience=100,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.35",
            "--translate", "0.06", "--mosaic", "0.80", "--erasing", "0.0",
            "--box", "7.8", "--cls", "0.50", "--dfl", "1.5",
            "--hsv-h", "0.008", "--hsv-s", "0.40", "--hsv-v", "0.22",
            "--seed", "263", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_sgd_seed269_recallbalance_640_tmux",
        label="iter4 fresh SGD seed269 recall-balanced 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=250,
        patience=100,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.0058", "--lrf", "0.04", "--scale", "0.32",
            "--translate", "0.06", "--mosaic", "0.78", "--erasing", "0.0",
            "--box", "7.5", "--cls", "0.58", "--dfl", "1.5",
            "--hsv-h", "0.008", "--hsv-s", "0.38", "--hsv-v", "0.20",
            "--seed", "269", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_best223_tiny_polish_640_tmux",
        label="iter4 seed223 tiny-part low-LR polish 640",
        device=0,
        weights="runs/detect/peek_yolo26s_bbox_iter3_sgd_seed223_generalize_640_tmux/weights/best.pt",
        epochs=90,
        patience=35,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "8",
            "--lr0", "0.00045", "--lrf", "0.08", "--scale", "0.06",
            "--translate", "0.015", "--mosaic", "0.03", "--erasing", "0.0",
            "--box", "8.0", "--cls", "0.55", "--dfl", "1.5",
            "--hsv-h", "0.003", "--hsv-s", "0.14", "--hsv-v", "0.08",
            "--seed", "271", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_seed239_map50_recover_640_tmux",
        label="iter4 seed239 mAP50 recovery polish 640",
        device=1,
        weights="runs/detect/peek_yolo26s_bbox_iter3_sgd_seed239_box8_640_tmux/weights/best.pt",
        epochs=90,
        patience=35,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "8",
            "--lr0", "0.00040", "--lrf", "0.08", "--scale", "0.05",
            "--translate", "0.015", "--mosaic", "0.02", "--erasing", "0.0",
            "--box", "7.2", "--cls", "0.50", "--dfl", "1.5",
            "--hsv-h", "0.003", "--hsv-s", "0.14", "--hsv-v", "0.08",
            "--seed", "277", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_sgd_seed281_lowcolor_precision_640_tmux",
        label="iter4 fresh SGD seed281 low-color precision 640",
        device=0,
        weights="weights/yolo26s.pt",
        epochs=250,
        patience=100,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.30",
            "--translate", "0.05", "--mosaic", "0.72", "--erasing", "0.0",
            "--box", "7.5", "--cls", "0.50", "--dfl", "1.5",
            "--hsv-h", "0.005", "--hsv-s", "0.24", "--hsv-v", "0.14",
            "--seed", "281", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_sgd_seed283_copypaste_tiny_640_tmux",
        label="iter4 fresh SGD seed283 mild copy-paste tiny parts 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=250,
        patience=100,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.35",
            "--translate", "0.06", "--mosaic", "0.80", "--copy-paste", "0.08",
            "--erasing", "0.0", "--box", "7.8", "--cls", "0.55", "--dfl", "1.5",
            "--hsv-h", "0.008", "--hsv-s", "0.36", "--hsv-v", "0.20",
            "--seed", "283", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_best223_recall_nudge_640_tmux",
        label="iter4 seed223 recall nudge 640",
        device=0,
        weights="runs/detect/peek_yolo26s_bbox_iter3_sgd_seed223_generalize_640_tmux/weights/best.pt",
        epochs=80,
        patience=30,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "10",
            "--lr0", "0.00055", "--lrf", "0.08", "--scale", "0.10",
            "--translate", "0.02", "--mosaic", "0.08", "--erasing", "0.0",
            "--box", "7.8", "--cls", "0.62", "--dfl", "1.5",
            "--hsv-h", "0.004", "--hsv-s", "0.18", "--hsv-v", "0.10",
            "--seed", "293", "--deterministic",
        ),
    ),
    Job(
        key="peek_yolo26s_bbox_iter4_sgd_seed307_seed223clone_640_tmux",
        label="iter4 fresh SGD seed307 seed223 clone check 640",
        device=1,
        weights="weights/yolo26s.pt",
        epochs=250,
        patience=100,
        args=(
            "--optimizer", "SGD", "--cos-lr", "--close-mosaic", "25",
            "--lr0", "0.006", "--lrf", "0.04", "--scale", "0.35",
            "--translate", "0.06", "--mosaic", "0.80", "--erasing", "0.0",
            "--hsv-h", "0.008", "--hsv-s", "0.40", "--hsv-v", "0.22",
            "--seed", "307", "--deterministic",
        ),
    ),
]


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def post_discord(webhook: str | None, content: str) -> None:
    content = "fit-afrl: " + content
    log(content)
    if not webhook:
        return
    for start in range(0, len(content), 1900):
        chunk = content[start : start + 1900]
        subprocess.run(
            [
                "curl", "-fsS", "--connect-timeout", "10", "--max-time", "30",
                "-A", "fit-afrl-yolo-overnight/1.0", "-H", "Content-Type: application/json",
                "-d", json.dumps({"content": chunk}), webhook,
            ],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def read_results(run_dir: Path) -> dict | None:
    path = run_dir / "results.csv"
    if not path.exists():
        return None
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(path.open(newline=""))]
    if not rows:
        return None
    best = max(rows, key=lambda r: (float(r["metrics/mAP50(B)"]), float(r["metrics/mAP50-95(B)"])))
    latest = rows[-1]
    return {
        "epoch": int(float(best["epoch"])),
        "latest_epoch": int(float(latest["epoch"])),
        "map50": float(best["metrics/mAP50(B)"]),
        "map5095": float(best["metrics/mAP50-95(B)"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
    }


def metric_tuple(metrics: dict) -> tuple[float, float, int]:
    return (float(metrics["map50"]), float(metrics["map5095"]), int(metrics["epoch"]))


def load_state_unlocked() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "done": [],
        "running": {},
        "best": BASELINE_BEST,
        "started_at": time.time(),
    }


def save_state_unlocked(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_PATH)


@contextmanager
def state_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_state() -> dict:
    with state_lock():
        return load_state_unlocked()


def save_state(state: dict) -> None:
    with state_lock():
        save_state_unlocked(state)


def update_state(mutator) -> dict:
    with state_lock():
        state = load_state_unlocked()
        mutator(state)
        save_state_unlocked(state)
        return state


def run_job(job: Job, webhook: str | None, state: dict) -> None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "train.log"
    log(
        f"Starting overnight run on GPU{job.device}: {job.label}\n"
        f"Run: {job.key}\n"
        f"Current overall best to beat: mAP50={state['best']['map50']:.5f}, "
        f"mAP50-95={state['best']['map5095']:.5f} ({state['best']['run']}).",
    )
    state = update_state(
        lambda current: current.setdefault("running", {}).__setitem__(
            job.key, {"device": job.device, "started_at": time.time(), "label": job.label}
        )
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            job.command(),
            cwd=REPO,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_reported: tuple[float, float, int] | None = None
        while process.poll() is None:
            metrics = read_results(job.run_dir)
            if metrics and metric_tuple(metrics) > metric_tuple(state["best"]):
                if last_reported != metric_tuple(metrics):
                    state = update_state(
                        lambda current: current.__setitem__("best", {"run": job.key, **metrics})
                        if metric_tuple(metrics) > metric_tuple(current["best"])
                        else None
                    )
                    if state["best"]["run"] == job.key and metric_tuple(state["best"]) == metric_tuple(metrics):
                        post_discord(
                            webhook,
                            f"New overnight overall best: {job.label}\n"
                            f"Epoch: {metrics['epoch']}\n"
                            f"mAP50: {metrics['map50']:.5f}\n"
                            f"mAP50-95: {metrics['map5095']:.5f}\n"
                            f"Precision: {metrics['precision']:.5f}\n"
                            f"Recall: {metrics['recall']:.5f}",
                        )
                    last_reported = metric_tuple(metrics)
            time.sleep(60)

        returncode = process.wait()
    metrics = read_results(job.run_dir)
    def finish_mutation(current: dict) -> None:
        current.setdefault("running", {}).pop(job.key, None)
        done = current.setdefault("done", [])
        if job.key not in done:
            done.append(job.key)
        if metrics and metric_tuple(metrics) > metric_tuple(current["best"]):
            current["best"] = {"run": job.key, **metrics}

    state = update_state(finish_mutation)
    if metrics:
        log(
            f"Finished overnight run on GPU{job.device}: {job.label}\n"
            f"Return code: {returncode}\n"
            f"Best epoch: {metrics['epoch']}\n"
            f"mAP50: {metrics['map50']:.5f}\n"
            f"mAP50-95: {metrics['map5095']:.5f}\n"
            f"Precision: {metrics['precision']:.5f}\n"
            f"Recall: {metrics['recall']:.5f}\n"
            f"Next: GPU{job.device} will take the next queued run, or we are done if the queue/time window is exhausted.",
        )
    else:
        log(
            f"Finished overnight run on GPU{job.device}: {job.label}, but no metrics were found. "
            f"Return code: {returncode}. Log: {log_path}",
        )


def main() -> None:
    webhook = os.environ.get("DISCORD_WEBHOOK")
    duration_hours = float(os.environ.get("YOLO26_QUEUE_HOURS", "12"))
    stop_at = time.time() + duration_hours * 3600
    def init_mutation(current: dict) -> None:
        current.setdefault("started_at", time.time())
        current.setdefault("done", [])
        current.setdefault("running", {})
        current.setdefault("best", BASELINE_BEST)

    state = update_state(init_mutation)

    log(
        f"Starting 12-hour YOLO26 overnight queue: {len(JOBS)} targeted 640px runs across 2 GPUs. "
        "Discord will only report new overall bests and the final out-of-jobs summary.",
    )

    queues = {
        0: [job for job in JOBS if job.device == 0],
        1: [job for job in JOBS if job.device == 1],
    }
    children: list[subprocess.Popen] = []
    for device, jobs in queues.items():
        worker = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(device),
                str(stop_at),
                json.dumps([job.key for job in jobs]),
            ],
            cwd=REPO,
            env={**os.environ, "YOLO26_QUEUE_PARENT": "0"},
        )
        children.append(worker)
    for child in children:
        child.wait()

    state = load_state()
    best = state["best"]
    post_discord(
        webhook,
        f"Overnight YOLO26 queue complete or time window exhausted.\n"
        f"Best tracked run: {best['run']}\n"
        f"Epoch: {best['epoch']}\n"
        f"mAP50: {best['map50']:.5f}\n"
        f"mAP50-95: {best['map5095']:.5f}\n"
        f"Precision: {best.get('precision', 0.0):.5f}\n"
        f"Recall: {best.get('recall', 0.0):.5f}",
    )


def worker_main() -> None:
    device = int(sys.argv[2])
    stop_at = float(sys.argv[3])
    keys = json.loads(sys.argv[4])
    by_key = {job.key: job for job in JOBS}
    webhook = os.environ.get("DISCORD_WEBHOOK")
    for key in keys:
        if time.time() >= stop_at:
            log(f"GPU{device} overnight queue reached the time limit before starting {key}.")
            return
        state = load_state()
        if key in state.get("done", []):
            continue
        run_job(by_key[key], webhook, state)
    log(f"GPU{device} is out of queued jobs for this YOLO26 iteration.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main()
    else:
        main()
