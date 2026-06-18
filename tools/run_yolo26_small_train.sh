#!/usr/bin/env bash
set -euo pipefail

cd /home/rwhite/NFS/All_Team/Ryan/PEEK

exec /home/rwhite/mambaforge/envs/peek/bin/python -u tools/train_yolo26_small.py \
  --epochs 100 \
  --imgsz 640 \
  --device 0 \
  --batch 16 \
  --name peek_yolo26s_bbox
