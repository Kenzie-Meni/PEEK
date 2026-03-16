#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/rwhite/Documents/PEEK"
PYTHON_BIN="/home/rwhite/.local/share/mamba/envs/yolo26/bin/python"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "$REPO_ROOT"
mkdir -p analysis/vgg16_dense_head analysis/resnet50_dense_head analysis/convnext_base_dense_head weights

run_train() {
  local model_name="$1"
  local checkpoint_out="$2"
  local last_checkpoint_out="$3"
  local history_out="$4"
  local batch_size="$5"
  local epochs="$6"
  local head_warmup_epochs="$7"
  local learning_rate="$8"
  local weight_decay="$9"
  local dropout="${10}"
  local patience="${11}"
  local log_file="${12}"

  "$PYTHON_BIN" -c "
from peek import ImagenetteFineTuneConfig, train_imagenette_model
cfg = ImagenetteFineTuneConfig(
    model_name='${model_name}',
    data_root='../datasets/imagenette2',
    checkpoint_out='${checkpoint_out}',
    last_checkpoint_out='${last_checkpoint_out}',
    history_out='${history_out}',
    batch_size=${batch_size},
    epochs=${epochs},
    head_warmup_epochs=${head_warmup_epochs},
    learning_rate=${learning_rate},
    weight_decay=${weight_decay},
    label_smoothing=0.10,
    dropout=${dropout},
    early_stop_patience=${patience},
    grad_clip_norm=1.0,
    random_seed=1337,
    num_workers=4,
    amp=True,
    freeze_backbone_during_warmup=True,
)
train_imagenette_model(cfg)
" |& tee "${log_file}"
}

run_train \
  "vgg16" \
  "weights/vgg16_one_dense_nopool_imagenette2_best_basictrain.pt" \
  "weights/vgg16_one_dense_nopool_imagenette2_last_basictrain.pt" \
  "analysis/vgg16_dense_head/train_history.csv" \
  "64" \
  "100" \
  "2" \
  "3e-4" \
  "1e-2" \
  "0.2" \
  "10" \
  "analysis/vgg16_dense_head/peek_vgg16_dense_head_imagenette.log"

run_train \
  "resnet50" \
  "weights/resnet50_one_dense_nopool_imagenette_best.pt" \
  "weights/resnet50_one_dense_nopool_imagenette_last.pt" \
  "analysis/resnet50_dense_head/train_history.csv" \
  "64" \
  "50" \
  "1" \
  "3e-4" \
  "1e-4" \
  "0.2" \
  "8" \
  "analysis/resnet50_dense_head/peek_resnet50_dense_head_imagenette.log"

run_train \
  "convnext_base" \
  "weights/convnext_base_one_dense_nopool_imagenette_best.pt" \
  "weights/convnext_base_one_dense_nopool_imagenette_last.pt" \
  "analysis/convnext_base_dense_head/train_history.csv" \
  "32" \
  "50" \
  "1" \
  "3e-4" \
  "5e-4" \
  "0.1" \
  "8" \
  "analysis/convnext_base_dense_head/peek_convnext_base_dense_head_imagenette.log"
