#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RGB_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_rgb
TRAIN_OUT=/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8_pad448
VAL_OUT=/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8_pad448

bash "${SCRIPT_DIR}/cids_prepare_dataset.sh" \
  --dataset imagenet1k \
  --input-root "${RGB_ROOT}" \
  --output-root "${TRAIN_OUT}" \
  --split train \
  --dtype uint8 \
  --preprocess-mode pad448

bash "${SCRIPT_DIR}/cids_prepare_dataset.sh" \
  --dataset imagenet1k \
  --input-root "${RGB_ROOT}" \
  --output-root "${VAL_OUT}" \
  --split val \
  --dtype uint8 \
  --preprocess-mode pad448
