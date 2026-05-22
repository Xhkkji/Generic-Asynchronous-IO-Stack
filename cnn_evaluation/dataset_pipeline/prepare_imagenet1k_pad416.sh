#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RGB_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/imagenet1k_rgb_subset_190_20
PIPELINE_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad416

bash "${SCRIPT_DIR}/cids_prepare_dataset.sh" \
  --dataset imagenet1k \
  --input-root "${RGB_ROOT}" \
  --output-root "${PIPELINE_ROOT}/cids_train_u8_pad416" \
  --split train \
  --dtype uint8 \
  --preprocess-mode pad416

bash "${SCRIPT_DIR}/cids_prepare_dataset.sh" \
  --dataset imagenet1k \
  --input-root "${RGB_ROOT}" \
  --output-root "${PIPELINE_ROOT}/cids_val_u8_pad416" \
  --split val \
  --dtype uint8 \
  --preprocess-mode pad416
