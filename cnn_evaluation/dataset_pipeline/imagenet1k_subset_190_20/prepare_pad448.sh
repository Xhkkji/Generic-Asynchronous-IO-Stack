#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448
SUBSET_ROOT="${PIPELINE_ROOT}/imagenet1k_rgb_subset_190_20"

bash "${PARENT_DIR}/cids_prepare_dataset.sh" \
  --dataset imagenet1k \
  --input-root "${SUBSET_ROOT}" \
  --output-root "${PIPELINE_ROOT}/cids_train_u8_pad448" \
  --split train \
  --dtype uint8 \
  --preprocess-mode pad448

bash "${PARENT_DIR}/cids_prepare_dataset.sh" \
  --dataset imagenet1k \
  --input-root "${SUBSET_ROOT}" \
  --output-root "${PIPELINE_ROOT}/cids_val_u8_pad448" \
  --split val \
  --dtype uint8 \
  --preprocess-mode pad448
