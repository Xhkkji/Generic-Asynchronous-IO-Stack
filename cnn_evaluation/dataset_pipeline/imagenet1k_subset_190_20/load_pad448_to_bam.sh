#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448

TRAIN_ROOT="${PIPELINE_ROOT}/cids_train_u8_pad448" \
VAL_ROOT="${PIPELINE_ROOT}/cids_val_u8_pad448" \
bash "${PARENT_DIR}/cids_load_imagenet1k_to_bam.sh"
