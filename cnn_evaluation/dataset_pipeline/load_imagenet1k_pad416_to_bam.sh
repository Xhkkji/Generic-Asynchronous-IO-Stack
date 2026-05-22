#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad416

TRAIN_ROOT="${PIPELINE_ROOT}/cids_train_u8_pad416_bam" \
VAL_ROOT="${PIPELINE_ROOT}/cids_val_u8_pad416_bam" \
bash "${SCRIPT_DIR}/cids_load_imagenet1k_to_bam.sh"
