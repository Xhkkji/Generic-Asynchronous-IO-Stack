#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8_pad448_bam \
VAL_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8_pad448_bam \
bash "${SCRIPT_DIR}/cids_load_imagenet1k_to_bam.sh"
