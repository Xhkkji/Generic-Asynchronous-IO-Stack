#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/cids_prepare_dataset_imagenet1k.sh" \
  --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_train_u8_pad448 \
  --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_val_u8_pad448 \
  --train-bam-root /home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_train_u8_pad448_bam \
  --val-bam-root /home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_val_u8_pad448_bam \
  --page-size 1048576
