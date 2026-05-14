#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/home/xhk/hyperion/GIDS
SUBSET_ROOT="${REPO_ROOT}/dataset/imagenet/imagenet1k_subset_190_20_pad448"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_ROOT="${SUBSET_ROOT}/cids_train_u8_pad448" \
VAL_ROOT="${SUBSET_ROOT}/cids_val_u8_pad448" \
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/profiles_resnet50}" \
bash "${REPO_ROOT}/cnn_evaluation/cids_train_imagenet1k_resnet50.sh"
