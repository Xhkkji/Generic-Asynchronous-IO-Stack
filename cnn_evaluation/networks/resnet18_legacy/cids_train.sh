#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=/home/xhk/hyperion/GIDS

# 训练配置
IO_MODE="${IO_MODE:-registered}"
TORCH_READ_MODE="${TORCH_READ_MODE:-mmap}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-0}"
RUN_VAL="${RUN_VAL:-1}"
CACHE_SIZE="${CACHE_SIZE:-1024}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-1}"
REGISTERED_SPLIT="${REGISTERED_SPLIT:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
REGISTERED_SKIP_FRONT="${REGISTERED_SKIP_FRONT:-0}"
COLD_START="${COLD_START:-0}"
AUTO_LOG="${AUTO_LOG:-1}"

if [[ "${IO_MODE}" == "torch" ]]; then
  PROFILE_MODE="torch_${TORCH_READ_MODE}"
else
  PROFILE_MODE="${IO_MODE}"
fi
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/profiles/${PROFILE_MODE}}"

if [[ "${AUTO_LOG}" == "1" ]]; then
  if [[ "${IO_MODE}" == "torch" ]]; then
    LOG_PATH="${SCRIPT_DIR}/output_torch_${TORCH_READ_MODE}.log"
  else
    LOG_PATH="${SCRIPT_DIR}/output_${IO_MODE}.log"
  fi
  exec > >(tee "${LOG_PATH}") 2>&1
  echo "[CIDS_RESNET18_LEGACY] auto log -> ${LOG_PATH}"
fi

if [[ "${COLD_START}" == "1" ]]; then
  echo "[CIDS_RESNET18_LEGACY] cold start: sync + drop_caches"
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
fi

sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-0}" \
  GIDS_ASYNC_DEBUG_ROWS="${GIDS_ASYNC_DEBUG_ROWS:-0}" \
  GIDS_ASYNC_DEBUG_DIMS="${GIDS_ASYNC_DEBUG_DIMS:-16}" \
  GIDS_WARP_CTX_DEBUG_SAMPLE="${GIDS_WARP_CTX_DEBUG_SAMPLE:-0}" \
  CIDS_DEBUG="${CIDS_DEBUG:-0}" \
  CIDS_REGISTERED_TRY_WINDOW_SIZE="${CIDS_REGISTERED_TRY_WINDOW_SIZE:-2}" \
  CIDS_REGISTERED_POLL_DEBUG="${CIDS_REGISTERED_POLL_DEBUG:-0}" \
  CIDS_PROFILE_GPU_TIMING="${CIDS_PROFILE_GPU_TIMING:-0}" \
  /home/xhk/miniconda3/envs/pytorch/bin/python "${REPO_ROOT}/evaluation/cids_train.py" \
  --train-root "${REPO_ROOT}/dataset/imagenet/cids_tiny_imagenet_train_u8" \
  --val-root "${REPO_ROOT}/dataset/imagenet/cids_tiny_imagenet_val_u8" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-iters "${MAX_TRAIN_ITERS}" \
  --run-val "${RUN_VAL}" \
  --ctrl-idx 0 \
  --io-mode "${IO_MODE}" \
  --torch-read-mode "${TORCH_READ_MODE}" \
  --cache-size "${CACHE_SIZE}" \
  --enable-profile "${ENABLE_PROFILE}" \
  --profile-dir "${PROFILE_DIR}" \
  --prefetch-depth "${PREFETCH_DEPTH}" \
  --registered-skip-front "${REGISTERED_SKIP_FRONT}" \
  --registered-split "${REGISTERED_SPLIT}"
