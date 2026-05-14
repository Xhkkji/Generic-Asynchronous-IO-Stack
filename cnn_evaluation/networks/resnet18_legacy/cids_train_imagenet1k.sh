#!/usr/bin/env bash

set -euo pipefail

# ImageNet-1K 训练脚本：
# - 默认使用已经写入 BaM 的 ImageNet-1K uint8 prepared dataset
# - --io-mode 可选：
#   sync        使用 CIDS + BaM 的同步读取
#   registered  使用 CIDS + BaM 的 registered 异步读取
#   torch       使用 PyTorch 原生 DataLoader 直接读取 prepared dataset 文件
# - --torch-read-mode:
#   mmap        使用 np.memmap 直接映射 prepared 文件
#   buffered    启动时整块读入内存，作为 torch 的非 mmap 对照

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=/home/xhk/hyperion/GIDS

# 训练配置
IO_MODE="${IO_MODE:-torch}"
TORCH_READ_MODE="${TORCH_READ_MODE:-mmap}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-5000}"
RUN_VAL="${RUN_VAL:-0}"
CACHE_SIZE="${CACHE_SIZE:-1024}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-1}"
REGISTERED_SPLIT="${REGISTERED_SPLIT:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
REGISTERED_SKIP_FRONT="${REGISTERED_SKIP_FRONT:-0}"
COLD_START="${COLD_START:-1}"
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
  GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-1}" \
  GIDS_ASYNC_DEBUG_ROWS="${GIDS_ASYNC_DEBUG_ROWS:-0}" \
  GIDS_ASYNC_DEBUG_DIMS="${GIDS_ASYNC_DEBUG_DIMS:-16}" \
  GIDS_WARP_CTX_DEBUG_SAMPLE="${GIDS_WARP_CTX_DEBUG_SAMPLE:-0}" \
  CIDS_DEBUG="${CIDS_DEBUG:-0}" \
  CIDS_REGISTERED_TRY_WINDOW_SIZE="${CIDS_REGISTERED_TRY_WINDOW_SIZE:-2}" \
  CIDS_REGISTERED_POLL_DEBUG="${CIDS_REGISTERED_POLL_DEBUG:-0}" \
  CIDS_PROFILE_GPU_TIMING="${CIDS_PROFILE_GPU_TIMING:-0}" \
  /home/xhk/miniconda3/envs/pytorch/bin/python "${REPO_ROOT}/evaluation/cids_train.py" \
  --train-root "${REPO_ROOT}/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_train_u8_pad448" \
  --val-root "${REPO_ROOT}/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_val_u8_pad448" \
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
