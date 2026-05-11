#!/usr/bin/env bash

set -euo pipefail

# ResNet-50 + ImageNet-1K 训练脚本：
# - 默认使用已经写入 BaM 的 ImageNet-1K uint8 prepared dataset
# - --io-mode 可选：
#   sync        使用 CIDS + BaM 的同步读取
#   registered  使用 CIDS + BaM 的 registered 异步读取
#   torch       使用 PyTorch 原生 DataLoader 直接读取 prepared dataset 文件
# - --torch-read-mode:
#   mmap        使用 np.memmap 直接映射 prepared 文件
#   buffered    启动时整块读入内存，作为 torch 的非 mmap 对照

IO_MODE="${IO_MODE:-torch}"
TORCH_READ_MODE="${TORCH_READ_MODE:-buffered}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-300}"
RUN_VAL="${RUN_VAL:-0}"
CACHE_SIZE="${CACHE_SIZE:-1024}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-1}"
REGISTERED_SPLIT="${REGISTERED_SPLIT:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
PROFILE_DIR="${PROFILE_DIR:-./cids_profile_resnet50}"
REGISTERED_SKIP_FRONT="${REGISTERED_SKIP_FRONT:-0}"

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
  /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/cnn_evaluation/cids_train_resnet50.py \
  --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8 \
  --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8 \
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
