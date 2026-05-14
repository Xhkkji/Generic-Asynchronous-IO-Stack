#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=/home/xhk/hyperion/GIDS
PROFILE_ROOT="${SCRIPT_DIR}/profiles"

# AlexNet + ImageNet-1K 训练脚本：
# - 默认使用已经写入 BaM 的 ImageNet-1K prepared dataset
# - 为了更容易放大 CIDS 异步 IO 的优势，默认把 cache 调小、预取和 registered split 调高
# - 如果你已经准备了 float32 版本的数据，只需要覆盖 TRAIN_ROOT / VAL_ROOT 即可
# - --io-mode 可选：
#   sync        使用 CIDS + BaM 的同步读取
#   registered  使用 CIDS + BaM 的 registered 异步读取
#   torch       使用 PyTorch 原生 DataLoader 直接读取 prepared dataset 文件
# - --torch-read-mode:
#   mmap        使用 np.memmap 直接映射 prepared 文件
#   buffered    启动时整块读入内存，作为 torch 的非 mmap 对照

IO_MODE="${IO_MODE:-registered}"
TORCH_READ_MODE="${TORCH_READ_MODE:-buffered}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-6000}"
RUN_VAL="${RUN_VAL:-0}"
CACHE_SIZE="${CACHE_SIZE:-256}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-1}"
REGISTERED_SPLIT="${REGISTERED_SPLIT:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
REGISTERED_SKIP_FRONT="${REGISTERED_SKIP_FRONT:-0}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8}"
VAL_ROOT="${VAL_ROOT:-/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8}"

if [[ "${IO_MODE}" == "torch" ]]; then
  PROFILE_MODE="torch_${TORCH_READ_MODE}"
else
  PROFILE_MODE="${IO_MODE}"
fi
PROFILE_DIR="${PROFILE_DIR:-${PROFILE_ROOT}/${PROFILE_MODE}}"

if [[ -n "${GIDS_FORCE_SYNC_READ:-}" ]]; then
  FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ}"
elif [[ "${IO_MODE}" == "sync" ]]; then
  FORCE_SYNC_READ="1"
else
  FORCE_SYNC_READ="0"
fi

sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GIDS_FORCE_SYNC_READ="${FORCE_SYNC_READ}" \
  GIDS_ASYNC_DEBUG_ROWS="${GIDS_ASYNC_DEBUG_ROWS:-0}" \
  GIDS_ASYNC_DEBUG_DIMS="${GIDS_ASYNC_DEBUG_DIMS:-16}" \
  GIDS_WARP_CTX_DEBUG_SAMPLE="${GIDS_WARP_CTX_DEBUG_SAMPLE:-0}" \
  CIDS_DEBUG="${CIDS_DEBUG:-0}" \
  CIDS_REGISTERED_TRY_WINDOW_SIZE="${CIDS_REGISTERED_TRY_WINDOW_SIZE:-2}" \
  CIDS_REGISTERED_POLL_DEBUG="${CIDS_REGISTERED_POLL_DEBUG:-0}" \
  CIDS_PROFILE_GPU_TIMING="${CIDS_PROFILE_GPU_TIMING:-0}" \
  /home/xhk/miniconda3/envs/pytorch/bin/python "${SCRIPT_DIR}/cids_train_alexnet.py" \
  --train-root "${TRAIN_ROOT}" \
  --val-root "${VAL_ROOT}" \
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
