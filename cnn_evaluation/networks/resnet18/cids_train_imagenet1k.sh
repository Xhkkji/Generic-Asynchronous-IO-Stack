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
IO_MODE="${IO_MODE:-registered}"
TORCH_READ_MODE="${TORCH_READ_MODE:-mmap}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-0}"
RUN_VAL="${RUN_VAL:-0}"
CACHE_SIZE="${CACHE_SIZE:-4096}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-1}"
REGISTERED_SPLIT="${REGISTERED_SPLIT:-1}"
ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
REGISTERED_SKIP_FRONT="${REGISTERED_SKIP_FRONT:-0}"
COLD_START="${COLD_START:-1}"
AUTO_LOG="${AUTO_LOG:-1}"
# registered 调试/诊断开关
CIDS_REGISTERED_TRACE_CALLS="${CIDS_REGISTERED_TRACE_CALLS:-0}"
CIDS_REGISTERED_POLL_TIMEOUT_SEC="${CIDS_REGISTERED_POLL_TIMEOUT_SEC:-60}"
CIDS_REGISTERED_POLL_LOG_INTERVAL="${CIDS_REGISTERED_POLL_LOG_INTERVAL:-128}"
# shader cache相关
SHUFFLE="${SHUFFLE:-1}"
ENABLE_SAMPLE_CACHE="${ENABLE_SAMPLE_CACHE:-0}"
SAMPLE_CACHE_CAPACITY="${SAMPLE_CACHE_CAPACITY:-24000}"
SAMPLE_CACHE_PIN_MEMORY="${SAMPLE_CACHE_PIN_MEMORY:-0}"
ENABLE_SAMPLE_IMPORTANCE="${ENABLE_SAMPLE_IMPORTANCE:-1}"
ENABLE_BAM_POLICY_CACHE="${ENABLE_BAM_POLICY_CACHE:-1}"
IMPORTANCE_EMA_ALPHA="${IMPORTANCE_EMA_ALPHA:-0.9}"
IMPORTANCE_TOPK="${IMPORTANCE_TOPK:-5}"
# PADS resident 调度策略：
# - replace: 轻量 resident 替换
# - shade: 更接近 SHADE 的 locality replay
# - hotset: 逻辑 hotset 主导的“高质量样本 + 随机覆盖”混合提交
PADS_STRATEGY="${PADS_STRATEGY:-hotset}"  
if [[ -z "${PADS_BIAS_SCALE+x}" ]]; then
  if [[ "${PADS_STRATEGY}" == "hotset" ]]; then
    PADS_BIAS_SCALE="0.0625"
  elif [[ "${PADS_STRATEGY}" == "shade" ]]; then
    PADS_BIAS_SCALE="0.35"
  else
    PADS_BIAS_SCALE="0.10"
  fi
fi
if [[ -z "${PADS_MAX_REPLACE_FRACTION+x}" ]]; then
  if [[ "${PADS_STRATEGY}" == "hotset" ]]; then
    PADS_MAX_REPLACE_FRACTION="0.008"
  elif [[ "${PADS_STRATEGY}" == "shade" ]]; then
    PADS_MAX_REPLACE_FRACTION="0.008"
  else
    PADS_MAX_REPLACE_FRACTION="0.003"
  fi
fi

if [[ "${IO_MODE}" == "torch" ]]; then
  PROFILE_MODE="torch_${TORCH_READ_MODE}"
else
  PROFILE_MODE="${IO_MODE}"
fi
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/profiles/${PROFILE_MODE}}"
mkdir -p "${PROFILE_DIR}"

if [[ "${AUTO_LOG}" == "1" ]]; then
  if [[ "${IO_MODE}" == "torch" ]]; then
    LOG_PATH="${SCRIPT_DIR}/output_torch_${TORCH_READ_MODE}.log"
  else
    if [[ "${ENABLE_BAM_POLICY_CACHE}" == "1" ]]; then
      LOG_SUFFIX="bam_policy_${PADS_STRATEGY}"
    else
      LOG_SUFFIX="bam_nopolicy"
    fi
    LOG_PATH="${SCRIPT_DIR}/output_${IO_MODE}_${LOG_SUFFIX}_cache4096_1epoch_hotest.log"
  fi
  exec > >(tee "${LOG_PATH}") 2>&1
  echo "[CIDS_RESNET18] auto log -> ${LOG_PATH}"
fi

if [[ "${COLD_START}" == "1" ]]; then
  echo "[CIDS_RESNET18] cold start: sync + drop_caches"
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
fi

sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GIDS_ASYNC_DEBUG_ROWS="${GIDS_ASYNC_DEBUG_ROWS:-0}" \
  GIDS_ASYNC_DEBUG_DIMS="${GIDS_ASYNC_DEBUG_DIMS:-16}" \
  GIDS_WARP_CTX_DEBUG_SAMPLE="${GIDS_WARP_CTX_DEBUG_SAMPLE:-0}" \
  CIDS_DEBUG="${CIDS_DEBUG:-0}" \
  CIDS_REGISTERED_TRY_WINDOW_SIZE="${CIDS_REGISTERED_TRY_WINDOW_SIZE:-2}" \
  CIDS_REGISTERED_POLL_DEBUG="${CIDS_REGISTERED_POLL_DEBUG:-0}" \
  CIDS_REGISTERED_TRACE_CALLS="${CIDS_REGISTERED_TRACE_CALLS}" \
  CIDS_REGISTERED_POLL_TIMEOUT_SEC="${CIDS_REGISTERED_POLL_TIMEOUT_SEC}" \
  CIDS_REGISTERED_POLL_LOG_INTERVAL="${CIDS_REGISTERED_POLL_LOG_INTERVAL}" \
  CIDS_PROFILE_GPU_TIMING="${CIDS_PROFILE_GPU_TIMING:-0}" \
  /home/xhk/miniconda3/envs/pytorch/bin/python "${SCRIPT_DIR}/cids_train.py" \
  --train-root "${REPO_ROOT}/dataset/imagenet/imagenet1k_subset_190_20_pad416/cids_train_u8_pad416_bam" \
  --val-root "${REPO_ROOT}/dataset/imagenet/imagenet1k_subset_190_20_pad416/cids_val_u8_pad416_bam" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --shuffle "${SHUFFLE}" \
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
  --registered-split "${REGISTERED_SPLIT}" \
  --enable-sample-cache "${ENABLE_SAMPLE_CACHE}" \
  --sample-cache-capacity "${SAMPLE_CACHE_CAPACITY}" \
  --sample-cache-pin-memory "${SAMPLE_CACHE_PIN_MEMORY}" \
  --enable-sample-importance "${ENABLE_SAMPLE_IMPORTANCE}" \
  --enable-bam-policy-cache "${ENABLE_BAM_POLICY_CACHE}" \
  --importance-ema-alpha "${IMPORTANCE_EMA_ALPHA}" \
  --importance-topk "${IMPORTANCE_TOPK}" \
  --pads-strategy "${PADS_STRATEGY}" \
  --pads-bias-scale "${PADS_BIAS_SCALE}" \
  --pads-max-replace-fraction "${PADS_MAX_REPLACE_FRACTION}"
