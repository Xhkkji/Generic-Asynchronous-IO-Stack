#!/usr/bin/env bash

set -euo pipefail

# 使用 BaM 自带的 readwrite_stripe benchmark 直接把已经准备好的
# ImageNet-1K _bam 对齐版 images.bin 写入 SSD。

BENCH=/home/xhk/hyperion/GIDS/bam/build/bin/nvm-readwrite_stripe-bench
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch/bin/python

TRAIN_ROOT="${TRAIN_ROOT:-/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8_bam}"
VAL_ROOT="${VAL_ROOT:-/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8_bam}"

TRAIN_INPUT="${TRAIN_ROOT}/images.bin"
VAL_INPUT="${VAL_ROOT}/images.bin"

if [[ ! -f "${TRAIN_INPUT}" ]]; then
  echo "[CIDS_LOAD_IMAGENET1K] 未找到 train _bam 文件: ${TRAIN_INPUT}" >&2
  echo "[CIDS_LOAD_IMAGENET1K] 请先运行: bash /home/xhk/hyperion/GIDS/cnn_evaluation/cids_prepare_dataset_imagenet1k.sh" >&2
  exit 1
fi

if [[ ! -f "${VAL_INPUT}" ]]; then
  echo "[CIDS_LOAD_IMAGENET1K] 未找到 val _bam 文件: ${VAL_INPUT}" >&2
  echo "[CIDS_LOAD_IMAGENET1K] 请先运行: bash /home/xhk/hyperion/GIDS/cnn_evaluation/cids_prepare_dataset_imagenet1k.sh" >&2
  exit 1
fi

TRAIN_BYTES=$(
  "${PYTHON_BIN}" -c 'import json; from pathlib import Path; meta=json.load(open(Path("'"${TRAIN_ROOT}"'")/"meta.json","r",encoding="utf-8")); print(int(meta["num_samples"])*int(meta["sample_bytes"]))'
)

echo "[CIDS_LOAD_IMAGENET1K] 使用 BaM benchmark 写入训练集: ${TRAIN_INPUT}"
echo "[CIDS_LOAD_IMAGENET1K] 训练集总字节数=${TRAIN_BYTES}"
sudo "${BENCH}" \
  --input "${TRAIN_INPUT}" \
  --queue_depth 1024 \
  --access_type 1 \
  --num_queues 128 \
  --threads 102400 \
  --n_ctrls 1 \
  --ioffset 0 \
  --loffset 0

echo "[CIDS_LOAD_IMAGENET1K] 使用 BaM benchmark 写入验证集: ${VAL_INPUT}"
echo "[CIDS_LOAD_IMAGENET1K] 验证集 loffset(bytes)=${TRAIN_BYTES}"
sudo "${BENCH}" \
  --input "${VAL_INPUT}" \
  --queue_depth 1024 \
  --access_type 1 \
  --num_queues 128 \
  --threads 102400 \
  --n_ctrls 1 \
  --ioffset 0 \
  --loffset "${TRAIN_BYTES}"
