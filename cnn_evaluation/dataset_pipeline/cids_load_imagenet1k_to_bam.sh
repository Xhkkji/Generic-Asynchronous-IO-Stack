#!/usr/bin/env bash

set -euo pipefail

BENCH=/home/xhk/hyperion/GIDS/bam/build/bin/nvm-readwrite_stripe-bench
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch/bin/python

TRAIN_ROOT="${TRAIN_ROOT:-/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8_bam}"
VAL_ROOT="${VAL_ROOT:-/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8_bam}"

TRAIN_INPUT="${TRAIN_ROOT}/images.bin"
VAL_INPUT="${VAL_ROOT}/images.bin"

if [[ ! -f "${TRAIN_INPUT}" ]]; then
  echo "[CIDS_LOAD_IMAGENET1K] missing train _bam file: ${TRAIN_INPUT}" >&2
  echo "[CIDS_LOAD_IMAGENET1K] run prepare_imagenet1k_pad448_bam.sh first" >&2
  exit 1
fi

if [[ ! -f "${VAL_INPUT}" ]]; then
  echo "[CIDS_LOAD_IMAGENET1K] missing val _bam file: ${VAL_INPUT}" >&2
  echo "[CIDS_LOAD_IMAGENET1K] run prepare_imagenet1k_pad448_bam.sh first" >&2
  exit 1
fi

TRAIN_BYTES=$(
  "${PYTHON_BIN}" -c 'import json; from pathlib import Path; meta=json.load(open(Path("'"${TRAIN_ROOT}"'")/"meta.json","r",encoding="utf-8")); print(int(meta["num_samples"])*int(meta["sample_bytes"]))'
)

echo "[CIDS_LOAD_IMAGENET1K] train input=${TRAIN_INPUT}"
echo "[CIDS_LOAD_IMAGENET1K] train bytes=${TRAIN_BYTES}"
sudo "${BENCH}" \
  --input "${TRAIN_INPUT}" \
  --queue_depth 1024 \
  --access_type 1 \
  --num_queues 128 \
  --threads 102400 \
  --n_ctrls 1 \
  --ioffset 0 \
  --loffset 0

echo "[CIDS_LOAD_IMAGENET1K] val input=${VAL_INPUT}"
echo "[CIDS_LOAD_IMAGENET1K] val loffset(bytes)=${TRAIN_BYTES}"
sudo "${BENCH}" \
  --input "${VAL_INPUT}" \
  --queue_depth 1024 \
  --access_type 1 \
  --num_queues 128 \
  --threads 102400 \
  --n_ctrls 1 \
  --ioffset 0 \
  --loffset "${TRAIN_BYTES}"
