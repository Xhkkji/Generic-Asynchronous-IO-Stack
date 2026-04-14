#!/usr/bin/env bash

set -euo pipefail

sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-0}" \
  GIDS_USE_REGISTERED_TRY_SERVICE="${GIDS_USE_REGISTERED_TRY_SERVICE:-1}" \
  GIDS_USE_REGISTERED_POLL="${GIDS_USE_REGISTERED_POLL:-0}" \
  GIDS_USE_ASYNC_SAMPLE_IO_PIPELINE="${GIDS_USE_ASYNC_SAMPLE_IO_PIPELINE:-0}" \
  GIDS_ASYNC_DEBUG_ROWS="${GIDS_ASYNC_DEBUG_ROWS:-0}" \
  GIDS_ASYNC_DEBUG_DIMS="${GIDS_ASYNC_DEBUG_DIMS:-16}" \
  GIDS_WARP_CTX_DEBUG_SAMPLE="${GIDS_WARP_CTX_DEBUG_SAMPLE:-0}" \
  GIDS_REGISTERED_DEBUG="${GIDS_REGISTERED_DEBUG:-0}" \
  GIDS_REGISTERED_TRY_WINDOW_SIZE="${GIDS_REGISTERED_TRY_WINDOW_SIZE:-2}" \
  GIDS_MAX_REGISTERED_OUTSTANDING_IOS="${GIDS_MAX_REGISTERED_OUTSTANDING_IOS:-200000}" \
  /home/xhk/miniconda3/envs/pytorch/bin/python homogenous_train_debug.py \
  --stop_after_step 150 \
  --hidden_channels 256 \
  --path /data/igb/ \
  --dataset_size medium \
  --epochs 1 \
  --num_heads 8 \
  --log_every 1000 \
  --uva_graph 1 \
  --GIDS \
  --batch_size 512 \
  --num_classes 19 \
  --data IGB \
  --emb_size 1024 \
  --model_type sage \
  --num_layers 3 \
  --fan_out '10,5,5' \
  --modelpath /home/xhk/hyperion/GIDS/dataset/igb/pr_medium_full.pt \
  --pin_file /home/xhk/hyperion/GIDS/dataset/igb/pr_medium.pt \
  --cache_size $((1024)) \
  --num_ssd 1 \
  --num_ele $((550*1000*1000*1024)) \
  --page_size 4096 \
  "$@"
