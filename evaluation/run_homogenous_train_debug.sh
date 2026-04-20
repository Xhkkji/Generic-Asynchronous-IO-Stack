#!/usr/bin/env bash

set -euo pipefail

# 关键运行开关说明：
# - GIDS_USE_REGISTERED_TRY_SERVICE=1: 使用当前新的 registered try-service 路径
# - GIDS_USE_ASYNC_SAMPLE_IO_PIPELINE=0: 关闭旧的采样+IO 后台流水线，避免和新逻辑混用
# - GIDS_REGISTERED_DEBUG=1: 打开中文调试日志，便于观察 iter/batch 的提交与轮询
# - GIDS_REGISTERED_TRY_WINDOW_SIZE: skip-front 一次最多预热多少个后排 request
# - GIDS_REGISTERED_ENABLE_SKIP_FRONT=1: 是否启用 skip-front 预热；设为 0 时只强轮询当前 front iter
# - GIDS_REGISTERED_SUBMIT_COMMANDS_PER_BATCH: 一个逻辑 batch 拆成多少个提交命令；1 表示不拆分，等于 baseline
# - GIDS_MAX_REGISTERED_OUTSTANDING_IOS: registered 路径允许的最大 outstanding IO 预算
sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-1}" \
  GIDS_USE_REGISTERED_TRY_SERVICE="${GIDS_USE_REGISTERED_TRY_SERVICE:-1}" \
  GIDS_USE_ASYNC_SAMPLE_IO_PIPELINE="${GIDS_USE_ASYNC_SAMPLE_IO_PIPELINE:-0}" \
  GIDS_ASYNC_DEBUG_ROWS="${GIDS_ASYNC_DEBUG_ROWS:-0}" \
  GIDS_ASYNC_DEBUG_DIMS="${GIDS_ASYNC_DEBUG_DIMS:-16}" \
  GIDS_WARP_CTX_DEBUG_SAMPLE="${GIDS_WARP_CTX_DEBUG_SAMPLE:-0}" \
  GIDS_REGISTERED_DEBUG="${GIDS_REGISTERED_DEBUG:-1}" \
  GIDS_REGISTERED_TRY_WINDOW_SIZE="${GIDS_REGISTERED_TRY_WINDOW_SIZE:-2}" \
  GIDS_REGISTERED_ENABLE_SKIP_FRONT="${GIDS_REGISTERED_ENABLE_SKIP_FRONT:-0}" \
  GIDS_REGISTERED_SUBMIT_COMMANDS_PER_BATCH="${GIDS_REGISTERED_SUBMIT_COMMANDS_PER_BATCH:-1}" \
  GIDS_MAX_REGISTERED_OUTSTANDING_IOS="${GIDS_MAX_REGISTERED_OUTSTANDING_IOS:-260000}" \
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
  --batch_size 1024 \
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
