#!/usr/bin/env bash

set -euo pipefail

sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-0}" \
  /home/wzq/miniconda3/envs/gids/bin/python3 homogenous_train.py \
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
