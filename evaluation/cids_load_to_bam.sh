# 使用 BaM 自带的 readwrite_stripe benchmark 直接把 prepared images.bin 写入 SSD。
# 这样更贴近 README 推荐路径，也避免 Python store_tensor 大块写入不稳定的问题。

TRAIN_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_f32
VAL_ROOT=/home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_f32
BENCH=/home/xhk/hyperion/GIDS/bam/build/bin/nvm-readwrite_stripe-bench

TRAIN_INPUT="${TRAIN_ROOT}/images.bin"
VAL_INPUT="${VAL_ROOT}/images.bin"

TRAIN_BYTES=$(/home/xhk/miniconda3/envs/pytorch/bin/python -c 'import json; from pathlib import Path; meta=json.load(open(Path("'"${TRAIN_ROOT}"'")/"meta.json","r",encoding="utf-8")); print(int(meta["num_samples"])*int(meta["sample_bytes"]))')

echo "[CIDS_LOAD] 使用 BaM benchmark 写入训练集: ${TRAIN_INPUT}"
sudo "${BENCH}" \
  --input "${TRAIN_INPUT}" \
  --queue_depth 1024 \
  --access_type 1 \
  --num_queues 128 \
  --threads 102400 \
  --n_ctrls 1 \
  --ioffset 0 \
  --loffset 0

echo "[CIDS_LOAD] 使用 BaM benchmark 写入验证集: ${VAL_INPUT}"
echo "[CIDS_LOAD] 验证集 loffset(bytes)=${TRAIN_BYTES}"
sudo "${BENCH}" \
  --input "${VAL_INPUT}" \
  --queue_depth 1024 \
  --access_type 1 \
  --num_queues 128 \
  --threads 102400 \
  --n_ctrls 1 \
  --ioffset 0 \
  --loffset "${TRAIN_BYTES}"
