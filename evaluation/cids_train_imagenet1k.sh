# ImageNet-1K 训练脚本：
# - 默认使用已经写入 BaM 的 ImageNet-1K uint8 prepared dataset
# - --io-mode 可选：
#   sync        使用 CIDS + BaM 的同步读取
#   registered  使用 CIDS + BaM 的 registered 异步读取
#   torch       使用 PyTorch 原生 DataLoader 直接读取 prepared dataset 文件
# - --torch-read-mode:
#   mmap        使用 np.memmap 直接映射 prepared 文件
#   buffered    启动时整块读入内存，作为 torch 的非 mmap 对照
#
# 训练配置参数更适合直接通过命令行传给 cids_train.py：
# - io-mode / torch-read-mode / epochs / batch-size / cache-size / prefetch-depth /
#   registered-split / enable-profile / profile-dir
#
# 只有偏底层、偏调试的系统开关继续保留为环境变量：
# - CUDA_VISIBLE_DEVICES
# - GIDS_FORCE_SYNC_READ
# - GIDS_ASYNC_DEBUG_ROWS / GIDS_ASYNC_DEBUG_DIMS / GIDS_WARP_CTX_DEBUG_SAMPLE
# - CIDS_DEBUG
# - CIDS_REGISTERED_TRY_WINDOW_SIZE
# - CIDS_REGISTERED_POLL_DEBUG
# - CIDS_PROFILE_GPU_TIMING

# 训练配置
IO_MODE="torch"
TORCH_READ_MODE="buffered"
EPOCHS=1
BATCH_SIZE=256
MAX_TRAIN_ITERS=300
RUN_VAL=0
CACHE_SIZE=1024
PREFETCH_DEPTH=1
REGISTERED_SPLIT=1
ENABLE_PROFILE=1
PROFILE_DIR="./cids_profile"
REGISTERED_SKIP_FRONT=0

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
  /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/evaluation/cids_train.py \
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
