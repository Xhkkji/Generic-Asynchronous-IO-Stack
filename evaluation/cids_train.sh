# 训练脚本默认假设 prepared images 已经先通过 cids_load_to_bam.sh 写入 BaM。
# --io-mode 可选：
#   sync        使用 CIDS + BaM 的同步读取
#   registered  使用 CIDS + BaM 的 registered 异步读取
#   torch       使用 PyTorch 原生 DataLoader 直接读取 prepared dataset 文件
# 训练配置参数更适合直接通过命令行传给 cids_train.py：
# - io-mode / epochs / batch-size / cache-size / prefetch-depth /
#   registered-split / enable-profile / profile-dir
# 只有偏底层、偏调试的系统开关继续保留为环境变量。
# 系统级运行开关说明：
# - CUDA_VISIBLE_DEVICES: 指定运行时可见 GPU
# - GIDS_FORCE_SYNC_READ: sync 模式下是否强制走 read_feature_kernel
# - GIDS_ASYNC_DEBUG_ROWS / GIDS_ASYNC_DEBUG_DIMS / GIDS_WARP_CTX_DEBUG_SAMPLE:
#   底层异步读取调试开关，默认关闭
# - CIDS_DEBUG: CIDS Python 侧调试日志开关
# - CIDS_REGISTERED_TRY_WINDOW_SIZE: registered 模式 skip-front 一次最多预热多少个后排 request
# - CIDS_REGISTERED_POLL_DEBUG: registered compatible poll 的主机侧日志开关
# - CIDS_PROFILE_GPU_TIMING: profiling 时是否对阶段末尾做 cuda synchronize，便于看到 submit/poll/get 的 GPU 时间
# --prefetch-depth 可控制 registered 路径的预取深度
# --registered-skip-front:
#   1 表示开启 skip-front 预热
#   0 表示关闭 skip-front，便于做最保守的 registered 排查
# --registered-split:
#   把一个训练 batch 拆成多个 sub-request，再在 Python wait/get 阶段合并

# 训练配置
IO_MODE="registered"
EPOCHS=5
BATCH_SIZE=1024
CACHE_SIZE=1024
PREFETCH_DEPTH=1
REGISTERED_SPLIT=1
ENABLE_PROFILE=1
PROFILE_DIR="./cids_profile"
REGISTERED_SKIP_FRONT=0

sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-0}" \
  GIDS_ASYNC_DEBUG_ROWS="${GIDS_ASYNC_DEBUG_ROWS:-0}" \
  GIDS_ASYNC_DEBUG_DIMS="${GIDS_ASYNC_DEBUG_DIMS:-16}" \
  GIDS_WARP_CTX_DEBUG_SAMPLE="${GIDS_WARP_CTX_DEBUG_SAMPLE:-0}" \
  CIDS_DEBUG="${CIDS_DEBUG:-0}" \
  CIDS_REGISTERED_TRY_WINDOW_SIZE="${CIDS_REGISTERED_TRY_WINDOW_SIZE:-2}" \
  CIDS_REGISTERED_POLL_DEBUG="${CIDS_REGISTERED_POLL_DEBUG:-0}" \
  CIDS_PROFILE_GPU_TIMING="${CIDS_PROFILE_GPU_TIMING:-0}" \
  /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/evaluation/cids_train.py \
  --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_f32 \
  --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_f32 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --ctrl-idx 0 \
  --io-mode "${IO_MODE}" \
  --cache-size "${CACHE_SIZE}" \
  --enable-profile "${ENABLE_PROFILE}" \
  --profile-dir "${PROFILE_DIR}" \
  --prefetch-depth "${PREFETCH_DEPTH}" \
  --registered-skip-front "${REGISTERED_SKIP_FRONT}" \
  --registered-split "${REGISTERED_SPLIT}"
