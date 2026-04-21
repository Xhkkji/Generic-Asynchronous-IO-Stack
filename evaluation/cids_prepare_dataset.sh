# 训练集
python /home/xhk/hyperion/GIDS/evaluation/cids_prepare_dataset.py \
  --dataset tiny-imagenet \
  --input-root /home/xhk/hyperion/GIDS/dataset/imagenet/tiny-imagenet-200 \
  --output-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train \
  --split train \
  --dtype float32

# 验证集
python /home/xhk/hyperion/GIDS/evaluation/cids_prepare_dataset.py \
  --dataset tiny-imagenet \
  --input-root /home/xhk/hyperion/GIDS/dataset/imagenet/tiny-imagenet-200 \
  --output-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val \
  --split val \
  --dtype float32