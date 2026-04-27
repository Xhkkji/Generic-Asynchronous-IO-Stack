# 训练集
python /home/xhk/hyperion/GIDS/evaluation/cids_prepare_dataset.py \
  --dataset tiny-imagenet \
  --input-root /home/xhk/hyperion/GIDS/dataset/imagenet/tiny-imagenet-200 \
  --output-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_uint8 \
  --split train \
  --dtype uint8

# 验证集
python /home/xhk/hyperion/GIDS/evaluation/cids_prepare_dataset.py \
  --dataset tiny-imagenet \
  --input-root /home/xhk/hyperion/GIDS/dataset/imagenet/tiny-imagenet-200 \
  --output-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_uint8 \
  --split val \
  --dtype uint8

/home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/evaluation/cids_prepare_dataset.py \
  --dataset imagenet1k \
  --input-root /home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_rgb \
  --output-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8 \
  --split train \
  --dtype uint8

/home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/evaluation/cids_prepare_dataset.py \
  --dataset imagenet1k \
  --input-root /home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_rgb \
  --output-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8 \
  --split val \
  --dtype uint8
