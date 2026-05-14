# sudo /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/cnn_evaluation/common_tools/cids_compare_readback.py \
#   --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_u8 \
#   --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_u8 \
#   --split train \
#   --batch-size 4 \
#   --start-index 1234 \
#   --ctrl-idx 0


# sudo /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/cnn_evaluation/common_tools/cids_compare_readback.py \
#   --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_f32 \
#   --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_f32 \
#   --split val \
#   --batch-size 4 \
#   --start-index 0 \
#   --ctrl-idx 0

sudo /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/cnn_evaluation/common_tools/cids_compare_readback.py \
  --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8 \
  --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8 \
  --split train \
  --batch-size 4 \
  --start-index 1235 \
  --ctrl-idx 0
