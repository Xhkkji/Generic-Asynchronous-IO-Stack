# 先把 prepared CIDS 数据写入 BaM 数组，再单独启动训练。
sudo /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/evaluation/cids_load_to_bam.py \
  --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_f32 \
  --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_f32 \
  --ctrl-idx 0
