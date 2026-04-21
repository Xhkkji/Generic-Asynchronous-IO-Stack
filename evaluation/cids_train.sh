# 训练脚本默认假设 prepared images 已经先通过 cids_load_to_bam.sh 写入 BaM。
# --io-mode registered/sync
sudo /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/evaluation/cids_train.py \
  --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_f32 \
  --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_f32 \
  --epochs 1 \
  --batch-size 64 \
  --ctrl-idx 0 \
  --io-mode sync \
  --force-sync-read 1
