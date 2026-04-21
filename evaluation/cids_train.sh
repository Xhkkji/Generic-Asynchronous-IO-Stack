# 训练脚本默认假设 prepared images 已经先通过 cids_load_to_bam.sh 写入 BaM。
# --io-mode 可选：
#   sync        使用 CIDS + BaM 的同步读取
#   registered  使用 CIDS + BaM 的 registered 异步读取
#   torch       使用 PyTorch 原生 DataLoader 直接读取 prepared dataset 文件
sudo /home/xhk/miniconda3/envs/pytorch/bin/python /home/xhk/hyperion/GIDS/evaluation/cids_train.py \
  --train-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_f32 \
  --val-root /home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_f32 \
  --epochs 30 \
  --batch-size 64 \
  --ctrl-idx 0 \
  --io-mode sync \
  --force-sync-read 1
