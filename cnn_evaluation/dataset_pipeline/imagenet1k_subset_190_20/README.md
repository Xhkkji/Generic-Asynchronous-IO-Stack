Balanced ImageNet-1K subset pipeline.

Goal:

- train: 190 images per class
- val: 20 images per class
- keep 1000 classes
- organize subset as an ImageFolder-compatible tree using symlinks
- then run `pad448` prepared-dataset generation on top of the subset

Notes:

- the current `imagenet1k_rgb/val` links are not usable as a raw-image source in this workspace
- this pipeline therefore samples `210` images per class from `train`, then splits them into
  `190` train images and `20` disjoint val images per class

Outputs:

- pipeline root:
  `/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448`
- subset root:
  `/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/imagenet1k_rgb_subset_190_20`
- prepared roots:
  - `/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_train_u8_pad448`
  - `/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/cids_val_u8_pad448`

Suggested order:

1. `bash build_subset.sh`
2. `bash prepare_pad448.sh`
3. `bash load_pad448_to_bam.sh`
