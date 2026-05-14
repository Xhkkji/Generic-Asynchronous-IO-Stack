`cnn_evaluation` current layout:

- `dataset_pipeline/`
  shared image dataset preparation and BaM write helpers
- `networks/`
  network-specific CNN training code and launchers
- `common_tools/`
  shared CIDS helper tools
- `imagenet1k_subset_190_20_pad448/`
  fixed launchers for the 190/20 ImageNet-1K subset pipeline

Top level now keeps only shared CIDS dataset / transform helpers:

- `cids_prepare_dataset.py`
- `cids_prepare_dataset.sh`
- `cids_prepare_dataset_imagenet1k.py`
- `cids_prepare_dataset_imagenet1k.sh`
- `cids_load_imagenet1k_to_bam.sh`
- `cids_image_transforms.py`

Network-specific training files have been moved under `networks/`.
