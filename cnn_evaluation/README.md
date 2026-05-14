`cnn_evaluation` organization:

- `dataset_pipeline/`
  image dataset preparation and BaM write helpers
- `networks/`
  network-specific CIDS training/model launchers
- `common_tools/`
  shared CIDS helper tools migrated from `evaluation/`
- `imagenet1k_subset_190_20_pad448/`
  fixed launchers for the 190/20 ImageNet-1K subset pad448 pipeline

Top-level `cids_*.py` and generic launchers are kept for compatibility.
Specialized subset/pad448 wrappers are organized into their dedicated subfolders instead of staying at the top level.
