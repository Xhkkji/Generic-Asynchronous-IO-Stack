ResNet-18 CIDS line migrated from the old `evaluation/` location.

Files:

- `cids_resnet18.py`
- `cids_train.py`
- `cids_train.sh`
- `cids_train_imagenet1k.sh`

This folder now contains local runnable copies instead of depending on
`evaluation/cids_*`.

Profiles are stored under:

- `profiles/torch_mmap`
- `profiles/torch_buffered`
- `profiles/sync`
- `profiles/registered`

Historical traces originally left in `cids_profile/` have been copied into the
mode-specific subdirectories above.
