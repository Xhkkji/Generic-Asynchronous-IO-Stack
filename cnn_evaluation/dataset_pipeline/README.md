ImageNet-1K `pad448` dataset pipeline for `cnn_evaluation`.

Files:

- `cids_prepare_dataset.sh`
  Runs the Python prepared-dataset generator.
- `cids_prepare_dataset_imagenet1k.sh`
  Builds BaM-aligned prepared datasets.
- `cids_load_imagenet1k_to_bam.sh`
  Writes the BaM-aligned `images.bin` files into SSD via BaM benchmark.
- `prepare_imagenet1k_pad448.sh`
  Generates the train/val `pad448` prepared datasets.
- `prepare_imagenet1k_pad448_bam.sh`
  Builds the `pad448_bam` directories.
- `load_imagenet1k_pad448_to_bam.sh`
  Loads the `pad448_bam` datasets into BaM.
- `run_imagenet1k_pad448_pipeline.sh`
  One-shot pipeline: prepare -> bam align -> load to bam.

Suggested order:

1. `bash run_imagenet1k_pad448_pipeline.sh`
2. Run the training scripts in `cnn_evaluation/`.
