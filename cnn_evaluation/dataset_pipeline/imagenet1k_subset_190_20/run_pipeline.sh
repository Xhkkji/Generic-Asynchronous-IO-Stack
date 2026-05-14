#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[IMAGENET_SUBSET_PIPELINE] step 1/3 build subset"
bash "${SCRIPT_DIR}/build_subset.sh"

echo "[IMAGENET_SUBSET_PIPELINE] step 2/3 prepare pad448 dataset"
bash "${SCRIPT_DIR}/prepare_pad448.sh"

echo "[IMAGENET_SUBSET_PIPELINE] step 3/3 load pad448 dataset into BaM"
bash "${SCRIPT_DIR}/load_pad448_to_bam.sh"

echo "[IMAGENET_SUBSET_PIPELINE] done"
