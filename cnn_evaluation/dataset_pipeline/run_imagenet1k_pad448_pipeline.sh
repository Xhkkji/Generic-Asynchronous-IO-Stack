#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PAD448_PIPELINE] step 1/3 prepare fixed-size dataset"
bash "${SCRIPT_DIR}/prepare_imagenet1k_pad448.sh"

echo "[PAD448_PIPELINE] step 2/3 build bam-aligned dataset"
bash "${SCRIPT_DIR}/prepare_imagenet1k_pad448_bam.sh"

echo "[PAD448_PIPELINE] step 3/3 load bam-aligned dataset into BaM"
bash "${SCRIPT_DIR}/load_imagenet1k_pad448_to_bam.sh"

echo "[PAD448_PIPELINE] done"
