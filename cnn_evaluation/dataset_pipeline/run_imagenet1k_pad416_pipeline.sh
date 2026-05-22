#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PAD416_PIPELINE] step 1/2 build bam-aligned dataset directly"
bash "${SCRIPT_DIR}/prepare_imagenet1k_pad416_bam.sh"

echo "[PAD416_PIPELINE] step 2/2 load bam-aligned dataset into BaM"
bash "${SCRIPT_DIR}/load_imagenet1k_pad416_to_bam.sh"

echo "[PAD416_PIPELINE] done"
