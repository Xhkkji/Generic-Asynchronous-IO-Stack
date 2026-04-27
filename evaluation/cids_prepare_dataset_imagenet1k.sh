#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch/bin/python
PREP_SCRIPT=/home/xhk/hyperion/GIDS/evaluation/cids_prepare_dataset_imagenet1k.py

"${PYTHON_BIN}" "${PREP_SCRIPT}" "$@"
