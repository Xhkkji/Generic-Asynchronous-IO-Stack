#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch/bin/python
PREP_SCRIPT=/home/xhk/hyperion/GIDS/cnn_evaluation/cids_prepare_dataset.py

"${PYTHON_BIN}" "${PREP_SCRIPT}" "$@"
