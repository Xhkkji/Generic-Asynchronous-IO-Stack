#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch/bin/python
PREP_SCRIPT="${SCRIPT_DIR}/../cids_prepare_dataset.py"

"${PYTHON_BIN}" "${PREP_SCRIPT}" "$@"
