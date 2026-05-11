#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/home/xhk/hyperion/GIDS
RUN_SCRIPT="${REPO_ROOT}/cnn_evaluation/cids_train_imagenet1k_resnet50.sh"
LOG_DIR="${REPO_ROOT}/cnn_evaluation/logs_resnet50"
SUMMARY_LOG="${LOG_DIR}/resnet50_compare_summary.log"

mkdir -p "${LOG_DIR}"
: > "${SUMMARY_LOG}"

echo "[RESNET50_COMPARE] 1/4 torch mmap"
IO_MODE=torch \
TORCH_READ_MODE=mmap \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_resnet50_torch_mmap" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/resnet50_torch_mmap.log"

echo "[RESNET50_COMPARE] 2/4 torch buffered"
IO_MODE=torch \
TORCH_READ_MODE=buffered \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_resnet50_torch_buffered" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/resnet50_torch_buffered.log"

echo "[RESNET50_COMPARE] 3/4 sync"
IO_MODE=sync \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_resnet50_sync" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/resnet50_sync.log"

echo "[RESNET50_COMPARE] 4/4 registered"
IO_MODE=registered \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_resnet50_registered" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/resnet50_registered.log"

echo "[RESNET50_COMPARE] done"
echo "[RESNET50_COMPARE] logs: ${LOG_DIR}"
echo "[RESNET50_COMPARE] summary"
{
  echo "[RESNET50_COMPARE] logs: ${LOG_DIR}"
  echo "[RESNET50_COMPARE] summary"
} | tee -a "${SUMMARY_LOG}"
for log_name in \
  resnet50_torch_mmap.log \
  resnet50_torch_buffered.log \
  resnet50_sync.log \
  resnet50_registered.log
do
  summary_line="$(grep '^\[CIDS_TRAIN_SUMMARY\]' "${LOG_DIR}/${log_name}" | tail -n 1 || true)"
  echo "[RESNET50_COMPARE] ${log_name}: ${summary_line:-MISSING_SUMMARY}" | tee -a "${SUMMARY_LOG}"
done
