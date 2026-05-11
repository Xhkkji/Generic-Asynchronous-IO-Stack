#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/home/xhk/hyperion/GIDS
RUN_SCRIPT="${REPO_ROOT}/cnn_evaluation/cids_train_imagenet1k_alexnet.sh"
LOG_DIR="${REPO_ROOT}/cnn_evaluation/logs_alexnet"
SUMMARY_LOG="${LOG_DIR}/alexnet_compare_summary.log"

mkdir -p "${LOG_DIR}"
: > "${SUMMARY_LOG}"

echo "[ALEXNET_COMPARE] 1/4 torch mmap"
IO_MODE=torch \
TORCH_READ_MODE=mmap \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_alexnet_torch_mmap" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/alexnet_torch_mmap.log"

echo "[ALEXNET_COMPARE] 2/4 torch buffered"
IO_MODE=torch \
TORCH_READ_MODE=buffered \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_alexnet_torch_buffered" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/alexnet_torch_buffered.log"

echo "[ALEXNET_COMPARE] 3/4 sync"
IO_MODE=sync \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_alexnet_sync" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/alexnet_sync.log"

echo "[ALEXNET_COMPARE] 4/4 registered"
IO_MODE=registered \
PROFILE_DIR="${REPO_ROOT}/cnn_evaluation/cids_profile_alexnet_registered" \
bash "${RUN_SCRIPT}" | tee "${LOG_DIR}/alexnet_registered.log"

echo "[ALEXNET_COMPARE] done"
echo "[ALEXNET_COMPARE] logs: ${LOG_DIR}"
echo "[ALEXNET_COMPARE] summary"
{
  echo "[ALEXNET_COMPARE] logs: ${LOG_DIR}"
  echo "[ALEXNET_COMPARE] summary"
} | tee -a "${SUMMARY_LOG}"
for log_name in \
  alexnet_torch_mmap.log \
  alexnet_torch_buffered.log \
  alexnet_sync.log \
  alexnet_registered.log
do
  summary_line="$(grep '^\[CIDS_TRAIN_SUMMARY\]' "${LOG_DIR}/${log_name}" | tail -n 1 || true)"
  echo "[ALEXNET_COMPARE] ${log_name}: ${summary_line:-MISSING_SUMMARY}" | tee -a "${SUMMARY_LOG}"
done
