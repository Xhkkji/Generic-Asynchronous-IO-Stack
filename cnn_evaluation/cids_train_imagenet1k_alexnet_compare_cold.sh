#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/home/xhk/hyperion/GIDS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${REPO_ROOT}/cnn_evaluation/cids_train_imagenet1k_alexnet.sh"
LOG_DIR="${REPO_ROOT}/cnn_evaluation/logs_alexnet"
SUMMARY_LOG="${LOG_DIR}/alexnet_compare_cold_summary.log"
PROFILE_ROOT="${SCRIPT_DIR}/profiles/cold"

mkdir -p "${LOG_DIR}"
: > "${SUMMARY_LOG}"

drop_linux_caches() {
  echo "[ALEXNET_COMPARE_COLD] sync + drop_caches" | tee -a "${SUMMARY_LOG}"
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

run_branch() {
  local step="$1"
  local name="$2"
  local io_mode="$3"
  local torch_read_mode="$4"
  local profile_dir="$5"
  local log_path="$6"

  echo "[ALEXNET_COMPARE_COLD] ${step}/4 ${name}"
  echo "[ALEXNET_COMPARE_COLD] ${step}/4 ${name}" >> "${SUMMARY_LOG}"
  drop_linux_caches

  if [[ -n "${torch_read_mode}" ]]; then
    IO_MODE="${io_mode}" \
    TORCH_READ_MODE="${torch_read_mode}" \
    PROFILE_DIR="${profile_dir}" \
    bash "${RUN_SCRIPT}" | tee "${log_path}"
  else
    IO_MODE="${io_mode}" \
    PROFILE_DIR="${profile_dir}" \
    bash "${RUN_SCRIPT}" | tee "${log_path}"
  fi
}

run_branch \
  1 \
  "torch mmap" \
  "torch" \
  "mmap" \
  "${PROFILE_ROOT}/torch_mmap" \
  "${LOG_DIR}/alexnet_torch_mmap_cold.log"

run_branch \
  2 \
  "torch buffered" \
  "torch" \
  "buffered" \
  "${PROFILE_ROOT}/torch_buffered" \
  "${LOG_DIR}/alexnet_torch_buffered_cold.log"

run_branch \
  3 \
  "sync" \
  "sync" \
  "" \
  "${PROFILE_ROOT}/sync" \
  "${LOG_DIR}/alexnet_sync_cold.log"

run_branch \
  4 \
  "registered" \
  "registered" \
  "" \
  "${PROFILE_ROOT}/registered" \
  "${LOG_DIR}/alexnet_registered_cold.log"

echo "[ALEXNET_COMPARE_COLD] done"
echo "[ALEXNET_COMPARE_COLD] logs: ${LOG_DIR}"
echo "[ALEXNET_COMPARE_COLD] summary"
{
  echo "[ALEXNET_COMPARE_COLD] logs: ${LOG_DIR}"
  echo "[ALEXNET_COMPARE_COLD] summary"
} | tee -a "${SUMMARY_LOG}"

for log_name in \
  alexnet_torch_mmap_cold.log \
  alexnet_torch_buffered_cold.log \
  alexnet_sync_cold.log \
  alexnet_registered_cold.log
do
  summary_line="$(grep '^\[CIDS_TRAIN_SUMMARY\]' "${LOG_DIR}/${log_name}" | tail -n 1 || true)"
  echo "[ALEXNET_COMPARE_COLD] ${log_name}: ${summary_line:-MISSING_SUMMARY}" | tee -a "${SUMMARY_LOG}"
done
