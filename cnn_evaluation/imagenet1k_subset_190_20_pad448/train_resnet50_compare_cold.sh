#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/home/xhk/hyperion/GIDS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/train_resnet50.sh"
LOG_DIR="${SCRIPT_DIR}/logs_resnet50"
SUMMARY_LOG="${LOG_DIR}/resnet50_compare_cold_summary.log"

mkdir -p "${LOG_DIR}"
: > "${SUMMARY_LOG}"

drop_linux_caches() {
  echo "[RESNET50_SUBSET_COMPARE_COLD] sync + drop_caches" | tee -a "${SUMMARY_LOG}"
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

  echo "[RESNET50_SUBSET_COMPARE_COLD] ${step}/4 ${name}"
  echo "[RESNET50_SUBSET_COMPARE_COLD] ${step}/4 ${name}" >> "${SUMMARY_LOG}"
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
  "${SCRIPT_DIR}/profiles_resnet50_torch_mmap_cold" \
  "${LOG_DIR}/resnet50_torch_mmap_cold.log"

run_branch \
  2 \
  "torch buffered" \
  "torch" \
  "buffered" \
  "${SCRIPT_DIR}/profiles_resnet50_torch_buffered_cold" \
  "${LOG_DIR}/resnet50_torch_buffered_cold.log"

run_branch \
  3 \
  "sync" \
  "sync" \
  "" \
  "${SCRIPT_DIR}/profiles_resnet50_sync_cold" \
  "${LOG_DIR}/resnet50_sync_cold.log"

run_branch \
  4 \
  "registered" \
  "registered" \
  "" \
  "${SCRIPT_DIR}/profiles_resnet50_registered_cold" \
  "${LOG_DIR}/resnet50_registered_cold.log"

echo "[RESNET50_SUBSET_COMPARE_COLD] done"
echo "[RESNET50_SUBSET_COMPARE_COLD] logs: ${LOG_DIR}"
echo "[RESNET50_SUBSET_COMPARE_COLD] summary"
{
  echo "[RESNET50_SUBSET_COMPARE_COLD] logs: ${LOG_DIR}"
  echo "[RESNET50_SUBSET_COMPARE_COLD] summary"
} | tee -a "${SUMMARY_LOG}"

for log_name in \
  resnet50_torch_mmap_cold.log \
  resnet50_torch_buffered_cold.log \
  resnet50_sync_cold.log \
  resnet50_registered_cold.log
do
  summary_line="$(grep '^\[CIDS_TRAIN_SUMMARY\]' "${LOG_DIR}/${log_name}" | tail -n 1 || true)"
  echo "[RESNET50_SUBSET_COMPARE_COLD] ${log_name}: ${summary_line:-MISSING_SUMMARY}" | tee -a "${SUMMARY_LOG}"
done
