#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/evaluation/run_homogenous_train_debug.sh"
LOG_DIR="${ROOT_DIR}/evaluation/logs/async_scan"

mkdir -p "${LOG_DIR}"

STEP_GROUPS_RAW="${SCAN_STEP_GROUPS:-20,40,60}"
DEBUG_ROWS_RAW="${SCAN_DEBUG_ROWS_LIST:-32}"
DEFAULT_DEBUG_ROWS="${SCAN_DEFAULT_DEBUG_ROWS:-0}"
DEBUG_DIMS="${SCAN_DEBUG_DIMS:-8}"
WARP_CTX_SAMPLE="${SCAN_WARP_CTX_SAMPLE:-32}"
FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-0}"

IFS=';' read -r -a STEP_GROUPS <<< "${STEP_GROUPS_RAW}"
IFS=',' read -r -a DEBUG_ROWS_LIST <<< "${DEBUG_ROWS_RAW}"

max_step_from_group() {
  local group="$1"
  local max_step=-1
  local step

  IFS=',' read -r -a step_array <<< "${group}"
  for step in "${step_array[@]}"; do
    step="${step// /}"
    [[ -z "${step}" ]] && continue
    if (( step > max_step )); then
      max_step="${step}"
    fi
  done

  echo "${max_step}"
}

sanitize_group() {
  local group="$1"
  group="${group//,/x}"
  group="${group// /}"
  echo "${group}"
}

for step_group in "${STEP_GROUPS[@]}"; do
  [[ -z "${step_group// /}" ]] && continue
  stop_after_step="$(max_step_from_group "${step_group}")"
  if (( stop_after_step < 0 )); then
    echo "[scan] skip empty step group: ${step_group}" >&2
    continue
  fi

  for debug_rows in "${DEBUG_ROWS_LIST[@]}"; do
    debug_rows="${debug_rows// /}"
    [[ -z "${debug_rows}" ]] && continue

    safe_group="$(sanitize_group "${step_group}")"
    timestamp="$(date +%Y%m%d-%H%M%S)"
    log_file="${LOG_DIR}/async_steps-${safe_group}_rows-${debug_rows}_${timestamp}.log"

    echo "[scan] steps=${step_group} stop_after_step=${stop_after_step} rows=${debug_rows} dims=${DEBUG_DIMS} ctx_sample=${WARP_CTX_SAMPLE}"
    echo "[scan] log=${log_file}"

    GIDS_FORCE_SYNC_READ="${FORCE_SYNC_READ}" \
    GIDS_ASYNC_DEBUG_ROWS="${DEFAULT_DEBUG_ROWS}" \
    GIDS_ASYNC_DEBUG_DIMS="${DEBUG_DIMS}" \
    GIDS_WARP_CTX_DEBUG_SAMPLE="${WARP_CTX_SAMPLE}" \
    bash "${RUN_SCRIPT}" \
      --stop_after_step "${stop_after_step}" \
      --async_debug_steps "${step_group}" \
      --async_debug_rows "${debug_rows}" \
      --async_debug_default_rows "${DEFAULT_DEBUG_ROWS}" \
      --async_debug_dims "${DEBUG_DIMS}" \
      --async_debug_warp_ctx_sample "${WARP_CTX_SAMPLE}" \
      "$@" > "${log_file}" 2>&1
  done
done
