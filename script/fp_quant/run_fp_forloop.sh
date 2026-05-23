#!/usr/bin/env bash
# Run FP activation presets concurrently (one per GPU) with per-job logs.
#
# Run from anywhere:
#   bash script/fp_quant/run_fp_forloop.sh
#   STEP=eval bash script/fp_quant/run_fp_forloop.sh
#   nohup bash script/fp_quant/run_fp_forloop.sh > log_fp_forloop.log 2>&1 &
#
set -euo pipefail

FP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${FP_SCRIPT_DIR}/../.." && pwd)"
RUNNER="${FP_SCRIPT_DIR}/qwen3-4b-fp.sh"

STEP="${STEP:-3}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log/fp_forloop}"

PRESETS=(
  # fp8_e4m3_perchannel
  e4m3_perchannel
  nvfp4_perchannel
  nvfp4_plus_perchannel
)
# GPUS=(0 1 2 3)
GPUS=(1 2 3)

if (( ${#PRESETS[@]} != ${#GPUS[@]} )); then
  echo "PRESETS and GPUS must have the same length" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "[INFO] STEP=${STEP}"
echo "[INFO] LOG_DIR=${LOG_DIR}"

run_one_preset() {
  local gpu="$1"
  local preset="$2"
  local log_file="$3"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  {
    echo "[INFO] gpu=${gpu} preset=${preset} step=${STEP}"
    bash "${RUNNER}" "${preset}" "${STEP}"
  } > "${log_file}" 2>&1
}

pids=()
declare -A pid_meta=()
failed=0

for i in "${!PRESETS[@]}"; do
  preset="${PRESETS[$i]}"
  gpu="${GPUS[$i]}"
  log_file="${LOG_DIR}/gpu${gpu}_${preset}.log"

  echo "=== launch gpu=${gpu} preset=${preset} (log: ${log_file}) ==="
  run_one_preset "${gpu}" "${preset}" "${log_file}" &
  pid=$!
  pids+=("${pid}")
  pid_meta["${pid}"]="gpu=${gpu} preset=${preset}"
done

for pid in "${pids[@]}"; do
  if wait "${pid}"; then
    echo "[INFO] ${pid_meta[$pid]} finished ok"
  else
    echo "[ERROR] ${pid_meta[$pid]} failed (see logs in ${LOG_DIR})" >&2
    failed=1
  fi
done

if (( failed )); then
  echo "[ERROR] fp forloop finished with failures" >&2
  exit 1
fi

echo "[INFO] fp forloop complete"
