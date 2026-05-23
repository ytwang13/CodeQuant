#!/usr/bin/env bash
# Run FP activation presets concurrently (one per GPU) with per-job logs.
# Passes rotation/cluster LR overrides into qwen3-4b-fp.sh (cluster + eval steps).
#
# Run from anywhere:
#   bash script/fp_quant/run_fp_forloop_cluster.sh
#   STEP=cluster CLUSTER_LR=0.0001 bash script/fp_quant/run_fp_forloop_cluster.sh
#   ROTATION_LR=0.00001 CLUSTER_LRS="0.00001 0.0001 0.0005" bash script/fp_quant/run_fp_forloop_cluster.sh
#   nohup bash script/fp_quant/run_fp_forloop_cluster.sh > log_fp_forloop_cluster.log 2>&1 &
#
set -euo pipefail

FP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${FP_SCRIPT_DIR}/../.." && pwd)"
RUNNER="${FP_SCRIPT_DIR}/qwen3-4b-fp-cluster.sh"

STEP="${STEP:-2}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log/fp_forloop_cluster_ep1}"
ROTATION_LR="${ROTATION_LR:-}"
CLUSTER_LR="${CLUSTER_LR:-}"
CLUSTER_LRS="${CLUSTER_LRS:-}"

PRESETS=(
  fp8_e4m3_perchannel
  e4m3_perchannel
  nvfp4_perchannel
  nvfp4_plus_perchannel
)
GPUS=(0 1 2 3)
# GPUS=(1 2 3)

if (( ${#PRESETS[@]} != ${#GPUS[@]} )); then
  echo "PRESETS and GPUS must have the same length" >&2
  exit 1
fi

read -r -a CLUSTER_LR_ARRAY <<< "${CLUSTER_LRS}"
if (( ${#CLUSTER_LR_ARRAY[@]} > 0 )) && (( ${#CLUSTER_LR_ARRAY[@]} != ${#PRESETS[@]} )); then
  echo "CLUSTER_LRS must have the same length as PRESETS when set (got ${#CLUSTER_LR_ARRAY[@]} LRs, ${#PRESETS[@]} presets)" >&2
  exit 1
fi
if [[ -n "${CLUSTER_LR}" && ${#CLUSTER_LR_ARRAY[@]} -gt 0 ]]; then
  echo "Set CLUSTER_LR or CLUSTER_LRS, not both" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

lr_tag() {
  python - <<PY
lr = float("${1}")
print(format(lr, "f").rstrip("0").rstrip("."))
PY
}

echo "[INFO] STEP=${STEP}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
[[ -n "${ROTATION_LR}" ]] && echo "[INFO] ROTATION_LR=${ROTATION_LR}"
if (( ${#CLUSTER_LR_ARRAY[@]} > 0 )); then
  echo "[INFO] CLUSTER_LRS (per preset)=${CLUSTER_LRS}"
elif [[ -n "${CLUSTER_LR}" ]]; then
  echo "[INFO] CLUSTER_LR=${CLUSTER_LR}"
fi

run_one_preset() {
  local gpu="$1"
  local preset="$2"
  local cluster_lr="$3"
  local log_file="$4"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export ROTATION_LR
  if [[ -n "${cluster_lr}" ]]; then
    export CLUSTER_LR="${cluster_lr}"
  else
    unset CLUSTER_LR || true
  fi
  {
    echo "[INFO] gpu=${gpu} preset=${preset} step=${STEP} rotation_lr=${ROTATION_LR:-<config>} cluster_lr=${cluster_lr:-<config>}"
    bash "${RUNNER}" "${preset}" "${STEP}"
  } > "${log_file}" 2>&1
}

pids=()
declare -A pid_meta=()
failed=0

for i in "${!PRESETS[@]}"; do
  preset="${PRESETS[$i]}"
  gpu="${GPUS[$i]}"
  if (( ${#CLUSTER_LR_ARRAY[@]} > 0 )); then
    preset_cluster_lr="${CLUSTER_LR_ARRAY[$i]}"
  else
    preset_cluster_lr="${CLUSTER_LR}"
  fi

  lr_log_suffix=""
  if [[ -n "${preset_cluster_lr}" ]]; then
    lr_log_suffix="_lr$(lr_tag "${preset_cluster_lr}")"
  fi
  log_file="${LOG_DIR}/gpu${gpu}_${preset}${lr_log_suffix}_${STEP}.log"

  echo "=== launch gpu=${gpu} preset=${preset} cluster_lr=${preset_cluster_lr:-<config>} (log: ${log_file}) ==="
  run_one_preset "${gpu}" "${preset}" "${preset_cluster_lr}" "${log_file}" &
  pid=$!
  pids+=("${pid}")
  pid_meta["${pid}"]="gpu=${gpu} preset=${preset} cluster_lr=${preset_cluster_lr:-<config>}"
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
  echo "[ERROR] fp forloop cluster finished with failures" >&2
  exit 1
fi

echo "[INFO] fp forloop cluster complete"
