#!/usr/bin/env bash
# Sweep rotation fine_tune_lr: AOS fine-tune + rotation-only eval per LR.
# Runs up to one experiment per GPU concurrently (default: GPUs 0,1,2).
#
# Run from anywhere:
#   bash script/fp_quant/sweep_rotation_lr.sh
#   ROTATION_LRS="0.0001 0.0005 0.001" bash script/fp_quant/sweep_rotation_lr.sh
#   GPUS="0,1,2" ROTATION_LRS="0.0001 0.001" bash script/fp_quant/sweep_rotation_lr.sh
#   nohup bash script/fp_quant/sweep_rotation_lr.sh > log_sweep_rot_lr.log 2>&1 &
#
set -euo pipefail
source /home/wyt/miniconda3/bin/activate code

FP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "${FP_SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/hdd/wyt/hf}"

CONFIG="${CONFIG:-act_fp_quant_clus/qwen3_4_act_fp8_e4m3_perchannel.yaml}"
ROTATION_LRS="${ROTATION_LRS:-0.00001 0.00005 0.0001 0.0005 0.001}"
GPUS="${GPUS:-1,2,3}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log/sweep_rotation_lr}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
MAX_PARALLEL="${#GPU_ARRAY[@]}"
if (( MAX_PARALLEL < 1 )); then
  echo "GPUS must list at least one device, got: ${GPUS}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/configs/${CONFIG}" ]]; then
  echo "Config not found: ${REPO_ROOT}/configs/${CONFIG}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] ROTATION_LRS=${ROTATION_LRS}"
echo "[INFO] GPUS=${GPUS} (max ${MAX_PARALLEL} concurrent)"
echo "[INFO] LOG_DIR=${LOG_DIR}"

lr_tag() {
  python - <<PY
lr = float("${1}")
print(format(lr, "f").rstrip("0").rstrip("."))
PY
}

run_one_lr() {
  local gpu="$1"
  local lr="$2"
  local log_file="$3"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  {
    echo "[INFO] gpu=${gpu} rotation fine-tune, lr=${lr}"
    python rotation_fine_tune_script.py --config "${CONFIG}" --rotation-lr "${lr}"

    echo "[INFO] gpu=${gpu} rotation-only eval, lr=${lr}"
    python evaluation_script_rotonly.py --config "${CONFIG}" --rotation-lr "${lr}"
  } > "${log_file}" 2>&1
}

free_gpus=("${GPU_ARRAY[@]}")
declare -A pid_gpu=()
job_pids=()
failed=0

wait_for_free_gpu() {
  local finished_pid gpu
  if ! wait -n -p finished_pid; then
    gpu="${pid_gpu[$finished_pid]}"
    echo "[ERROR] gpu=${gpu} pid=${finished_pid} failed (see logs in ${LOG_DIR})" >&2
    failed=1
  else
    gpu="${pid_gpu[$finished_pid]}"
    echo "[INFO] gpu=${gpu} pid=${finished_pid} finished ok"
  fi
  unset "pid_gpu[${finished_pid}]"
  free_gpus+=("${gpu}")
}

for lr in ${ROTATION_LRS}; do
  while (( ${#free_gpus[@]} == 0 )); do
    wait_for_free_gpu
  done

  gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")
  lr_label="$(lr_tag "${lr}")"
  log_file="${LOG_DIR}/gpu${gpu}_rot_lr_${lr_label}.log"

  echo "=== launch gpu=${gpu} rotation_lr=${lr} (log: ${log_file}) ==="
  run_one_lr "${gpu}" "${lr}" "${log_file}" &
  pid=$!
  job_pids+=("${pid}")
  pid_gpu["${pid}"]="${gpu}"
done

while (( ${#job_pids[@]} > 0 )); do
  wait_for_free_gpu
  new_pids=()
  for pid in "${job_pids[@]}"; do
    if [[ -n "${pid_gpu[$pid]+x}" ]]; then
      new_pids+=("${pid}")
    fi
  done
  job_pids=("${new_pids[@]}")
done

if (( failed )); then
  echo "[ERROR] sweep finished with failures" >&2
  exit 1
fi

echo "[INFO] sweep complete"
