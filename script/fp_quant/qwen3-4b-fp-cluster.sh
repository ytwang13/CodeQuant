#!/usr/bin/env bash
# CodeQuant FP activation pipeline for configs/act_fp_quant_clus/qwen3_4_act_*.yaml
#
# Run from anywhere:
#   bash script/fp_quant/qwen3-4b-fp.sh [preset|config] [step]
#   CONFIG=act_fp_quant_clus/qwen3_4_act_e4m3_perblock.yaml bash script/fp_quant/qwen3-4b-fp.sh all
#
# Preset shorthand (maps to act_fp_quant_clus/qwen3_4_act_<preset>.yaml):
#   fp8_e4m3_perchannel, e4m3_perblock, nvfp4_plus_perblock, ...
#
# Steps: 1 | rotation | 2 | cluster | 3 | eval | all (default: all)
#
# Optional LR overrides (also used by run_fp_forloop_cluster.sh):
#   ROTATION_LR=0.00001 CLUSTER_LR=0.0001 bash script/fp_quant/qwen3-4b-fp.sh fp8_e4m3_perchannel cluster
#
set -euo pipefail
source /home/wyt/miniconda3/bin/activate code

FP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "${FP_SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/mnt/hdd/wyt/hf}"

VALID_STEPS="1|rotation|aos|2|cluster|accf|3|eval|evaluation|all"

resolve_config() {
  local arg="${1:-}"
  if [[ -z "${arg}" ]]; then
    echo "${CONFIG:-act_fp_quant_clus/qwen3_4_act_fp8_e4m3_perchannel.yaml}"
    return
  fi
  if [[ "${arg}" == */* ]]; then
    echo "${arg}"
    return
  fi
  if [[ "${arg}" == *.yaml ]]; then
    echo "act_fp_quant_clus/${arg}"
    return
  fi
  echo "act_fp_quant_clus/qwen3_4_act_${arg}.yaml"
}

is_step() {
  case "${1:-}" in
    1|rotation|aos|2|cluster|accf|3|eval|evaluation|all) return 0 ;;
    *) return 1 ;;
  esac
}

config_exists() {
  [[ -f "${REPO_ROOT}/configs/$(resolve_config "$1")" ]]
}

if [[ $# -ge 1 ]] && is_step "${1}"; then
  CONFIG="$(resolve_config "")"
  STEP="${1}"
elif [[ $# -ge 1 ]] && config_exists "${1}"; then
  CONFIG="$(resolve_config "${1}")"
  STEP="${2:-all}"
elif [[ $# -ge 1 ]]; then
  echo "Unknown preset or config: ${1}" >&2
  echo "Expected act_fp_quant_clus preset (e.g. e4m3_perblock) or step: ${VALID_STEPS}" >&2
  exit 1
else
  CONFIG="$(resolve_config "")"
  STEP="all"
fi

if [[ ! -f "${REPO_ROOT}/configs/${CONFIG}" ]]; then
  echo "Config not found: ${REPO_ROOT}/configs/${CONFIG}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

run_rotation() {
  echo "=== Step 1: AOS rotation fine-tune (${CONFIG}) ==="
  local -a extra_args=()
  if [[ -n "${ROTATION_LR:-}" ]]; then
    extra_args+=(--rotation-lr "${ROTATION_LR}")
  fi
  python rotation_fine_tune_script.py --config "${CONFIG}" "${extra_args[@]}"
}

run_cluster() {
  echo "=== Step 2: ACCF cluster fine-tune (${CONFIG}) ==="
  local -a extra_args=()
  [[ -n "${ROTATION_LR:-}" ]] && extra_args+=(--rotation-lr "${ROTATION_LR}")
  [[ -n "${CLUSTER_LR:-}" ]] && extra_args+=(--cluster-lr "${CLUSTER_LR}")
  python cluster_fine_tune_script.py --config "${CONFIG}" "${extra_args[@]}"
}

run_eval() {
  echo "=== Step 3: evaluation (${CONFIG}) ==="
  local -a extra_args=()
  [[ -n "${ROTATION_LR:-}" ]] && extra_args+=(--rotation-lr "${ROTATION_LR}")
  [[ -n "${CLUSTER_LR:-}" ]] && extra_args+=(--cluster-lr "${CLUSTER_LR}")
  python evaluation_script.py --config "${CONFIG}" "${extra_args[@]}"
}
# script/evaluation_script_rotonly.py
# evaluation_script
echo "[INFO] CONFIG=${CONFIG} STEP=${STEP}"
[[ -n "${ROTATION_LR:-}" ]] && echo "[INFO] ROTATION_LR=${ROTATION_LR}"
[[ -n "${CLUSTER_LR:-}" ]] && echo "[INFO] CLUSTER_LR=${CLUSTER_LR}"

case "${STEP}" in
  1|rotation|aos)
    run_rotation
    ;;
  2|cluster|accf)
    run_cluster
    ;;
  3|eval|evaluation)
    run_eval
    ;;
  all)
    run_rotation
    run_cluster
    run_eval
    ;;
  *)
    echo "Unknown step: ${STEP}" >&2
    echo "Usage: $0 [preset|config] [1|rotation|2|cluster|3|eval|all]" >&2
    exit 1
    ;;
esac
