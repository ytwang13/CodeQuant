#!/usr/bin/env bash
# CodeQuant FP activation pipeline for configs/act_fp_quant/qwen3_4_act_*.yaml
#
# Run from anywhere:
#   bash script/fp_quant/qwen3-4b-fp.sh [preset|config] [step]
#   CONFIG=act_fp_quant/qwen3_4_act_e4m3_perblock.yaml bash script/fp_quant/qwen3-4b-fp.sh all
#
# Preset shorthand (maps to act_fp_quant/qwen3_4_act_<preset>.yaml):
#   fp8_e4m3_perchannel, e4m3_perblock, nvfp4_plus_perblock, ...
#
# Steps: 1 | rotation | 2 | cluster | 3 | eval | all (default: all)
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
    echo "${CONFIG:-act_fp_quant/qwen3_4_act_fp8_e4m3_perchannel.yaml}"
    return
  fi
  if [[ "${arg}" == */* ]]; then
    echo "${arg}"
    return
  fi
  if [[ "${arg}" == *.yaml ]]; then
    echo "act_fp_quant/${arg}"
    return
  fi
  echo "act_fp_quant/qwen3_4_act_${arg}.yaml"
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
  echo "Expected act_fp_quant preset (e.g. e4m3_perblock) or step: ${VALID_STEPS}" >&2
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
  python rotation_fine_tune_script.py --config "${CONFIG}"
}

run_cluster() {
  echo "=== Step 2: ACCF cluster fine-tune (${CONFIG}) ==="
  python cluster_fine_tune_script.py --config "${CONFIG}"
}

run_eval() {
  echo "=== Step 3: evaluation (${CONFIG}) ==="
  python evaluation_script_rotonly.py --config "${CONFIG}"
}
# script/evaluation_script_rotonly.py
# evaluation_script
echo "[INFO] CONFIG=${CONFIG} STEP=${STEP}"

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
