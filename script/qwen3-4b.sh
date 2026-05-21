#!/usr/bin/env bash
# CodeQuant pipeline for configs/qwen3_4.yaml (Qwen/Qwen3-4B).
#
# Run from anywhere:
#   bash script/qwen3-4b.sh [step]
#
# Steps: 1 | rotation | 2 | cluster | 3 | eval | all (default: all)
#
set -euo pipefail
source /home/wyt/miniconda3/bin/activate code
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-qwen3_4.yaml}"
STEP="${1:-all}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1 # 4b only uses one GPU
export HF_HOME=/mnt/hdd/wyt/hf

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
  python evaluation_script.py --config "${CONFIG}"
}

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
    echo "Usage: $0 [1|rotation|2|cluster|3|eval|all]" >&2
    exit 1
    ;;
esac
