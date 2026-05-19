#!/usr/bin/env bash
# CodeQuant pipeline for configs/qwen3.yaml (Qwen/Qwen3-30B-A3B MoE).
#
#   bash script/qwen3-30b.sh [step]
# Steps: 1|rotation, 2|cluster, 3|eval, all (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-qwen3.yaml}"
STEP="${1:-all}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
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
  1|rotation|aos) run_rotation ;;
  2|cluster|accf) run_cluster ;;
  3|eval|evaluation) run_eval ;;
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
