#!/usr/bin/env bash
# Background run of all FP presets (full pipeline each).
#
#   bash script/fp_quant/nohup-all.sh
#   STEP=cluster bash script/fp_quant/nohup-all.sh
#
set -euo pipefail

FP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${FP_SCRIPT_DIR}/../.." && pwd)"
STEP="${STEP:-all}"
LOG="${REPO_ROOT}/log_fp_run_all_${STEP}.log"

nohup env STEP="${STEP}" bash "${FP_SCRIPT_DIR}/run-all.sh" > "${LOG}" 2>&1 &
echo "[INFO] started run-all (STEP=${STEP}), pid=$!, log=${LOG}"
