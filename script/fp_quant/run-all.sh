#!/usr/bin/env bash
# Run the full FP activation pipeline for every Qwen3-4B act_fp_quant preset.
#
#   bash script/fp_quant/run-all.sh              # sequential, logs under repo root
#   bash script/fp_quant/run-all.sh rotation     # only step 1 for each preset
#   STEP=eval bash script/fp_quant/run-all.sh    # only eval for each preset
#
set -euo pipefail

FP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${FP_SCRIPT_DIR}/../.." && pwd)"
RUNNER="${FP_SCRIPT_DIR}/qwen3-4b-fp.sh"
STEP="${STEP:-all}"

PRESETS=(
  fp8_e4m3_perchannel
  # # fp8_e4m3_perblock
  # # fp8_e5m2_perchannel
  # # fp8_e5m2_perblock
  # e4m3_perchannel
  # # e4m3_perblock
  # # e5m2_perchannel
  # # e5m2_perblock
  # nvfp4_perchannel
  # # nvfp4_perblock
  # nvfp4_plus_perchannel
  # # nvfp4_plus_perblock
)

for preset in "${PRESETS[@]}"; do
  log="${REPO_ROOT}/log_fp_${preset}.log"
  echo "=== FP preset: ${preset} (step=${STEP}) -> ${log} ==="
  bash "${RUNNER}" "${preset}" "${STEP}" > "${log}" 2>&1
  echo "=== done: ${preset} ==="
done

echo "[INFO] finished ${#PRESETS[@]} presets; logs: ${REPO_ROOT}/log_fp_<preset>.log"
