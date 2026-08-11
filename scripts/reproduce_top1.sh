#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/tdc-top1-reproduction}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"

cd "${PROJECT_ROOT}"

for task in AMES PPBR_AZ Solubility_AqSolDB; do
  "${PYTHON_BIN}" -m admet_agent.cli finalize \
    --task "${task}" \
    --seeds 1,2,3,4,5 \
    --finalists results/top1_finalists.json \
    --data-dir "${DATA_DIR}" \
    --run-dir "${RUN_ROOT}/${task}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}"
done

"${PYTHON_BIN}" scripts/verify_top1_results.py --run-root "${RUN_ROOT}"
