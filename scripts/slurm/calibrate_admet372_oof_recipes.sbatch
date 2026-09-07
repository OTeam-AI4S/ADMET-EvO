#!/bin/bash
#SBATCH --job-name=admet372-cal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wyilin/admet/runs/admet372/logs/%A_%a.out
#SBATCH --error=/home/wyilin/admet/runs/admet372/logs/%A_%a.err

set -euo pipefail
cd /home/wyilin/admet
/home/wyilin/.conda/envs/admet-agent/bin/python -m scripts.calibrate_admet372_oof_recipes \
  --input-dir "${ADMET372_UNCALIBRATED_DIR}" \
  --output-dir "${ADMET372_CALIBRATED_DIR}" \
  --expected-endpoints 372 \
  --amendment reports/admet322_holdout_preregistration_amendment_threshold_v2.json \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count 48
