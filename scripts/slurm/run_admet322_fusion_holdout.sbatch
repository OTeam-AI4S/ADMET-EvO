#!/bin/bash
#SBATCH --job-name=admet322-hold
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/wyilin/admet/runs/admet372/logs/%A_%a.out
#SBATCH --error=/home/wyilin/admet/runs/admet372/logs/%A_%a.err

set -euo pipefail
cd /home/wyilin/admet
/home/wyilin/.conda/envs/admet-agent/bin/python -m scripts.run_admet_fusion_holdout_endpoint \
  --cohort "${ADMET322_CONFIRMATION_COHORT}" \
  --recipe-dir "${ADMET372_RECIPE_DIR}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --output-dir "${ADMET322_HOLDOUT_OUTPUT_DIR}"
