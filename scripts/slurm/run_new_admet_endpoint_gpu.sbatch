#!/bin/bash
#SBATCH --job-name=new-admet-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/wyilin/admet/runs/new_admet/logs/%A_%a.out
#SBATCH --error=/home/wyilin/admet/runs/new_admet/logs/%A_%a.err

set -euo pipefail
cd /home/wyilin/admet

EXTRA_ARGS=()
if [[ -n "${NEW_ADMET_HOLDOUT_FOLDS:-}" ]]; then
  EXTRA_ARGS+=(--holdout-folds "${NEW_ADMET_HOLDOUT_FOLDS}")
fi

/home/wyilin/.conda/envs/admet-agent/bin/python -m scripts.run_new_admet_endpoint \
  --cohort "${NEW_ADMET_COHORT}" \
  --seeds "${NEW_ADMET_SEEDS}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${NEW_ADMET_OUTPUT_DIR}" \
  --actions gpu_catboost \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "${EXTRA_ARGS[@]}"
