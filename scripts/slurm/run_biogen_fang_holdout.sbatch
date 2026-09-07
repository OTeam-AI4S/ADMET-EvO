#!/bin/bash
#SBATCH --job-name=fang-holdout
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

/home/wyilin/.conda/envs/admet-agent/bin/python -m scripts.run_biogen_fang_holdout \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir /home/wyilin/admet/runs/new_admet/biogen_fang_holdout \
  --workers "${SLURM_CPUS_PER_TASK}"
