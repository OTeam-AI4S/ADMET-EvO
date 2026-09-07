#!/bin/bash
#SBATCH --job-name=admet372-build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/wyilin/admet/runs/admet372/logs/build_%j.out
#SBATCH --error=/home/wyilin/admet/runs/admet372/logs/build_%j.err

set -euo pipefail
cd /home/wyilin/admet
/home/wyilin/.conda/envs/admet-agent/bin/python -m scripts.prepare_admet50_discovery_cohort \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --all-eligible \
  --version admet372-all-eligible-toxicology-cohort-v1 \
  --output reports/admet372_all_eligible_toxicology_cohort_v1.csv \
  --manifest reports/admet372_all_eligible_toxicology_cohort_manifest_v1.json

/home/wyilin/.conda/envs/admet-agent/bin/python -m scripts.prepare_admet50_strict_split \
  --source reports/admet372_all_eligible_toxicology_cohort_v1.csv \
  --output reports/admet372_model_selection_cohort_v1.csv \
  --manifest reports/admet372_strict_split_manifest_v1.json \
  --version admet372-strict-selection-confirmation-split-v1
