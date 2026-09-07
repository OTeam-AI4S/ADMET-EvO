#!/bin/bash
#SBATCH --job-name=admet50-build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/wyilin/admet/runs/admet50/logs/build_%j.out
#SBATCH --error=/home/wyilin/admet/runs/admet50/logs/build_%j.err

set -euo pipefail
cd /home/wyilin/admet
/home/wyilin/.conda/envs/admet-agent/bin/python -m scripts.prepare_admet50_discovery_cohort \
  --workers "${SLURM_CPUS_PER_TASK}"
