#!/usr/bin/env bash
#SBATCH --job-name=v27-ext-best
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v3/slurm/logs/best-seed-%j.out
#SBATCH --error=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v3/slurm/logs/best-seed-%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
PYTHON="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/bin/python"

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT/scripts:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON" scripts/build_all22_v3_best_seed_summary.py
