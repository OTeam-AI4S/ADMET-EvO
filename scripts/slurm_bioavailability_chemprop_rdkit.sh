#!/usr/bin/env bash
#SBATCH --job-name=bio-chemprop-rdkit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/admet_evo/runs/bioavailability-chemprop-rdkit-tdc5/slurm-%j.out
#SBATCH --error=/data/projects/admet_evo/runs/bioavailability-chemprop-rdkit-tdc5/slurm-%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
RUN_DIR="$PROJECT_ROOT/runs/bioavailability-chemprop-rdkit-tdc5"
mkdir -p "$RUN_DIR"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR="$RUN_DIR/matplotlib-cache"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PROJECT_ROOT/.venvs/chemprop/bin/python" \
  "$PROJECT_ROOT/scripts/run_bioavailability_chemprop_rdkit.py" \
  --output-dir "$RUN_DIR" \
  --seeds 1 2 3 4 5 \
  --epochs 30 \
  --ensemble-size 1 \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"
