#!/usr/bin/env bash
#SBATCH --job-name=ep-v54-final
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/admet_evo/reports/evidence_packet_data_axis_v54/slurm/logs/finalize-%j.out
#SBATCH --error=/data/projects/admet_evo/reports/evidence_packet_data_axis_v54/slurm/logs/finalize-%j.err

set -euo pipefail
PROJECT_ROOT=/data/projects/admet_evo
PYTHON="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/bin/python"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/build_evidence_packet_v54_status.py \
  --root reports/evidence_packet_data_axis_v54
