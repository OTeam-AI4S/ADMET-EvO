#!/usr/bin/env bash
#SBATCH --job-name=admet-3axis-v3-final
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/admet_evo/reports/all22_three_axis_ablation_v3/slurm/logs/finalize-%j.out
#SBATCH --error=/data/projects/admet_evo/reports/all22_three_axis_ablation_v3/slurm/logs/finalize-%j.err

set -euo pipefail
PROJECT_ROOT=/data/projects/admet_evo
PYTHON=/home/wyilin/.conda/envs/admet-agent/bin/python
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export THREE_AXIS_VERSION=v3
"$PYTHON" scripts/build_all22_three_axis_ablation_summary_v2.py
test -s reports/all22_three_axis_ablation_v3/all22_three_axis_ablation_v3_results.csv
test -s reports/all22_three_axis_ablation_v3/all22_three_axis_ablation_v3_results.xlsx
test -s reports/all22_three_axis_ablation_v3/all22_three_axis_ablation_v3_results.audit.json
echo "THREE_AXIS_V3_FINALIZE_OK"
