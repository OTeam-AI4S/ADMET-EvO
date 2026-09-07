#!/usr/bin/env bash
#SBATCH --job-name=admet-combined-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/data/projects/admet_evo/reports/combined_external_best_seed_v1/slurm/%x-%A_%a.out
#SBATCH --error=/data/projects/admet_evo/reports/combined_external_best_seed_v1/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
PYTHON=/data/projects/admet_evo/.conda/envs/admet-agent-gpu/bin/python
TASKS=(
  Caco2_Wang HIA_Hou Pgp_Broccatelli Bioavailability_Ma
  Lipophilicity_AstraZeneca AMES Solubility_AqSolDB BBB_Martins
  PPBR_AZ VDss_Lombardo CYP2C9_Veith CYP2D6_Veith CYP3A4_Veith
  CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels
  CYP3A4_Substrate_CarbonMangels Half_Life_Obach Clearance_Hepatocyte_AZ
  Clearance_Microsome_AZ LD50_Zhu hERG DILI
)

PHASE=${PHASE:?PHASE must be development-herg or certification}
cd "$PROJECT_ROOT"
mkdir -p reports/combined_external_best_seed_v1/slurm
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT/scripts:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UNIMOL_WEIGHT_DIR="$PROJECT_ROOT/runs/unimol2-weights"
export HF_HOME="$UNIMOL_WEIGHT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
test -s "$UNIMOL_WEIGHT_DIR/modelzoo/84M/checkpoint.pt"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

if [[ "$PHASE" == "development-herg" ]]; then
  "$PYTHON" scripts/run_combined_external_best_seed_v1.py \
    --phase development-herg --device cuda --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
elif [[ "$PHASE" == "certification" ]]; then
  TASK=${TASKS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}
  "$PYTHON" scripts/run_combined_external_best_seed_v1.py \
    --phase certification --task "$TASK" --device cuda \
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
else
  echo "invalid PHASE=$PHASE" >&2
  exit 2
fi
