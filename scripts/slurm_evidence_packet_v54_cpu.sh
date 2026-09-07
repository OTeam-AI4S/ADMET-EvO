#!/usr/bin/env bash
#SBATCH --job-name=ep-v54-cpu
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/data/projects/admet_evo/reports/evidence_packet_data_axis_v54/slurm/logs/%x-%A_%a.out
#SBATCH --error=/data/projects/admet_evo/reports/evidence_packet_data_axis_v54/slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
PYTHON="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/bin/python"
TASKS=(
  AMES BBB_Martins Bioavailability_Ma CYP2C9_Substrate_CarbonMangels
  CYP2C9_Veith CYP2D6_Substrate_CarbonMangels CYP2D6_Veith
  CYP3A4_Substrate_CarbonMangels CYP3A4_Veith Caco2_Wang
  Clearance_Hepatocyte_AZ Clearance_Microsome_AZ HIA_Hou LD50_Zhu
  PPBR_AZ Pgp_Broccatelli Solubility_AqSolDB VDss_Lombardo
)

PHASE=${PHASE:?PHASE must be development, confirmation, or certification}
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}
case "$PHASE" in
  development) PREVIOUS_ARGS=() ;;
  confirmation)
    PREVIOUS_ARGS=(--previous-artifact "reports/evidence_packet_data_axis_v54/development/$TASK.json") ;;
  certification)
    PREVIOUS_ARGS=(--previous-artifact "reports/evidence_packet_data_axis_v54/confirmation/$TASK.json") ;;
  *) echo "invalid PHASE=$PHASE" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
mkdir -p "reports/evidence_packet_data_axis_v54/$PHASE" \
  reports/evidence_packet_data_axis_v54/slurm/logs
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT/scripts:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ADMET_MAPLIGHT_CHECKPOINT_ROOT=/home/wyilin/.cache/molfeat
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$PYTHON" scripts/run_evidence_packet_frozen_recipe_v54.py \
  --phase "$PHASE" \
  --task "$TASK" \
  --protocol reports/evidence_packet_data_axis_v54/protocol.json \
  --mapping runs/v27-maplight-gnn-freeze/full-22-task.mapping.json \
  --data-dir data \
  --output "reports/evidence_packet_data_axis_v54/$PHASE/$TASK.json" \
  --feature-cache-root checkpoints/evidence_packet_v54/features \
  --artifact-root reports/evidence_packet_data_axis_v54/artifacts \
  --device cpu \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --overwrite \
  "${PREVIOUS_ARGS[@]}"
