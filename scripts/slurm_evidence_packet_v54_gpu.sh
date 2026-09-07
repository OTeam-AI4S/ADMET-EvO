#!/usr/bin/env bash
#SBATCH --job-name=ep-v54-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/data/projects/admet_evo/reports/evidence_packet_data_axis_v54/slurm/logs/%x-%A_%a.out
#SBATCH --error=/data/projects/admet_evo/reports/evidence_packet_data_axis_v54/slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
PYTHON="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/bin/python"
TASKS=(DILI Half_Life_Obach Lipophilicity_AstraZeneca hERG)

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

"$PYTHON" scripts/run_evidence_packet_frozen_recipe_v54.py \
  --phase "$PHASE" \
  --task "$TASK" \
  --protocol reports/evidence_packet_data_axis_v54/protocol.json \
  --mapping runs/v27-maplight-gnn-freeze/full-22-task.mapping.json \
  --data-dir data \
  --output "reports/evidence_packet_data_axis_v54/$PHASE/$TASK.json" \
  --feature-cache-root checkpoints/evidence_packet_v54/features \
  --artifact-root reports/evidence_packet_data_axis_v54/artifacts \
  --device cuda \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --overwrite \
  "${PREVIOUS_ARGS[@]}"
