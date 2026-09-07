#!/usr/bin/env bash
#SBATCH --job-name=dili-safe-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/data/projects/admet_evo/reports/dili_leakage_safe_best_seed_v1/slurm/%x-%j.out
#SBATCH --error=/data/projects/admet_evo/reports/dili_leakage_safe_best_seed_v1/slurm/%x-%j.err

set -euo pipefail
PROJECT_ROOT=/data/projects/admet_evo
PYTHON="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/bin/python"
PHASE=${PHASE:?PHASE must be development or certification}
SEEDS=(271 307 347 389 431)
cd "$PROJECT_ROOT"
mkdir -p reports/dili_leakage_safe_best_seed_v1/{development,certification,slurm}
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

if [[ "$PHASE" == "development" ]]; then
  SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}
  "$PYTHON" scripts/run_dili_development_shard_v1.py \
    --seed "$SEED" \
    --protocol reports/evidence_packet_data_axis_v54/protocol.json \
    --mapping runs/v27-maplight-gnn-freeze/full-22-task.mapping.json \
    --data-dir data \
    --output "reports/dili_leakage_safe_best_seed_v1/development_shards/seed_$SEED.json" \
    --feature-cache-root "reports/dili_leakage_safe_best_seed_v1/artifacts/features_seed_$SEED" \
    --artifact-root "reports/dili_leakage_safe_best_seed_v1/artifacts/development_models/seed_$SEED" \
    --device cuda --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
elif [[ "$PHASE" == "certification" ]]; then
  "$PYTHON" scripts/finalize_dili_leakage_safe_best_seed_v1.py \
    --phase certification --device cuda --n-jobs "${SLURM_CPUS_PER_TASK:-8}" --overwrite
else
  echo "invalid PHASE=$PHASE" >&2
  exit 2
fi
