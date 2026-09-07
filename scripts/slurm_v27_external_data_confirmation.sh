#!/usr/bin/env bash
#SBATCH --job-name=v27-ext-confirm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v3/slurm/logs/confirmation-%j.out
#SBATCH --error=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v3/slurm/logs/confirmation-%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
PYTHON="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/bin/python"
TASK="${TASK:?TASK must name one Development-passing endpoint}"
DEVICE="${DEVICE:?DEVICE must be cpu or cuda}"

cd "$PROJECT_ROOT"
test -x "$PYTHON"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UNIMOL_WEIGHT_DIR="$PROJECT_ROOT/runs/unimol2-weights"
export ADMET_MAPLIGHT_CHECKPOINT_ROOT=/home/wyilin/.cache/molfeat
export HF_HOME="$UNIMOL_WEIGHT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$PROJECT_ROOT/.conda/envs/admet-agent-gpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-unset} task=$TASK device=$DEVICE"
if [[ "$DEVICE" == cuda ]]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  test -s "$UNIMOL_WEIGHT_DIR/modelzoo/84M/checkpoint.pt"
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
fi

"$PYTHON" scripts/run_all22_optimal_recipe_adaptive_data.py \
  --phase confirmation \
  --tasks "$TASK" \
  --mapping runs/v27-maplight-gnn-freeze/full-22-task.mapping.json \
  --output-dir reports/all22_optimal_recipe_adaptive_data_v3/confirmation \
  --development-dir reports/all22_optimal_recipe_adaptive_data_v3/development \
  --data-dir data \
  --cohort-dir data/external/chembl/chembl_37/tdc22_cohorts \
  --admission-report reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json \
  --device "$DEVICE" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --overwrite

"$PYTHON" - "$TASK" <<'PY'
import json
import sys
from pathlib import Path

task = sys.argv[1]
root = Path("reports/all22_optimal_recipe_adaptive_data_v3/confirmation")
error = root / f"{task}.error.json"
result = root / f"{task}.json"
if error.exists():
    payload = json.loads(error.read_text())
    raise SystemExit(f"{task} failed: {payload.get('error_type')}: {payload.get('error')}")
if not result.exists():
    raise SystemExit(f"{task} produced no confirmation artifact")
payload = json.loads(result.read_text())
safety = payload.get("safety", {})
if not (
    safety.get("tdc_test_loaded") is True
    and safety.get("tdc_test_structures_loaded_for_firewall") is True
    and safety.get("tdc_test_columns_loaded") == ["Drug"]
    and safety.get("tdc_test_labels_loaded") is False
    and safety.get("test_labels_accessible") is False
):
    raise SystemExit(f"{task} failed the structure-only TDC-test firewall assertion")
print(json.dumps({"task": task, "status": payload.get("status"), "summary": payload.get("agent_summary")}, sort_keys=True))
PY
