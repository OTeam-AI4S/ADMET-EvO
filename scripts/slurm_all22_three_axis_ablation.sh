#!/usr/bin/env bash
#SBATCH --job-name=admet-3axis
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/data/projects/admet_evo/reports/all22_three_axis_ablation_v1/slurm/logs/%x-%A_%a.out
#SBATCH --error=/data/projects/admet_evo/reports/all22_three_axis_ablation_v1/slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
PYTHON=/home/wyilin/.conda/envs/admet-agent/bin/python
TASKS=(
  Caco2_Wang
  HIA_Hou
  Pgp_Broccatelli
  Bioavailability_Ma
  Lipophilicity_AstraZeneca
  AMES
  Solubility_AqSolDB
  BBB_Martins
  PPBR_AZ
  VDss_Lombardo
  CYP2C9_Veith
  CYP2D6_Veith
  CYP3A4_Veith
  CYP2C9_Substrate_CarbonMangels
  CYP2D6_Substrate_CarbonMangels
  CYP3A4_Substrate_CarbonMangels
  Half_Life_Obach
  Clearance_Hepatocyte_AZ
  Clearance_Microsome_AZ
  LD50_Zhu
  hERG
  DILI
)

PHASE=${PHASE:?PHASE must be discovery, confirmation, or certification}
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}
case "$PHASE" in
  discovery|confirmation|certification) ;;
  *) echo "Invalid PHASE=$PHASE" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
test -x "$PYTHON"
mkdir -p reports/all22_three_axis_ablation_v1/slurm/logs
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "host=$(hostname) job=${SLURM_JOB_ID:-unset} array=${SLURM_ARRAY_TASK_ID:-unset} phase=$PHASE task=$TASK"
"$PYTHON" scripts/run_all22_three_axis_ablation.py \
  --phase "$PHASE" \
  --task "$TASK" \
  --protocol reports/all22_three_axis_ablation_v1/protocol.json \
  --output-root reports/all22_three_axis_ablation_v1 \
  --data-dir data \
  --external-data data/ADMET_v3_clean_external_candidate_pool.csv \
  --feature-cache-dir reports/all22_three_axis_ablation_v1/artifacts/feature_cache \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" - "$PHASE" "$TASK" <<'PY'
import json
import sys
from pathlib import Path

phase, task = sys.argv[1:]
path = Path("reports/all22_three_axis_ablation_v1") / phase / f"{task}.json"
error = path.with_suffix(".error.json")
if error.exists():
    value = json.loads(error.read_text())
    raise SystemExit(f"{task} {phase} failed: {value.get('error_type')}: {value.get('error')}")
value = json.loads(path.read_text())
if value.get("status") != "completed" or value.get("phase") != phase:
    raise SystemExit(f"Invalid output: {path}")
safety = value.get("safety", {})
if phase == "certification":
    passed = (
        safety.get("test_labels_loaded") is True
        and safety.get("test_labels_loaded_only_after_frozen_confirmation") is True
        and safety.get("test_results_used_for_selection") is False
    )
else:
    passed = (
        safety.get("test_structure_columns_loaded") == ["Drug"]
        and safety.get("test_labels_loaded") is False
        and safety.get("selection_uses_test_results") is False
    )
if not passed:
    raise SystemExit(f"Safety assertion failed: {path}")
print(json.dumps({"phase": phase, "task": task, "status": "verified"}))
PY
