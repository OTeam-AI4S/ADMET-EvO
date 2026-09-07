#!/usr/bin/env bash
#SBATCH --job-name=v27-ext-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v2/slurm/logs/smoke-%j.out
#SBATCH --error=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v2/slurm/logs/smoke-%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
MAIN_ENV=/data/shared/environments/torch213-0718
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-unset}"
id
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ ! -r admet_agent/data_governance.py ]]; then
  echo "ACL_SHIM_REQUIRED=admet_agent/data_governance.py"
  stat -c '%A %a %U:%G %u:%g %n' admet_agent/data_governance.py
fi

if [[ ! -x "$MAIN_ENV/bin/python" ]]; then
  echo "MISSING_MAIN_ENV=$MAIN_ENV" >&2
  exit 43
fi

"$MAIN_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import json
import sys

packages = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "rdkit",
    "PyTDC",
    "joblib",
    "xgboost",
    "lightgbm",
    "torch",
    "unimol-tools",
    "pytest",
    "openpyxl",
]
versions = {}
for package in packages:
    try:
        versions[package] = md.version(package)
    except md.PackageNotFoundError:
        versions[package] = None

import torch
from admet_agent.schemas import ExperimentSpec
from admet_agent.frozen_executor import FrozenRecipeExecutor
from scripts.run_all22_optimal_recipe_adaptive_data import PROTOCOL
import admet_agent.data_governance as data_governance

payload = {
    "python": sys.version,
    "executable": sys.executable,
    "packages": versions,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "runner_protocol": PROTOCOL,
    "imports": [ExperimentSpec.__name__, FrozenRecipeExecutor.__name__],
    "data_governance_source": data_governance.__file__,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the recorded Uni-Mol2 environment")
PY

CHECKPOINT_ROOT="${HOME}/.cache/molfeat/gin_supervised_masking"
for checkpoint_file in model.save metadata.json; do
  if [[ ! -r "$CHECKPOINT_ROOT/$checkpoint_file" ]]; then
    echo "MISSING_MAPLIGHT_CHECKPOINT=$CHECKPOINT_ROOT/$checkpoint_file" >&2
  else
    sha256sum "$CHECKPOINT_ROOT/$checkpoint_file"
  fi
done

echo "SMOKE_OK"
