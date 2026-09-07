#!/usr/bin/env bash
#SBATCH --job-name=v27-ext-env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v2/slurm/logs/bootstrap-%j.out
#SBATCH --error=/data/projects/admet_evo/reports/all22_optimal_recipe_adaptive_data_v2/slurm/logs/bootstrap-%j.err

set -euo pipefail

PROJECT_ROOT=/data/projects/admet_evo
MAIN_ENV="$PROJECT_ROOT/.envs/admet-v27-external-data"
MAPLIGHT_ENV="$PROJECT_ROOT/.venvs/maplight-gnn"
CHECKPOINT_ROOT="$PROJECT_ROOT/.cache/molfeat"
AUDIT_ROOT="$PROJECT_ROOT/reports/all22_optimal_recipe_adaptive_data_v2/environment"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/.envs" "$PROJECT_ROOT/.venvs" "$AUDIT_ROOT" \
  "$CHECKPOINT_ROOT/gin_supervised_masking"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/scripts/runtime_shims:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export UNIMOL_WEIGHT_DIR="$PROJECT_ROOT/runs/unimol2-weights"
export ADMET_MAPLIGHT_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"

echo "host=$(hostname) job_id=${SLURM_JOB_ID:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ ! -x "$MAIN_ENV/bin/python" ]]; then
  conda env create --prefix "$MAIN_ENV" --file environment-external-data.yml
fi

if [[ ! -x "$MAPLIGHT_ENV/bin/python" ]]; then
  conda env create --prefix "$MAPLIGHT_ENV" \
    --file vendor/tdc-admet-bench-audit/models/maplight_gnn/wrapper/environment.yml
fi

MODEL_URL=https://fs.molfeat.datamol.io/artifacts/dgllife/gin_supervised_masking/0/model.save
METADATA_URL=https://fs.molfeat.datamol.io/artifacts/dgllife/gin_supervised_masking/0/metadata.json
MODEL_PATH="$CHECKPOINT_ROOT/gin_supervised_masking/model.save"
METADATA_PATH="$CHECKPOINT_ROOT/gin_supervised_masking/metadata.json"
if [[ ! -s "$MODEL_PATH" ]]; then
  curl --fail --location --retry 5 --output "$MODEL_PATH" "$MODEL_URL"
fi
if [[ ! -s "$METADATA_PATH" ]]; then
  curl --fail --location --retry 5 --output "$METADATA_PATH" "$METADATA_URL"
fi
echo '6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9  '"$MODEL_PATH" | sha256sum --check
echo '75ea305d643d800b8b272819a78b842d5aca1c4ad55d47207321ab9b81d44d02  '"$METADATA_PATH" | sha256sum --check

"$MAIN_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import json
import platform
import sys

import torch
from admet_agent.schemas import ExperimentSpec
from admet_agent.frozen_executor import FrozenRecipeExecutor
from scripts.run_all22_optimal_recipe_adaptive_data import PROTOCOL

names = [
    "PyTDC", "rdkit", "numpy", "scipy", "pandas", "scikit-learn",
    "joblib", "xgboost", "lightgbm", "catboost", "torch",
    "unimol-tools", "openpyxl", "pytest",
]
versions = {name: md.version(name) for name in names}
payload = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": versions,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "runner_protocol": PROTOCOL,
    "import_check": [ExperimentSpec.__name__, FrozenRecipeExecutor.__name__],
    "acl_compatibility_shim": True,
}
with open(
    "reports/all22_optimal_recipe_adaptive_data_v2/environment/main_environment.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in the exact main environment")
PY

LD_LIBRARY_PATH="$MAPLIGHT_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$MAPLIGHT_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import json
import sys

names = [
    "numpy", "pandas", "scikit-learn", "rdkit", "torch", "torchdata",
    "molfeat", "dgl", "dgllife", "catboost", "PyTDC",
]
payload = {
    "python": sys.version,
    "executable": sys.executable,
    "packages": {name: md.version(name) for name in names},
}
with open(
    "reports/all22_optimal_recipe_adaptive_data_v2/environment/maplight_environment.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

conda list --explicit --prefix "$MAIN_ENV" > "$AUDIT_ROOT/main-conda-explicit.txt"
"$MAIN_ENV/bin/python" -m pip freeze --all > "$AUDIT_ROOT/main-pip-freeze.txt"
conda list --explicit --prefix "$MAPLIGHT_ENV" > "$AUDIT_ROOT/maplight-conda-explicit.txt"
"$MAPLIGHT_ENV/bin/python" -m pip freeze --all > "$AUDIT_ROOT/maplight-pip-freeze.txt"
"$MAIN_ENV/bin/python" -m pytest -q tests/test_all22_optimal_recipe_adaptive_data.py

echo "BOOTSTRAP_OK"
