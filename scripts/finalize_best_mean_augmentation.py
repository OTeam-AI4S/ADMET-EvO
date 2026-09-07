#!/usr/bin/env python3
"""Wait for all 22 developments, run missing best-mean confirmations, and build the workbook."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_all22_optimal_recipe_adaptive_data as runner


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "reports" / "all22_optimal_recipe_adaptive_data_v2"
DEVELOPMENT = ROOT / "development"
BEST = ROOT / "development_best_mean"
CONFIRMATION = ROOT / "confirmation_best_mean"
OLD_CONFIRMATION = ROOT / "confirmation"
LOGS = ROOT / "logs"
GPU_TASKS = {"Lipophilicity_AstraZeneca", "Half_Life_Obach", "hERG", "DILI"}
CPU_PYTHON = Path("/home/wyilin/.conda/envs/admet-agent/bin/python")
GPU_PYTHON = Path("/data/projects/admet_evo/.conda/envs/admet-agent-gpu/bin/python")


def _valid_development(path: Path, task: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("task") == task and payload.get("status") != "execution_failed"


def _matching_confirmation(task: str, ratio: float) -> bool:
    for directory in (CONFIRMATION, OLD_CONFIRMATION):
        path = directory / f"{task}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        selected = payload.get("selected_ratio")
        if selected is not None and abs(float(selected) - ratio) < 1e-12:
            return True
    return False


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def main() -> None:
    expected = list(runner.TASKS)
    while True:
        missing = [
            task
            for task in expected
            if not _valid_development(DEVELOPMENT / f"{task}.json", task)
        ]
        if not missing:
            break
        print(json.dumps({"waiting_for_development": missing}), flush=True)
        time.sleep(60)

    _run(
        [
            str(CPU_PYTHON),
            "scripts/build_best_mean_development_selection.py",
            "--source-dir",
            str(DEVELOPMENT),
            "--output-dir",
            str(BEST),
        ]
    )

    missing_confirmation: list[str] = []
    for task in expected:
        payload = json.loads((BEST / f"{task}.json").read_text(encoding="utf-8"))
        selection = payload.get("selection", {})
        ratio = selection.get("selected_ratio")
        mean = selection.get("best_mean")
        if ratio is None or mean is None or float(mean) <= 0:
            continue
        if not _matching_confirmation(task, float(ratio)):
            missing_confirmation.append(task)

    CONFIRMATION.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen, object]] = []
    gpu_index = 0
    for task in missing_confirmation:
        is_gpu = task in GPU_TASKS
        python = GPU_PYTHON if is_gpu else CPU_PYTHON
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT)
        env["PYTHONNOUSERSITE"] = "1"
        env["LD_LIBRARY_PATH"] = str(python.parent.parent / "lib")
        if is_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            env["UNIMOL_WEIGHT_DIR"] = str(PROJECT / "checkpoints" / "unimol_tools")
            for name in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy",
            ):
                env.pop(name, None)
            gpu_index += 1
        command = [
            str(python),
            "scripts/run_all22_optimal_recipe_adaptive_data.py",
            "--phase",
            "confirmation",
            "--tasks",
            task,
            "--development-dir",
            str(BEST),
            "--output-dir",
            str(CONFIRMATION),
            "--device",
            "cuda" if is_gpu else "cpu",
            "--n-jobs",
            "12",
            "--overwrite",
        ]
        log_handle = (LOGS / f"{task}.best_mean.confirmation.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((task, process, log_handle))
        print(json.dumps({"confirmation_started": task, "pid": process.pid}), flush=True)

    failures: dict[str, int] = {}
    for task, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode:
            failures[task] = returncode
    if failures:
        raise RuntimeError(f"Confirmation failures: {failures}")

    _run([str(CPU_PYTHON), "scripts/build_all22_best_mean_augmentation_workbook.py"])
    status = {
        "status": "complete",
        "development_tasks": len(expected),
        "confirmations_started": missing_confirmation,
        "workbook": str((ROOT / "all22_best_mean_augmentation_audit.xlsx").resolve()),
    }
    (ROOT / "best_mean_finalizer_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
