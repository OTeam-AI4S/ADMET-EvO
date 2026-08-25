"""Run classic-ML TDC-only baselines for all tasks under the safe expansion policy.

Tasks rejected by the frozen external-data admission report are deliberately
baseline-only.  The single admitted controlled bridge (hERG) is handled by
``run_classic_ml_external_data_confirmation.py`` so that its baseline and
augmented arms remain paired.  TDC test files are never opened.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from run_classic_ml_external_data_confirmation import (
    DEFAULT_MODELS,
    DEFAULT_SEEDS,
    TASKS,
    _classification_model,
    _classification_prediction,
    _regression_model,
    _regression_prediction,
    _score,
    _standardized_tdc,
)
from tdc.benchmark_group import admet_group

from admet_agent.herg_data_scaling import fingerprints_to_csr, morgan_fingerprints

PROTOCOL = "classic-ml-all22-safe-policy-baseline-v1"


def _baseline_summary(rows: list[dict], model_name: str) -> dict:
    scores = np.asarray(
        [row["baseline_score"] for row in rows if row["model"] == model_name],
        dtype=float,
    )
    return {
        "baseline_mean": float(scores.mean()),
        "baseline_std": float(scores.std()),
        "seed_count": len(scores),
    }


def _complete_result(path: Path, *, task: str, seeds: tuple[int, ...], models: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("protocol") == PROTOCOL
        and payload.get("task") == task
        and payload.get("seeds") == list(seeds)
        and payload.get("models") == list(models)
        and len(payload.get("per_seed_model", [])) == len(seeds) * len(models)
    )


def run_task(group, args: argparse.Namespace, task: str) -> dict:
    benchmark, task_type, metric, _ = TASKS[task]
    started = time.perf_counter()
    rows: list[dict] = []
    fallback_count = 0
    for seed in args.seeds:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=benchmark, split_type="default", seed=seed
        )
        train, train_fallbacks = _standardized_tdc(train_raw)
        valid, valid_fallbacks = _standardized_tdc(valid_raw)
        fallback_count += train_fallbacks + valid_fallbacks
        train_x = fingerprints_to_csr(
            morgan_fingerprints(train["standard_smiles"].tolist())
        )
        valid_x = fingerprints_to_csr(
            morgan_fingerprints(valid["standard_smiles"].tolist())
        )
        dtype = int if task_type == "classification" else float
        train_y = train["Y"].to_numpy(dtype=dtype)
        valid_y = valid["Y"].to_numpy(dtype=dtype)
        for model_name in args.models:
            fit_started = time.perf_counter()
            if task_type == "classification":
                prediction = _classification_prediction(
                    _classification_model(model_name, seed, args.n_jobs),
                    train_x,
                    train_y,
                    valid_x,
                )
            else:
                prediction = _regression_prediction(
                    _regression_model(model_name, seed, args.n_jobs),
                    train_x,
                    train_y,
                    valid_x,
                )
            row = {
                "seed": seed,
                "model": model_name,
                "tdc_train_rows": len(train),
                "tdc_validation_rows": len(valid),
                "baseline_score": _score(metric, valid_y, prediction),
                "fit_wall_seconds": time.perf_counter() - fit_started,
            }
            rows.append(row)
            print(json.dumps({"task": task, **row}), flush=True)
    result = {
        "protocol": PROTOCOL,
        "task": task,
        "benchmark": benchmark,
        "task_type": task_type,
        "metric": metric,
        "seeds": list(args.seeds),
        "models": list(args.models),
        "per_seed_model": rows,
        "model_summaries": {
            model: _baseline_summary(rows, model) for model in args.models
        },
        "tdc_standardization_fallback_count": fallback_count,
        "policy_decision": "baseline_only_external_data_failed_quality_gate",
        "safety": {
            "tdc_test_loaded": False,
            "test_labels_accessible": False,
            "external_rows_loaded": False,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), action="append", dest="tasks")
    parser.add_argument("--model", choices=DEFAULT_MODELS, action="append", dest="models")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/classic_ml_all22_safe_policy_v1/baselines"),
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.tasks = tuple(dict.fromkeys(args.tasks or TASKS))
    args.models = tuple(dict.fromkeys(args.models or DEFAULT_MODELS))
    args.seeds = tuple(dict.fromkeys(args.seeds or DEFAULT_SEEDS))
    if 1 in args.seeds:
        raise ValueError("Seed 1 is reserved for the earlier development screen")
    return args


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(args.data_dir))
    for task in args.tasks:
        output = args.output_dir / f"{task}.json"
        if not args.force and _complete_result(
            output, task=task, seeds=args.seeds, models=args.models
        ):
            print(json.dumps({"task": task, "status": "skipped_complete"}), flush=True)
            continue
        result = run_task(group, args, task)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"task": task, "status": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main(parse_args())
