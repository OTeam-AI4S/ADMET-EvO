"""Execute only the scheduled low-cost corrective matched experiments.

This runner is validation-only.  It never requests the TDC test split and it
reuses completed fits from the same source snapshot/run directory on resume.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from admet_agent.action_evidence import evaluate_action_evidence
from admet_agent.profiling import profile_task
from admet_agent.research_loop import EvidenceLedger
from admet_agent.runner import ADMETAgent
from admet_agent.schemas import Budget, ExperimentSpec, RunResult
from admet_agent.tasks import resolve_task

RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")
SMOKE_SEEDS = (1, 2, 3)
_CONTROL_MODELS = {"balanced_extra_trees": "extra_trees"}


def _columns(frame) -> tuple[str, str]:
    smiles = "Drug" if "Drug" in frame.columns else "SMILES"
    if smiles not in frame.columns or "Y" not in frame.columns:
        raise ValueError(f"Unexpected TDC columns: {list(frame.columns)}")
    return smiles, "Y"


def _result_from_memory(row: dict[str, Any]) -> RunResult:
    names = {item.name for item in fields(RunResult)}
    return RunResult(**{name: row[name] for name in names})


def _signature(spec: ExperimentSpec) -> tuple[Any, ...]:
    return (
        spec.task,
        spec.seed,
        spec.model,
        spec.representations,
        spec.imbalance,
        spec.data_policy,
        json.dumps(spec.hyperparameters, sort_keys=True),
    )


def _existing(agent: ADMETAgent, spec: ExperimentSpec) -> RunResult | None:
    wanted = _signature(spec)
    candidates = []
    for row in agent.memory.records(spec.task):
        try:
            row_spec = ExperimentSpec.from_dict(
                {
                    name: row[name]
                    for name in {item.name for item in fields(ExperimentSpec)}
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
        if _signature(row_spec) == wanted and row.get("status") == "completed":
            candidates.append(row)
    return _result_from_memory(candidates[-1]) if candidates else None


def _action(plan: dict[str, Any]) -> dict[str, Any]:
    selected = plan["selected_action_id"]
    action = next(
        (item for item in plan["actions"] if item.get("action_id") == selected),
        None,
    )
    if action is None:
        raise ValueError("Scheduled correction is missing its selected action")
    return action


def _specs(
    task_name: str,
    plan: dict[str, Any],
    *,
    seed: int,
    imbalance: str,
    n_jobs: int,
) -> tuple[ExperimentSpec, ExperimentSpec]:
    action = _action(plan)
    if action.get("axis") != "model":
        raise ValueError(
            "The bounded smoke runner currently accepts model-axis pairs only"
        )
    payload = action.get("payload", {})
    treatment_model = payload.get("candidate_model")
    control_model = _CONTROL_MODELS.get(treatment_model)
    if not treatment_model or not control_model:
        raise ValueError("Scheduled model action has no preregistered matched control")
    common = {
        "task": task_name,
        "seed": seed,
        "representations": RICH_2D,
        "imbalance": imbalance,
        "data_policy": "official",
        "device": "cpu",
        "n_jobs": n_jobs,
        "hyperparameters": {},
        "budget": Budget(max_candidates=2, max_minutes=15),
    }
    treatment = ExperimentSpec(
        **common,
        model=treatment_model,
        hypothesis=(
            f"corrective model intervention linked to {action['corrects_action_id']}"
        ),
    )
    control = ExperimentSpec(
        **common,
        model=control_model,
        hypothesis=(f"matched model control linked to {action['corrects_action_id']}"),
    )
    return treatment, control


def run(
    *,
    schedule_path: Path,
    data_dir: Path,
    run_dir: Path,
    feature_cache_dir: Path,
    n_jobs: int,
) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule.get("test_loaded") is not False:
        raise ValueError("Correction schedule must be validation-only")
    rows = [
        row
        for row in schedule["decisions"]
        if row.get("execution_status") == "scheduled"
    ]
    if not rows:
        raise ValueError("Correction schedule contains no executable tasks")
    agent = ADMETAgent(
        data_dir,
        run_dir,
        device="cpu",
        n_jobs=n_jobs,
        feature_cache_dir=feature_cache_dir,
        persist_discovery_models=False,
    )
    ledger = EvidenceLedger(run_dir / "evidence_ledger.jsonl")
    task_reports = []
    for row in rows:
        task_name = row["task"]
        output_path = run_dir / "tasks" / task_name / "corrective_summary.json"
        if output_path.exists():
            completed = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                completed.get("source_sha256") == agent.source_manifest["source_sha256"]
                and completed.get("validation_seeds") == list(SMOKE_SEEDS)
                and completed.get("test_loaded") is False
            ):
                task_reports.append(completed)
                continue
            raise RuntimeError(
                f"Refusing incompatible corrective summary: {output_path}"
            )
        plan = row["next_plan"]
        ledger.append(
            "research_plan_created",
            {
                "task": task_name,
                "research_plan": plan,
                "correction_smoke": True,
                "test_loaded": False,
            },
        )
        treatment_runs: dict[int, RunResult] = {}
        control_runs: dict[int, RunResult] = {}
        treatment_primary: tuple[str, ExperimentSpec] | None = None
        control_primary: tuple[str, ExperimentSpec] | None = None
        all_results = []
        for seed in SMOKE_SEEDS:
            train, valid = agent.group.get_train_valid_split(
                benchmark=task_name,
                split_type="default",
                seed=seed,
            )
            smiles_col, y_col = _columns(train)
            task = resolve_task(task_name)
            profile = profile_task(train, valid, task, smiles_col, y_col)
            imbalance = (
                "class_weight"
                if profile.task_type == "classification"
                and profile.positive_rate_train is not None
                and not 0.3 <= profile.positive_rate_train <= 0.7
                else "none"
            )
            treatment, control = _specs(
                task_name,
                plan,
                seed=seed,
                imbalance=imbalance,
                n_jobs=n_jobs,
            )
            seed_results = []
            for role, spec in (("treatment", treatment), ("control", control)):
                result = _existing(agent, spec)
                if result is None:
                    result = agent._execute(
                        spec,
                        train,
                        valid,
                        smiles_col,
                        y_col,
                        profile,
                        timeout_seconds=spec.budget.max_minutes * 60,
                    )
                if result.status != "completed":
                    raise RuntimeError(
                        f"Corrective {role} failed for {task_name} seed {seed}: "
                        f"{result.error}"
                    )
                seed_results.append(result)
                all_results.append(result.to_dict())
            treatment_result, control_result = seed_results
            treatment_runs[seed] = treatment_result
            control_runs[seed] = control_result
            if treatment_primary is None:
                treatment_primary = (treatment_result.run_id, treatment)
                control_primary = (control_result.run_id, control)
        if treatment_primary is None or control_primary is None:
            raise RuntimeError(f"No corrective results produced for {task_name}")
        treatment_id, treatment_spec = treatment_primary
        control_id, control_spec = control_primary
        evidence = evaluate_action_evidence(
            plan,
            resolve_task(task_name),
            {treatment_id: treatment_spec, control_id: control_spec},
            {treatment_id: treatment_runs, control_id: control_runs},
        ).to_dict()
        action = _action(plan)
        evidence.update(
            {
                "task": task_name,
                "attempt_id": action["attempt_id"],
                "sequence_index": action["sequence_index"],
                "action_family": action["action_family"],
                "corrects_action_id": action["corrects_action_id"],
                "parent_action_id": action["parent_action_id"],
                "preregistered": True,
                "execution_status": "completed",
                "test_loaded": False,
            }
        )
        ledger.append(
            "research_action_evaluated",
            {
                "task": task_name,
                "evidence": evidence,
                "correction_smoke": True,
                "test_loaded": False,
            },
        )
        report = {
            "task": task_name,
            "source_sha256": agent.source_manifest["source_sha256"],
            "validation_seeds": list(SMOKE_SEEDS),
            "research_plan": plan,
            "action_evidence": evidence,
            "results": all_results,
            "test_loaded": False,
            "selection_feedback_to_discovery": False,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        task_reports.append(report)
    payload = {
        "protocol": "validation-only-self-correction-smoke-v1",
        "schedule": str(schedule_path),
        "source_sha256": agent.source_manifest["source_sha256"],
        "validation_seeds": list(SMOKE_SEEDS),
        "tasks": task_reports,
        "completed_tasks": len(task_reports),
        "total_fits": sum(len(row["results"]) for row in task_reports),
        "total_fit_seconds": sum(
            result["elapsed_seconds"]
            for row in task_reports
            for result in row["results"]
        ),
        "test_loaded": False,
        "selection_feedback_to_discovery": False,
    }
    (run_dir / "corrective_smoke_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("reports/self_correction_schedule_v1.json"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/v5-self-correction-smoke")
    )
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=Path("cache/features")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    payload = run(
        schedule_path=args.schedule,
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        feature_cache_dir=args.feature_cache_dir,
        n_jobs=args.workers,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
