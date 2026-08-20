"""Fresh-seed confirmation of the preregistered metric-tail router guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from scipy.stats import wilcoxon
from tdc.benchmark_group import admet_group

from admet_agent.evidence_router import EvidenceRouter
from admet_agent.features import FeatureCache
from admet_agent.guard import validate_experiment
from admet_agent.runner import _columns, _percentile_ranks
from admet_agent.schemas import ExperimentSpec, RunResult
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task
from scripts.evaluate_router_context_ablation import PENALTIES, choose_penalty
from scripts.evaluate_router_loso import _candidate_evidence, replay_loso


FRESH_SEEDS = (4, 5)
MATERIAL_THRESHOLD = 0.005


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _canonical_identity(value: str) -> str:
    molecule = Chem.MolFromSmiles(str(value))
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        if molecule
        else ""
    )


def guard_policy(metric: str) -> str:
    if metric not in {"auroc", "auprc", "mae", "spearman"}:
        raise ValueError(f"Unsupported metric: {metric}")
    return "score" if metric == "spearman" else "unguarded_cost_aware"


def _aggregate_candidates(row: dict[str, Any]) -> list[RunResult]:
    candidates = []
    for run_id, template in row["templates"].items():
        values = list(row["seed_scores"][run_id].values())
        costs = list(row["seed_costs"][run_id].values())
        candidates.append(
            RunResult(
                **{
                    **template.to_dict(),
                    "representations": tuple(template.representations),
                    "validation_score": statistics.mean(values),
                    "validation_std": statistics.pstdev(values),
                    "elapsed_seconds": sum(costs),
                }
            )
        )
    return candidates


def _load_rows(search_root: Path, data_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(search_root.glob("tasks/*/tasks/*/search_seed_1.json"))
    if len(paths) != 22:
        raise ValueError(f"Expected 22 search summaries, found {len(paths)}")
    group = admet_group(path=str(data_dir))
    rows = []
    for path in paths:
        payload = _json(path)
        profile, templates, seed_scores, seed_costs = _candidate_evidence(
            payload, path, group
        )
        evidence = (profile, templates, seed_scores, seed_costs)
        rows.append(
            {
                "path": path,
                "payload": payload,
                "profile": profile,
                "templates": templates,
                "seed_scores": seed_scores,
                "seed_costs": seed_costs,
                "grid": {
                    penalty: replay_loso(*evidence, cost_penalty=penalty)
                    for penalty in PENALTIES
                },
            }
        )
    return rows


def select_locked_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locked = []
    for heldout, row in enumerate(rows):
        evidence_indices = [index for index in range(len(rows)) if index != heldout]
        penalty = choose_penalty(rows, evidence_indices)
        candidates = _aggregate_candidates(row)
        by_id = {candidate.run_id: candidate for candidate in candidates}
        score_decision = EvidenceRouter(
            uncertainty_penalty=0.0, cost_penalty=0.0
        ).select(row["profile"], candidates)
        cost_decision = EvidenceRouter(
            uncertainty_penalty=0.5, cost_penalty=penalty
        ).select(row["profile"], candidates)
        selected_policy = guard_policy(row["profile"].metric)
        score_id = score_decision["selected_run_id"]
        cost_id = cost_decision["selected_run_id"]
        guard_id = score_id if selected_policy == "score" else cost_id
        locked.append(
            {
                "task": row["profile"].task,
                "metric": row["profile"].metric,
                "selected_cost_penalty": penalty,
                "score_run_id": score_id,
                "unguarded_cost_aware_run_id": cost_id,
                "guard_policy": selected_policy,
                "guard_run_id": guard_id,
                "development_candidate_count": len(by_id),
            }
        )
    return locked


def _experiment_path(row: dict[str, Any], run_id: str) -> Path:
    run_dir = row["path"].parents[2]
    path = run_dir / "artifacts" / run_id / "experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source experiment for {run_id}: {path}")
    return path


def _base_prediction(
    *,
    row: dict[str, Any],
    source_run_id: str,
    seed: int,
    train,
    valid,
    cache: FeatureCache,
    artifact_root: Path,
) -> dict[str, Any]:
    task = resolve_task(row["profile"].task)
    target_dir = artifact_root / task.name / f"seed_{seed}" / source_run_id
    result_path = target_dir / "result.json"
    prediction_path = target_dir / "validation_predictions.npy"
    source_path = _experiment_path(row, source_run_id)
    source = _json(source_path)
    experiment = ExperimentSpec.from_dict(
        {**source, "seed": seed, "device": "cpu", "n_jobs": 8}
    )
    validate_experiment(experiment)
    spec_sha = hashlib.sha256(
        json.dumps(experiment.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if result_path.exists() and prediction_path.exists():
        persisted = _json(result_path)
        if persisted["experiment_sha256"] != spec_sha:
            raise ValueError(f"Cached experiment mismatch at {result_path}")
        return persisted

    target_dir.mkdir(parents=True, exist_ok=True)
    smiles_col, y_col = _columns(train)
    started = time.monotonic()
    combined = train[smiles_col].astype(str).tolist() + valid[smiles_col].astype(str).tolist()
    features = cache.get_or_compute(combined, experiment.representations)
    x_train, x_valid = features[: len(train)], features[len(train) :]
    y_train = train[y_col].to_numpy()
    y_valid = valid[y_col].to_numpy()
    model = build_model(experiment, task, y_train)
    model.fit(x_train, y_train)
    predictions = np.asarray(predict(model, x_valid, task), dtype=float)
    elapsed = time.monotonic() - started
    np.save(prediction_path, predictions, allow_pickle=False)
    result = {
        "source_run_id": source_run_id,
        "source_experiment": source_path.as_posix(),
        "source_experiment_sha256": _sha256(source_path),
        "experiment_sha256": spec_sha,
        "task": task.name,
        "metric": task.metric,
        "seed": seed,
        "validation_score": score(y_valid, predictions, task.metric),
        "elapsed_seconds": elapsed,
        "prediction_sha256": _sha256(prediction_path),
    }
    _write_json(result_path, result)
    return result


def _evaluate_candidate(
    *,
    row: dict[str, Any],
    run_id: str,
    seed: int,
    train,
    valid,
    cache: FeatureCache,
    artifact_root: Path,
) -> dict[str, Any]:
    template = row["templates"][run_id]
    if template.model != "ensemble":
        return _base_prediction(
            row=row,
            source_run_id=run_id,
            seed=seed,
            train=train,
            valid=valid,
            cache=cache,
            artifact_root=artifact_root,
        )

    task = resolve_task(row["profile"].task)
    members = template.hyperparameters["members"]
    weights = template.hyperparameters["weights"]
    aggregation = template.hyperparameters.get("aggregation", "probability")
    member_results = [
        _base_prediction(
            row=row,
            source_run_id=member,
            seed=seed,
            train=train,
            valid=valid,
            cache=cache,
            artifact_root=artifact_root,
        )
        for member in members
    ]
    predictions = []
    for member, result in zip(members, member_results):
        path = artifact_root / task.name / f"seed_{seed}" / member / "validation_predictions.npy"
        values = np.load(path, allow_pickle=False)
        predictions.append(_percentile_ranks(values) if aggregation == "rank" else values)
    combined = sum(float(weight) * values for weight, values in zip(weights, predictions))
    _, y_col = _columns(valid)
    return {
        "source_run_id": run_id,
        "task": task.name,
        "metric": task.metric,
        "seed": seed,
        "validation_score": score(valid[y_col].to_numpy(), combined, task.metric),
        "elapsed_seconds": sum(result["elapsed_seconds"] for result in member_results),
        "members": members,
        "weights": weights,
        "aggregation": aggregation,
    }


def _bootstrap(values: list[float], seed: int = 20260817) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    return [float(value) for value in np.quantile(array[indices].mean(axis=1), [0.025, 0.975])]


def _paired(values: list[float]) -> dict[str, Any]:
    if any(abs(value) > 1e-12 for value in values):
        statistic, p_value = wilcoxon(values, alternative="two-sided")
    else:
        statistic, p_value = 0.0, 1.0
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "bootstrap_mean_95_ci": _bootstrap(values),
        "improved": sum(value > 1e-12 for value in values),
        "tied": sum(abs(value) <= 1e-12 for value in values),
        "worse": sum(value < -1e-12 for value in values),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
    }


def build_report(
    repo_root: Path,
    search_root: Path,
    data_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = repo_root.resolve()
    preregistration = root / "runs/v27-router-risk-guard/preregistration.json"
    rows = _load_rows(search_root, data_dir)
    locked = select_locked_candidates(rows)
    group = admet_group(path=str(data_dir))
    task_records = []
    artifact_root = output_root / "artifacts"
    for row, selection in zip(rows, locked):
        task = resolve_task(selection["task"])
        benchmark = group.get(task.name)
        benchmark_name = benchmark["name"]
        policies = {
            "score": selection["score_run_id"],
            "unguarded_cost_aware": selection["unguarded_cost_aware_run_id"],
            "metric_risk_guard": selection["guard_run_id"],
        }
        seed_records = []
        identity_overlap = {}
        for seed in FRESH_SEEDS:
            train, valid = group.get_train_valid_split(
                benchmark=benchmark_name, split_type="default", seed=seed
            )
            smiles_col, _ = _columns(train)
            train_ids = {_canonical_identity(value) for value in train[smiles_col]}
            valid_ids = {_canonical_identity(value) for value in valid[smiles_col]}
            overlap = len((train_ids & valid_ids) - {""})
            identity_overlap[str(seed)] = overlap
            if overlap:
                raise ValueError(f"Identity leakage for {task.name} seed {seed}: {overlap}")
            cache = FeatureCache(output_root / "feature_cache" / task.name / f"seed_{seed}", device="cpu")
            unique_ids = sorted(set(policies.values()))
            evaluated = {
                run_id: _evaluate_candidate(
                    row=row,
                    run_id=run_id,
                    seed=seed,
                    train=train,
                    valid=valid,
                    cache=cache,
                    artifact_root=artifact_root,
                )
                for run_id in unique_ids
            }
            seed_records.append(
                {
                    "seed": seed,
                    "policies": {
                        name: evaluated[run_id] for name, run_id in policies.items()
                    },
                }
            )
        direction = -1.0 if task.metric == "mae" else 1.0
        summaries = {}
        for name in policies:
            values = [record["policies"][name]["validation_score"] for record in seed_records]
            summaries[name] = {
                "run_id": policies[name],
                "mean": statistics.mean(values),
                "scores": values,
                "elapsed_seconds": sum(
                    record["policies"][name]["elapsed_seconds"] for record in seed_records
                ),
            }
        guard_vs_cost = direction * (
            summaries["metric_risk_guard"]["mean"]
            - summaries["unguarded_cost_aware"]["mean"]
        )
        guard_vs_score = direction * (
            summaries["metric_risk_guard"]["mean"] - summaries["score"]["mean"]
        )
        cost_vs_score = direction * (
            summaries["unguarded_cost_aware"]["mean"] - summaries["score"]["mean"]
        )
        task_records.append(
            {
                **selection,
                "identity_overlap": identity_overlap,
                "fresh_seed_records": seed_records,
                "summaries": summaries,
                "guard_minus_unguarded_oriented_gain": guard_vs_cost,
                "guard_minus_score_oriented_gain": guard_vs_score,
                "unguarded_minus_score_oriented_gain": cost_vs_score,
            }
        )

    guard_gains = [row["guard_minus_unguarded_oriented_gain"] for row in task_records]
    guard_score_gains = [row["guard_minus_score_oriented_gain"] for row in task_records]
    cost_score_gains = [row["unguarded_minus_score_oriented_gain"] for row in task_records]
    guard_regressions = sum(value < -MATERIAL_THRESHOLD for value in guard_score_gains)
    cost_regressions = sum(value < -MATERIAL_THRESHOLD for value in cost_score_gains)
    added = sum(
        guard < -MATERIAL_THRESHOLD and cost >= -MATERIAL_THRESHOLD
        for guard, cost in zip(guard_score_gains, cost_score_gains)
    )
    guard_seconds = sum(row["summaries"]["metric_risk_guard"]["elapsed_seconds"] for row in task_records)
    score_seconds = sum(row["summaries"]["score"]["elapsed_seconds"] for row in task_records)
    cost_ratio = guard_seconds / score_seconds
    primary = _paired(guard_gains)
    gate = {
        "positive_mean_guard_minus_unguarded": primary["mean"] > 0,
        "material_regressions_removed_vs_score": cost_regressions - guard_regressions,
        "minimum_material_regressions_removed_met": cost_regressions - guard_regressions >= 1,
        "material_regressions_added_vs_score": added,
        "maximum_material_regressions_added_met": added == 0,
        "aggregate_guard_cost_ratio_vs_score": cost_ratio,
        "maximum_cost_ratio_met": cost_ratio <= 0.35,
    }
    gate["passed"] = all(
        gate[key]
        for key in (
            "positive_mean_guard_minus_unguarded",
            "minimum_material_regressions_removed_met",
            "maximum_material_regressions_added_met",
            "maximum_cost_ratio_met",
        )
    )
    significance = (
        primary["wilcoxon_p_value"] < 0.05
        and primary["bootstrap_mean_95_ci"][0] > 0
    )
    return {
        "version": "v2.7",
        "protocol": "candidate choice fixed on development seeds 1-3; first evaluation on fresh TDC default scaffold validation seeds 4-5",
        "task_count": 22,
        "fresh_validation_seeds": list(FRESH_SEEDS),
        "official_test_loaded": False,
        "official_results_used": False,
        "leaderboard_used": False,
        "selection_or_official_freeze_changed": False,
        "preregistration": {
            "path": preregistration.relative_to(root).as_posix(),
            "sha256": _sha256(preregistration),
        },
        "identity_overlap_total": sum(
            sum(row["identity_overlap"].values()) for row in task_records
        ),
        "primary_guard_minus_unguarded": primary,
        "guard_minus_score": _paired(guard_score_gains),
        "unguarded_minus_score": _paired(cost_score_gains),
        "unguarded_material_regressions_vs_score": cost_regressions,
        "guard_material_regressions_vs_score": guard_regressions,
        "claim_gate": gate,
        "significance_gate_passed": significance,
        "tasks": task_records,
    }


def markdown(report: dict[str, Any]) -> str:
    primary = report["primary_guard_minus_unguarded"]
    gate = report["claim_gate"]
    lines = [
        "# Fresh-seed metric-risk guard confirmation",
        "",
        report["protocol"] + ". Official test data, official results, and leaderboard values were not loaded.",
        "",
        "| Task | Metric | Guard | Guard-cost gain | Guard-score gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(report["tasks"], key=lambda value: value["task"].lower()):
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['guard_policy']} | "
            f"{row['guard_minus_unguarded_oriented_gain']:.4f} | "
            f"{row['guard_minus_score_oriented_gain']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            f"- Guard minus unguarded mean: {primary['mean']:.6f}; 95% task-bootstrap CI [{primary['bootstrap_mean_95_ci'][0]:.6f}, {primary['bootstrap_mean_95_ci'][1]:.6f}].",
            f"- Improved/tied/worse: {primary['improved']}/{primary['tied']}/{primary['worse']}; Wilcoxon p={primary['wilcoxon_p_value']:.4f}.",
            f"- Material regressions versus score, unguarded/guarded: {report['unguarded_material_regressions_vs_score']}/{report['guard_material_regressions_vs_score']}.",
            f"- Aggregate guarded cost ratio versus score: {gate['aggregate_guard_cost_ratio_vs_score']:.3f}.",
            f"- Claim gate: {'PASS' if gate['passed'] else 'FAIL'}; significance gate: {'PASS' if report['significance_gate_passed'] else 'FAIL'}.",
            "",
            "A passed claim gate supports only a bounded fresh-scaffold-validation safety result. A significance claim additionally requires the separate significance gate. The experiment cannot modify the v2.7 official freeze.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--search-root", type=Path, default=Path("runs/leaderboard-v21"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/v27-router-risk-guard"))
    args = parser.parse_args()
    report = build_report(args.repo_root, args.search_root, args.data_dir, args.output_root)
    _write_json(args.output_root / "report.json", report)
    (args.output_root / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"claim_gate": report["claim_gate"]["passed"], "mean": report["primary_guard_minus_unguarded"]["mean"], "cost_ratio": report["claim_gate"]["aggregate_guard_cost_ratio_vs_score"]}, sort_keys=True))


if __name__ == "__main__":
    main()
