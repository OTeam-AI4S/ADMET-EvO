#!/usr/bin/env python3
"""Reviewer-facing three-axis ADMET ablation with safe external-data admission.

Version 3 keeps the hidden-test firewall and confirmation-only routing while
requiring source-bound admission, confidence-capped external weights, and
matched random controls for small external-data coresets.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_all22_three_axis_ablation as core  # noqa: E402
from admet_agent.brain import RuleBasedBrain  # noqa: E402
from admet_agent.adaptive_external_data import (  # noqa: E402
    external_mass_capped_weights,
)
from admet_agent.data_axis_v2 import native_training_plan  # noqa: E402
from admet_agent.schemas import ExperimentSpec  # noqa: E402
from admet_agent.selection import build_model, predict  # noqa: E402
from admet_agent.tasks import resolve_task  # noqa: E402
from scripts.run_all22_optimal_recipe_adaptive_data import (  # noqa: E402
    _fit_with_sample_weight,
)


PROTOCOL_NAME = "all22-three-axis-agent-ablation-v3"
DEFAULT_OUTPUT = ROOT / "reports/all22_three_axis_ablation_v3"
DEFAULT_PROTOCOL = DEFAULT_OUTPUT / "protocol.json"
NO_DATA = {
    "enabled": False,
    "method": "official",
    "ratio": 0.0,
    "native_policy": "official",
    "weight_policy": "none",
}
RISK_PENALTY = 0.50


def _risk_adjusted(summary: dict[str, Any]) -> float:
    seeds = max(1, len(summary.get("seeds", [])))
    return float(summary["mean_normalized_improvement"]) - RISK_PENALTY * float(
        summary["sample_normalized_improvement_sd"]
    ) / math.sqrt(seeds)


def _best_arm_v2(
    rows: list[dict[str, Any]],
    configs: dict[str, Any],
    candidates: list[str],
) -> tuple[str, Any, dict[str, Any]]:
    """Select by a variance-penalized cross-seed utility, with simple tie-breaks."""

    summaries = core._summaries([row for row in rows if row["arm"] in candidates])
    best = max(
        candidates,
        key=lambda arm: (
            _risk_adjusted(summaries[arm]),
            summaries[arm]["positive_seed_fraction"],
            summaries[arm]["mean_normalized_improvement"],
            -len(json.dumps(configs[arm], sort_keys=True)),
            arm,
        ),
    )
    result = dict(summaries[best])
    result["risk_adjusted_utility"] = _risk_adjusted(result)
    result["selection_rule"] = "mean_minus_0.5_standard_error"
    return best, configs[best], result


def _paired_promotion(
    rows: list[dict[str, Any]],
    *,
    incumbent: str,
    challenger: str,
    threshold: float,
    positive_seed_fraction: float,
) -> tuple[bool, dict[str, Any]]:
    by_arm = {
        arm: {int(row["seed"]): float(row["normalized_improvement"]) for row in rows if row["arm"] == arm}
        for arm in (incumbent, challenger)
    }
    common = sorted(set(by_arm[incumbent]) & set(by_arm[challenger]))
    if not common:
        raise ValueError("Paired promotion requires common seeds")
    deltas = [by_arm[challenger][seed] - by_arm[incumbent][seed] for seed in common]
    mean = float(np.mean(deltas))
    sd = float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
    fraction = float(np.mean(np.asarray(deltas) > 0))
    lcb = mean - RISK_PENALTY * sd / math.sqrt(len(deltas))
    passed = mean >= threshold and fraction >= positive_seed_fraction and lcb > 0
    return passed, {
        "incumbent": incumbent,
        "challenger": challenger,
        "paired_mean_increment": mean,
        "paired_sample_sd": sd,
        "paired_positive_seed_fraction": fraction,
        "risk_adjusted_increment": lcb,
        "threshold": threshold,
        "required_positive_seed_fraction": positive_seed_fraction,
        "passed": passed,
        "seeds": common,
    }


class DataCentricEvaluator(core.Evaluator):
    """Evaluator that applies train-only curation and valuation before fitting."""

    def _single_prediction(
        self,
        member: dict[str, Any],
        representations: list[str],
        data_config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cache_key = core._signature(
            {"member": member, "features": representations, "data": data_config}
        )
        indices, external_audit = self._external_indices(data_config)
        native_positions, y_native, native_weights, native_audit = native_training_plan(
            self.train,
            task_type=self.task.task_type,
            native_policy=str(data_config.get("native_policy", "official")),
            weight_policy=str(data_config.get("weight_policy", "none")),
        )
        audit = {"native": native_audit, "external": external_audit}
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key], audit
        x_train, x_predict, x_external = self._features(
            representations, include_external=bool(len(indices))
        )
        x_fit = x_train[native_positions]
        y_fit = np.asarray(y_native)
        sample_weight = np.asarray(native_weights, dtype=float)
        if len(indices):
            x_fit = np.concatenate([x_fit, x_external[indices]], axis=0)
            y_external = self.external.iloc[indices]["Y"].to_numpy(dtype=y_fit.dtype)
            y_fit = np.concatenate([y_fit, y_external])
            if self.external_weight_policy is None:
                raise RuntimeError("External rows lack a verified weight policy")
            external_weights = external_mass_capped_weights(
                len(y_native),
                len(indices),
                external_mass_ratio=float(
                    self.external_weight_policy["external_mass_ratio"]
                ),
                external_row_weight_cap=float(
                    self.external_weight_policy["external_row_weight_cap"]
                ),
                external_to_min_native_weight_ratio_cap=float(
                    self.external_weight_policy[
                        "external_to_min_native_weight_ratio_cap"
                    ]
                ),
                native_sample_weights=native_weights,
            )
            sample_weight = np.concatenate(
                [sample_weight, external_weights[len(y_native) :]]
            )
            audit["external_weight_policy"] = self.external_weight_policy
            audit["external_total_weight"] = float(
                external_weights[len(y_native) :].sum()
            )
            audit["native_min_weight"] = float(native_weights.min())
            audit["external_max_weight"] = float(
                external_weights[len(y_native) :].max()
            )
            audit["native_strictly_dominates_external_per_row"] = bool(
                native_weights.min() > external_weights[len(y_native) :].max()
            )
        if np.allclose(sample_weight, 1.0):
            sample_weight_or_none = None
        else:
            sample_weight_or_none = sample_weight
        target_mean = 0.0
        target_scale = 1.0
        model_y = y_fit
        if self.task.task_type == "regression":
            target_mean = float(np.mean(y_fit))
            target_scale = float(np.std(y_fit))
            if not math.isfinite(target_scale) or target_scale <= 0:
                target_scale = 1.0
            model_y = (y_fit - target_mean) / target_scale
        experiment = ExperimentSpec.from_dict(
            {
                "task": self.task_name,
                "seed": self.seed,
                "representations": representations,
                "model": member["model"],
                "imbalance": member.get("imbalance", "none"),
                "data_policy": "official",
                "device": "cpu",
                "n_jobs": 1 if member["model"] == "svm_rbf" else self.n_jobs,
                "hyperparameters": member.get("hyperparameters", {}),
            }
        )
        model = build_model(experiment, self.task, np.asarray(model_y))
        fitted = _fit_with_sample_weight(
            model, x_fit, np.asarray(model_y), sample_weight_or_none
        )
        values = predict(fitted, x_predict, self.task)
        if self.task.task_type == "regression":
            values = values * target_scale + target_mean
        values = np.asarray(values, dtype=float)
        self._prediction_cache[cache_key] = values
        return values, audit


def _model_discovery_v2(
    evaluators: dict[int, core.Evaluator],
    baselines: dict[int, float],
    metric: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs: dict[str, Any] = {"M0": core.BASELINE_MODEL}
    task_name = next(iter(evaluators.values())).task_name
    task = resolve_task(task_name)
    n_train = len(next(iter(evaluators.values())).train)
    families = ["lightgbm", "xgboost", "extra_trees", "random_forest", "catboost"]
    if task.task_type == "classification":
        families.append("balanced_extra_trees")
    if n_train <= 5000:
        families.append("svm_rbf")
    default_arms = []
    for model in families:
        arm = f"M1_default_{model}"
        configs[arm] = core._model_single(model)
        default_arms.append(arm)
        rows.extend(
            core._evaluate_across(
                evaluators,
                arm=arm,
                model=configs[arm],
                feature=core.BASELINE_FEATURE,
                data=core.NO_DATA,
                baselines=baselines,
                metric=metric,
            )
        )
    m1_arm, m1, m1_summary = _best_arm_v2(rows, configs, default_arms)
    defaults = core._summaries(rows)
    top_families = [
        arm.removeprefix("M1_default_")
        for arm in sorted(default_arms, key=lambda a: _risk_adjusted(defaults[a]), reverse=True)[:4]
    ]
    tuned_arms = [m1_arm]
    for model in top_families:
        for variant in range(3):
            hyper = RuleBasedBrain._tuned(model, variant, task.task_type)
            imbalance_options = ["none"]
            if task.task_type == "classification" and model != "balanced_extra_trees":
                imbalance_options.append("class_weight")
            for imbalance in imbalance_options:
                arm = f"M2_{model}_v{variant}_{imbalance}"
                configs[arm] = core._model_single(model, hyper, imbalance)
                tuned_arms.append(arm)
                rows.extend(
                    core._evaluate_across(
                        evaluators,
                        arm=arm,
                        model=configs[arm],
                        feature=core.BASELINE_FEATURE,
                        data=core.NO_DATA,
                        baselines=baselines,
                        metric=metric,
                    )
                )
    m2_arm, m2, m2_summary = _best_arm_v2(rows, configs, tuned_arms)
    summaries = core._summaries(rows)
    best_by_family: dict[str, str] = {}
    for arm in tuned_arms:
        family = configs[arm]["model"]
        if family not in best_by_family or _risk_adjusted(summaries[arm]) > _risk_adjusted(
            summaries[best_by_family[family]]
        ):
            best_by_family[family] = arm
    ranked = sorted(
        best_by_family.values(),
        key=lambda arm: _risk_adjusted(summaries[arm]),
        reverse=True,
    )[:3]
    fusion_arms = [m2_arm]
    for left_index in range(len(ranked)):
        for right_index in range(left_index + 1, len(ranked)):
            left, right = ranked[left_index], ranked[right_index]
            for weight in (0.20, 0.35, 0.50, 0.65, 0.80):
                arm = f"M3_{configs[left]['model']}_{configs[right]['model']}_w{weight:.2f}"
                configs[arm] = {
                    "kind": "ensemble",
                    "members": [configs[left], configs[right]],
                    "weights": [weight, 1.0 - weight],
                    "aggregation": "probability_or_raw",
                }
                fusion_arms.append(arm)
                rows.extend(
                    core._evaluate_across(
                        evaluators,
                        arm=arm,
                        model=configs[arm],
                        feature=core.BASELINE_FEATURE,
                        data=core.NO_DATA,
                        baselines=baselines,
                        metric=metric,
                    )
                )
    if len(ranked) == 3:
        for weights in ([1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]):
            arm = "M3_three_" + "_".join(f"{value:.2f}" for value in weights)
            configs[arm] = {
                "kind": "ensemble",
                "members": [configs[value] for value in ranked],
                "weights": list(weights),
                "aggregation": "probability_or_raw",
            }
            fusion_arms.append(arm)
            rows.extend(
                core._evaluate_across(
                    evaluators,
                    arm=arm,
                    model=configs[arm],
                    feature=core.BASELINE_FEATURE,
                    data=core.NO_DATA,
                    baselines=baselines,
                    metric=metric,
                )
            )
    m3_arm, m3, m3_summary = _best_arm_v2(rows, configs, fusion_arms)
    selected = {
        "M1": {"source_arm": m1_arm, "config": m1, "development": m1_summary},
        "M2": {"source_arm": m2_arm, "config": m2, "development": m2_summary},
        "M3": {"source_arm": m3_arm, "config": m3, "development": m3_summary},
    }
    return rows, selected, {
        "top_default_families_tuned": top_families,
        "selection_rule": "variance_penalized_no_regret",
        "three_member_fusion_evaluated": len(ranked) == 3,
    }


def _feature_discovery_v2(
    evaluators: dict[int, core.Evaluator],
    baselines: dict[int, float],
    metric: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs: dict[str, list[str]] = {"F0": core.BASELINE_FEATURE}
    single_arms = []
    for index, view in enumerate(protocol["feature_axis"]["single_views"]):
        arm = f"F1_single_{index}"
        configs[arm] = list(view)
        single_arms.append(arm)
        rows.extend(core._evaluate_across(
            evaluators, arm=arm, model=core.BASELINE_MODEL,
            feature=configs[arm], data=core.NO_DATA,
            baselines=baselines, metric=metric,
        ))
    f1_arm, f1, f1_summary = _best_arm_v2(rows, configs, single_arms)
    single_summaries = core._summaries(rows)
    ranked_single = sorted(
        single_arms, key=lambda arm: _risk_adjusted(single_summaries[arm]), reverse=True
    )[:4]
    candidate_views = [list(view) for view in protocol["feature_axis"]["composed_views"]]
    for left in range(len(ranked_single)):
        for right in range(left + 1, len(ranked_single)):
            union = list(dict.fromkeys(configs[ranked_single[left]] + configs[ranked_single[right]]))
            candidate_views.append(union)
    composed_arms = [f1_arm]
    seen = {tuple(f1)}
    for index, view in enumerate(candidate_views):
        signature = tuple(view)
        if signature in seen:
            continue
        seen.add(signature)
        arm = f"F2_composed_{index}"
        configs[arm] = list(view)
        composed_arms.append(arm)
        rows.extend(core._evaluate_across(
            evaluators, arm=arm, model=core.BASELINE_MODEL,
            feature=configs[arm], data=core.NO_DATA,
            baselines=baselines, metric=metric,
        ))
    f2_arm, f2, f2_summary = _best_arm_v2(rows, configs, composed_arms)
    summaries = core._summaries(rows)
    ranked_composed = sorted(
        [arm for arm in composed_arms if arm != f1_arm],
        key=lambda arm: _risk_adjusted(summaries[arm]),
        reverse=True,
    )
    fusion_arms = [f2_arm]
    for count in (2, 3):
        if len(ranked_composed) < count:
            continue
        union: list[str] = []
        for arm in ranked_composed[:count] + [f1_arm]:
            for name in configs[arm]:
                if name not in union:
                    union.append(name)
        arm = f"F3_union_top{count}"
        configs[arm] = union
        fusion_arms.append(arm)
        rows.extend(core._evaluate_across(
            evaluators, arm=arm, model=core.BASELINE_MODEL,
            feature=union, data=core.NO_DATA,
            baselines=baselines, metric=metric,
        ))
    f3_arm, f3, f3_summary = _best_arm_v2(rows, configs, fusion_arms)
    return rows, {
        "F1": {"source_arm": f1_arm, "config": f1, "development": f1_summary},
        "F2": {"source_arm": f2_arm, "config": f2, "development": f2_summary},
        "F3": {"source_arm": f3_arm, "config": f3, "development": f3_summary},
    }


def _data_config(
    *,
    native_policy: str,
    weight_policy: str,
    enabled: bool = False,
    ratio: float = 0.0,
    rows: int | None = None,
    method: str = "official",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "enabled": enabled,
        "method": method,
        "ratio": float(ratio),
        "native_policy": native_policy,
        "weight_policy": weight_policy,
    }
    if rows is not None:
        value["rows"] = int(rows)
    return value


def _data_discovery_v2(
    evaluators: dict[int, core.Evaluator],
    baselines: dict[int, float],
    metric: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {
        "D1_official": core.NO_DATA,
        "D1_curated": _data_config(native_policy="curated", weight_policy="none"),
    }
    for arm in list(configs):
        rows.extend(core._evaluate_across(
            evaluators, arm=arm, model=core.BASELINE_MODEL,
            feature=core.BASELINE_FEATURE, data=configs[arm],
            baselines=baselines, metric=metric,
        ))
    d1_arm, d1, d1_summary = _best_arm_v2(rows, configs, list(configs))

    d2_arms = ["D2_inherit_D1"]
    configs[d2_arms[0]] = d1
    for native in ("official", "curated"):
        for weighting in ("scaffold_balance", "target_balance", "combined"):
            arm = f"D2_{native}_{weighting}"
            configs[arm] = _data_config(native_policy=native, weight_policy=weighting)
            d2_arms.append(arm)
    for arm in d2_arms:
        rows.extend(core._evaluate_across(
            evaluators, arm=arm, model=core.BASELINE_MODEL,
            feature=core.BASELINE_FEATURE, data=configs[arm],
            baselines=baselines, metric=metric,
        ))
    d2_challenger, _, _ = _best_arm_v2(rows, configs, d2_arms)
    if d2_challenger == "D2_inherit_D1":
        d2_arm, d2, d2_gate = d2_challenger, d1, {"passed": False, "reason": "incumbent_best"}
    else:
        passed, d2_gate = _paired_promotion(
            rows, incumbent="D2_inherit_D1", challenger=d2_challenger,
            threshold=float(protocol["selection"]["practical_threshold"]),
            positive_seed_fraction=float(protocol["selection"]["positive_seed_fraction"]),
        )
        d2_arm, d2 = (d2_challenger, configs[d2_challenger]) if passed else ("D2_inherit_D1", d1)
    d2_summary = core._summaries([row for row in rows if row["arm"] in d2_arms])[d2_arm]

    d3_arms = ["D3_inherit_D2"]
    configs[d3_arms[0]] = d2
    admitted = all(evaluator.external_admitted for evaluator in evaluators.values())
    if admitted:
        for method in ("coreset", "random"):
            for rows_budget in protocol["data_axis"]["external_row_budgets"]:
                arm = f"D3_{method}_n{rows_budget}"
                configs[arm] = _data_config(
                    native_policy=str(d2.get("native_policy", "official")),
                    weight_policy=str(d2.get("weight_policy", "none")),
                    enabled=True,
                    method=method,
                    rows=int(rows_budget),
                )
                d3_arms.append(arm)
            for ratio in protocol["data_axis"]["external_ratios"]:
                arm = f"D3_{method}_r{ratio:g}"
                configs[arm] = _data_config(
                    native_policy=str(d2.get("native_policy", "official")),
                    weight_policy=str(d2.get("weight_policy", "none")),
                    enabled=True,
                    ratio=float(ratio),
                    method=method,
                )
                d3_arms.append(arm)
    for arm in d3_arms:
        rows.extend(core._evaluate_across(
            evaluators, arm=arm, model=core.BASELINE_MODEL,
            feature=core.BASELINE_FEATURE, data=configs[arm],
            baselines=baselines, metric=metric,
        ))
    d3_challenger, _, _ = _best_arm_v2(rows, configs, d3_arms)
    if d3_challenger == "D3_inherit_D2":
        d3_arm, d3, d3_gate = d3_challenger, d2, {"passed": False, "reason": "incumbent_best_or_no_admitted_source"}
    else:
        passed, d3_gate = _paired_promotion(
            rows, incumbent="D3_inherit_D2", challenger=d3_challenger,
            threshold=float(protocol["selection"]["practical_threshold"]),
            positive_seed_fraction=float(protocol["selection"]["positive_seed_fraction"]),
        )
        d3_arm, d3 = (d3_challenger, configs[d3_challenger]) if passed else ("D3_inherit_D2", d2)
    d3_summary = core._summaries([row for row in rows if row["arm"] in d3_arms])[d3_arm]
    return rows, {
        "D1": {"source_arm": d1_arm, "config": d1, "development": d1_summary},
        "D2": {"source_arm": d2_arm, "config": d2, "development": d2_summary, "promotion": d2_gate},
        "D3": {"source_arm": d3_arm, "config": d3, "development": d3_summary, "promotion": d3_gate, "external_source_admitted": admitted},
    }


def _confirmation_route(payload: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    summaries = payload["factorial"]["summaries"]
    threshold = float(protocol["selection"]["confirmation_practical_threshold"])
    fraction_required = float(protocol["selection"]["positive_seed_fraction"])
    eligible = []
    audits = {}
    for cell in protocol["factorial"]["cells"]:
        summary = summaries[f"factorial_{cell}"]
        lcb = _risk_adjusted(summary)
        passed = cell == "000" or (
            summary["mean_normalized_improvement"] >= threshold
            and summary["positive_seed_fraction"] >= fraction_required
            and lcb > 0
        )
        audits[cell] = {"risk_adjusted_utility": lcb, "passed": passed, **summary}
        if passed:
            eligible.append(cell)
    selected = max(
        eligible,
        key=lambda cell: (
            audits[cell]["risk_adjusted_utility"] - 0.001 * cell.count("1"),
            audits[cell]["mean_normalized_improvement"],
            -cell.count("1"),
            cell,
        ),
    )
    return {
        "selected_cell": selected,
        "selection_data": "independent confirmation validation only",
        "selection_rule": "positive_seed_gate_then_mean_minus_0.5_standard_error",
        "cell_audit": audits,
        "development_summary": audits[selected],
    }


_CORE_RUN_CONFIRMATION = core._run_confirmation
_CORE_RUN_CERTIFICATION = core._run_certification


def _run_confirmation_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _CORE_RUN_CONFIRMATION(*args, **kwargs)
    protocol = args[2] if len(args) > 2 else kwargs["protocol"]
    payload["routing"] = _confirmation_route(payload, protocol)
    payload["safety"]["final_routing_uses_confirmation_validation"] = True
    return payload


def _run_certification_v2(
    args: Any,
    task: str,
    protocol: dict[str, Any],
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    external: Any,
    external_audit: dict[str, Any],
) -> dict[str, Any]:
    frozen = copy.deepcopy(discovery)
    frozen["routing"] = confirmation["routing"]
    payload = _CORE_RUN_CERTIFICATION(
        args, task, protocol, frozen, confirmation, external, external_audit
    )
    payload["routing"]["selection_data"] = "independent confirmation validation only"
    payload["routing"]["confirmation_cell_audit"] = confirmation["routing"]["cell_audit"]
    payload["safety"]["test_results_used_for_selection"] = False
    payload["safety"]["configuration_frozen_from_confirmation"] = True
    return payload


def main() -> None:
    core.PROTOCOL_NAME = PROTOCOL_NAME
    core.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.NO_DATA = NO_DATA
    core.Evaluator = DataCentricEvaluator
    core._best_arm = _best_arm_v2
    core._model_discovery = _model_discovery_v2
    core._feature_discovery = _feature_discovery_v2
    core._data_discovery = _data_discovery_v2
    core._run_confirmation = _run_confirmation_v2
    core._run_certification = _run_certification_v2
    core.main()


if __name__ == "__main__":
    main()
