"""Cross-fitted diagnosis-conditioned action-effect policy for EC-VOI v4."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StateDiagnosis:
    feature_shift: float
    imbalance_shift: float

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "StateDiagnosis":
        audit = record["state_audit"]
        return cls(
            feature_shift=max(
                0.0,
                float(audit["median_max_validation_similarity_before"])
                - float(audit["median_max_validation_similarity_after"]),
            ),
            imbalance_shift=abs(
                float(audit["output_positive_rate"])
                - float(audit["input_positive_rate"])
            ),
        )


def _features(action: str, diagnosis: StateDiagnosis, *, enabled: bool) -> list[float]:
    feature = float(action == "feature_rich")
    imbalance = float(action == "imbalance_aware")
    if not enabled:
        return [feature, imbalance]
    return [
        feature,
        imbalance,
        feature * diagnosis.feature_shift,
        feature * diagnosis.imbalance_shift,
        imbalance * diagnosis.feature_shift,
        imbalance * diagnosis.imbalance_shift,
    ]


def _shuffle_map(keys: list[str], diagnoses: dict[str, StateDiagnosis], seed: int) -> dict[str, StateDiagnosis]:
    ordered = sorted(keys)
    generator = np.random.default_rng(seed)
    permuted = list(generator.permutation(ordered))
    return {key: diagnoses[source] for key, source in zip(ordered, permuted, strict=True)}


def _seed_for(endpoint: str, state: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{endpoint}|{state}".encode()).hexdigest()
    return int(digest[:8], 16)


def evaluate_leave_one_endpoint_out(
    rows: list[dict[str, Any]],
    *,
    diagnosis_enabled: bool = True,
    shuffled_diagnosis: bool = False,
    shuffle_seed: int = 20260823,
    harm_margin: float = 0.005,
    harm_probability_limit: float = 0.5,
    cost_penalty: float = 0.00025,
) -> list[dict[str, Any]]:
    """Select one action per held-out endpoint-state with a strict firewall."""

    actions = ("baseline", "feature_rich", "imbalance_aware")
    by_key: dict[str, list[dict[str, Any]]] = {}
    diagnoses: dict[str, StateDiagnosis] = {}
    for row in rows:
        key = f"{row['task']}|{row['state']}"
        by_key.setdefault(key, []).append(row)
        diagnoses.setdefault(key, StateDiagnosis.from_record(row))
    keys = sorted(by_key)
    shuffled = _shuffle_map(keys, diagnoses, shuffle_seed) if shuffled_diagnosis else diagnoses
    outputs = []
    for target_key in keys:
        endpoint, state = target_key.split("|", 1)
        train_rows = [row for row in rows if row["task"] != endpoint and row["action"] != "baseline"]
        if not train_rows:
            raise ValueError("LOEO policy requires non-target action effects")
        x_train = []
        y_train = []
        elapsed_by_action: dict[str, list[float]] = {action: [] for action in actions}
        for row in train_rows:
            key = f"{row['task']}|{row['state']}"
            x_train.append(
                _features(
                    row["action"],
                    shuffled[key],
                    enabled=diagnosis_enabled,
                )
            )
            y_train.append(float(row["oriented_effect"]))
            elapsed_by_action[row["action"]].append(float(row["elapsed_seconds"]))
        model = make_pipeline(StandardScaler(), BayesianRidge())
        model.fit(np.asarray(x_train), np.asarray(y_train))

        target_diagnosis = shuffled[target_key]
        predictions: dict[str, dict[str, float | bool]] = {
            "baseline": {
                "mean": 0.0,
                "std": 0.0,
                "harm_probability": 0.0,
                "cost_ratio": 0.0,
                "utility": 0.0,
                "eligible": True,
            }
        }
        median_baseline_cost = float(
            np.median(
                [
                    row["elapsed_seconds"]
                    for row in rows
                    if row["task"] != endpoint and row["action"] == "baseline"
                ]
            )
        )
        for action in actions[1:]:
            mean, std = model.predict(
                np.asarray([_features(action, target_diagnosis, enabled=diagnosis_enabled)]),
                return_std=True,
            )
            mean_value = float(mean[0])
            std_value = max(float(std[0]), 1e-9)
            harm = float(norm.cdf((-harm_margin - mean_value) / std_value))
            action_cost = float(np.median(elapsed_by_action[action]))
            cost_ratio = action_cost / max(median_baseline_cost, 1e-9)
            eligible = harm <= harm_probability_limit
            utility = mean_value - cost_penalty * cost_ratio if eligible else -math.inf
            predictions[action] = {
                "mean": mean_value,
                "std": std_value,
                "harm_probability": harm,
                "cost_ratio": cost_ratio,
                "utility": utility,
                "eligible": eligible,
            }
        selected = max(actions, key=lambda action: (float(predictions[action]["utility"]), action))
        actual = {
            action: float(
                np.mean([row["oriented_effect"] for row in by_key[target_key] if row["action"] == action])
            )
            for action in actions
        }
        best = max(actual, key=lambda action: (actual[action], action))
        selected_effect = actual[selected]
        outputs.append(
            {
                "task": endpoint,
                "state": state,
                "selected_action": selected,
                "best_action": best,
                "selected_oriented_effect": selected_effect,
                "best_oriented_effect": actual[best],
                "selection_regret": actual[best] - selected_effect,
                "harmful_intervention": selected_effect < -harm_margin,
                "target_endpoint_training_rows": 0,
                "diagnosis": {
                    "feature_shift": target_diagnosis.feature_shift,
                    "imbalance_shift": target_diagnosis.imbalance_shift,
                },
                "predictions": predictions,
                "actual_effects": actual,
            }
        )
    return outputs


def fixed_rule_results(rows: list[dict[str, Any]], harm_margin: float = 0.005) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["state"]), {}).setdefault(row["action"], []).append(
            float(row["oriented_effect"])
        )
    outputs = []
    for (task, state), action_rows in sorted(grouped.items()):
        action = "feature_rich" if state.startswith("domain_shift") else (
            "imbalance_aware" if state.startswith("imbalance") else "baseline"
        )
        effects = {name: float(np.mean(values)) for name, values in action_rows.items()}
        selected = effects[action]
        best = max(effects.values())
        outputs.append({
            "task": task,
            "state": state,
            "selected_action": action,
            "selected_oriented_effect": selected,
            "selection_regret": best - selected,
            "harmful_intervention": selected < -harm_margin,
        })
    return outputs


def seeded_random_results(rows: list[dict[str, Any]], seed: int = 20260823, harm_margin: float = 0.005) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["state"]), {}).setdefault(row["action"], []).append(
            float(row["oriented_effect"])
        )
    outputs = []
    for (task, state), action_rows in sorted(grouped.items()):
        effects = {name: float(np.mean(values)) for name, values in action_rows.items()}
        generator = np.random.default_rng(_seed_for(task, state, seed))
        action = sorted(effects)[int(generator.integers(0, len(effects)))]
        selected = effects[action]
        best = max(effects.values())
        outputs.append({
            "task": task,
            "state": state,
            "selected_action": action,
            "selected_oriented_effect": selected,
            "selection_regret": best - selected,
            "harmful_intervention": selected < -harm_margin,
        })
    return outputs


__all__ = [
    "StateDiagnosis",
    "evaluate_leave_one_endpoint_out",
    "fixed_rule_results",
    "seeded_random_results",
]
