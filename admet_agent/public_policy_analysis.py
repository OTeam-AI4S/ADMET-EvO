"""Cross-source, evidence-conditioned analysis for the EC-VOI v5 public cohort."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ecvoi_v5_policy import EligibilityContract, gate_action


STATE_FEATURES = (
    "task_is_classification",
    "log_train_rows",
    "train_minority_fraction_filled",
    "minority_fraction_missing",
    "train_group_density",
    "structural_support",
)
POLICIES = ("full", "no_diagnosis", "shuffled_diagnosis")


def endpoint_effect_table(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse outer folds only after verifying a complete baseline pairing."""
    required = {
        "endpoint_id", "source_id", "outer_fold", "action", "task_type",
        "oriented_score", "train_rows", "train_chemistry_groups",
        "train_minority_fraction", "median_sampled_validation_to_train_tanimoto",
    }
    if missing := required - set(results):
        raise ValueError(f"Missing action-screen columns: {sorted(missing)}")
    if results.duplicated(["endpoint_id", "outer_fold", "action"]).any():
        raise ValueError("Duplicate endpoint/fold/action result")
    pivot = results.pivot(index=["endpoint_id", "outer_fold"], columns="action", values="oriented_score")
    if pivot["baseline"].isna().any():
        raise ValueError("Every endpoint fold requires a baseline")
    metadata = results.drop_duplicates(["endpoint_id", "outer_fold"]).set_index(["endpoint_id", "outer_fold"])
    rows = []
    for endpoint_id, frame in pivot.groupby(level="endpoint_id", sort=True):
        meta = metadata.loc[endpoint_id]
        task_type = str(meta["task_type"].iloc[0])
        row: dict[str, Any] = {
            "endpoint_id": endpoint_id,
            "source_id": str(meta["source_id"].iloc[0]),
            "task_type": task_type,
            "task_is_classification": float(task_type == "classification"),
            "log_train_rows": float(np.log1p(meta["train_rows"].mean())),
            "train_minority_fraction_filled": float(meta["train_minority_fraction"].mean()) if task_type == "classification" else 0.0,
            "minority_fraction_missing": float(task_type != "classification"),
            "train_group_density": float((meta["train_chemistry_groups"] / meta["train_rows"]).mean()),
            "structural_support": float(meta["median_sampled_validation_to_train_tanimoto"].mean()),
            "baseline_score": float(frame["baseline"].mean()),
        }
        for action in ("feature_rich", "nested_imbalance", "boosting_anchor", "automl_expanded"):
            row[f"effect_{action}"] = (
                float((frame[action] - frame["baseline"]).mean())
                if action in frame and frame[action].notna().any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("endpoint_id", kind="stable").reset_index(drop=True)


def _seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()
    return int(digest[:8], 16)


def cross_source_posterior(
    table: pd.DataFrame,
    held_index: int,
    *,
    action: str,
    mode: str,
    bootstraps: int,
    seed: int,
    harm_margin: float,
) -> dict[str, float]:
    """Predict an action effect after removing the held endpoint's whole source."""
    if mode not in POLICIES:
        raise ValueError(f"Unknown policy mode: {mode}")
    held = table.iloc[held_index]
    effect_column = f"effect_{action}"
    train = table.loc[table["source_id"] != held["source_id"]].copy()
    train = train.loc[np.isfinite(train[effect_column])].reset_index(drop=True)
    if len(train) < 2 or train["source_id"].nunique() < 2:
        return {"mean": 0.0, "harm_probability": 1.0, "cost_ratio": 6.0 if action == "nested_imbalance" else 1.0, "information_gain": 0.0, "training_endpoints": len(train), "training_sources": int(train["source_id"].nunique())}
    rng = np.random.default_rng(_seed(seed, held["endpoint_id"], action, mode))
    sources = sorted(train["source_id"].unique())
    held_x = held[list(STATE_FEATURES)].to_numpy(dtype=float).reshape(1, -1)
    draws = []
    for _ in range(bootstraps):
        sampled_sources = rng.choice(sources, size=len(sources), replace=True)
        sampled = pd.concat([train.loc[train["source_id"] == source] for source in sampled_sources], ignore_index=True)
        y = sampled[effect_column].to_numpy(dtype=float)
        if mode == "no_diagnosis":
            fitted = np.full(len(y), y.mean(), dtype=float)
            prediction = float(y.mean())
        else:
            x = sampled[list(STATE_FEATURES)].to_numpy(dtype=float)
            if mode == "shuffled_diagnosis":
                x = x[rng.permutation(len(x))]
            model = make_pipeline(StandardScaler(), Ridge(alpha=4.0))
            model.fit(x, y)
            fitted = model.predict(x)
            prediction = float(model.predict(held_x)[0])
        residual_scale = max(0.01, float(np.sqrt(np.mean((y - fitted) ** 2))))
        draws.append(prediction + float(rng.normal(0.0, residual_scale)))
    values = np.asarray(draws, dtype=float)
    return {
        "mean": float(values.mean()),
        "harm_probability": float(np.mean(values < -harm_margin)),
        "cost_ratio": 6.0 if action == "nested_imbalance" else 1.0,
        "information_gain": 0.0,
        "training_endpoints": len(train),
        "training_sources": len(sources),
    }


def select_policy(
    table: pd.DataFrame,
    *,
    mode: str,
    contract: EligibilityContract,
    bootstraps: int,
    seed: int,
    ignore_harm: bool = False,
) -> pd.DataFrame:
    """Cross-fit one action per endpoint under a shared per-endpoint max budget."""
    rows = []
    for index, held in table.iterrows():
        candidates = {
            "stop": gate_action({"mean": 0.0, "harm_probability": 0.0, "cost_ratio": 0.0, "information_gain": 0.0}, contract=contract, stop=True)
        }
        for action in ("feature_rich", "nested_imbalance"):
            if action == "nested_imbalance" and held["task_type"] != "classification":
                continue
            posterior = cross_source_posterior(table, index, action=action, mode=mode, bootstraps=bootstraps, seed=seed, harm_margin=contract.harm_margin)
            if ignore_harm:
                posterior["harm_probability"] = 0.0
            candidates[action] = gate_action(posterior, contract=contract)
        eligible = [(value["utility"], -value["cost_ratio"], action) for action, value in candidates.items() if value["eligible"]]
        _, _, selected = max(eligible)
        actual = 0.0 if selected == "stop" else float(held[f"effect_{selected}"])
        oracle = max(0.0, float(held["effect_feature_rich"]), float(held["effect_nested_imbalance"]) if held["task_type"] == "classification" else -np.inf)
        rows.append({
            "endpoint_id": held["endpoint_id"], "source_id": held["source_id"], "task_type": held["task_type"],
            "policy": mode + ("_no_harm" if ignore_harm else ""), "selected_action": selected,
            "actual_effect": actual, "oracle_effect": oracle, "oracle_regret": oracle - actual,
            "harmful_intervention": bool(actual < -contract.harm_margin),
            "realized_fit_cost": float(candidates[selected]["cost_ratio"]),
            "predicted_mean": float(candidates[selected]["mean"]),
            "predicted_harm_probability": float(candidates[selected]["harm_probability"]),
        })
    return pd.DataFrame(rows)


__all__ = [
    "POLICIES", "STATE_FEATURES", "cross_source_posterior",
    "endpoint_effect_table", "select_policy",
]
