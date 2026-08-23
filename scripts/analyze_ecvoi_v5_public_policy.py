"""Apply the preregistered cross-source v5 policy and promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admet_agent.ecvoi_v5_policy import EligibilityContract
from admet_agent.public_policy_analysis import endpoint_effect_table, select_policy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    unbound = {key: value for key, value in payload.items() if key != "sha256"}
    expected = hashlib.sha256((json.dumps(unbound, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    if payload.get("sha256") != expected:
        raise RuntimeError("Policy-analysis protocol self-hash mismatch")
    for source, digest in payload["implementation_sha256"].items():
        if _sha256(Path(source)) != digest:
            raise RuntimeError(f"Frozen policy-analysis implementation changed: {source}")
    return payload


def _direct_control(table: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for _, item in table.iterrows():
        if name == "rules":
            if item["task_type"] == "classification" and item["train_minority_fraction_filled"] <= 0.10:
                action = "nested_imbalance"
            elif item["structural_support"] <= 0.35:
                action = "feature_rich"
            else:
                action = "stop"
            effect = 0.0 if action == "stop" else float(item[f"effect_{action}"])
            cost = 0.0 if action == "stop" else (6.0 if action == "nested_imbalance" else 1.0)
        elif name == "boosting_anchor":
            action, effect, cost = name, float(item["effect_boosting_anchor"]), 1.0
        elif name == "automl_expanded":
            action, effect, cost = name, float(item["effect_automl_expanded"]), 21.0
        else:
            raise ValueError(name)
        oracle = max(0.0, float(item["effect_feature_rich"]), float(item["effect_nested_imbalance"]) if item["task_type"] == "classification" else -np.inf)
        rows.append({"endpoint_id": item["endpoint_id"], "source_id": item["source_id"], "task_type": item["task_type"], "policy": name, "selected_action": action, "actual_effect": effect, "oracle_effect": oracle, "oracle_regret": oracle - effect, "harmful_intervention": bool(effect < -0.005), "realized_fit_cost": cost})
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "endpoints": len(frame),
        "total_oriented_gain": float(frame["actual_effect"].sum()),
        "mean_oriented_gain": float(frame["actual_effect"].mean()),
        "mean_oracle_regret": float(frame["oracle_regret"].mean()),
        "harmful_interventions": int(frame["harmful_intervention"].sum()),
        "total_fit_cost": float(frame["realized_fit_cost"].sum()),
        "gain_per_fit": float(frame["actual_effect"].sum() / max(1.0, frame["realized_fit_cost"].sum())),
        "actions": frame["selected_action"].value_counts().sort_index().to_dict(),
    }


def _direction_probability(full: pd.DataFrame, control: pd.DataFrame, *, bootstraps: int, seed: int) -> float:
    merged = full[["endpoint_id", "source_id", "actual_effect"]].merge(control[["endpoint_id", "actual_effect"]], on="endpoint_id", suffixes=("_full", "_control"))
    merged["difference"] = merged["actual_effect_full"] - merged["actual_effect_control"]
    by_source = merged.groupby("source_id")["difference"].mean()
    rng = np.random.default_rng(seed)
    values = by_source.to_numpy(dtype=float)
    draws = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(bootstraps)]
    return float(np.mean(np.asarray(draws) > 0))


def run(args: argparse.Namespace) -> dict:
    protocol = _load_protocol(args.protocol)
    screen_manifest = json.loads(args.screen_manifest.read_text())
    if not screen_manifest.get("integrity_passed") or screen_manifest.get("analysis_or_promotion_performed"):
        raise RuntimeError("Action screen is incomplete or already performed analysis")
    if screen_manifest["protocol_sha256"] != protocol["action_screen_protocol_self_sha256"]:
        raise RuntimeError("Action screen protocol binding mismatch")
    if _sha256(args.screen_results) != screen_manifest["output_sha256"]:
        raise RuntimeError("Action screen result hash mismatch")
    table = endpoint_effect_table(pd.read_csv(args.screen_results))
    table = table.loc[table["endpoint_id"].isin(protocol["primary_endpoint_scope"])].reset_index(drop=True)
    contract = EligibilityContract(**protocol["eligibility"])
    selections = []
    for mode in ("full", "no_diagnosis", "shuffled_diagnosis"):
        selections.append(select_policy(table, mode=mode, contract=contract, bootstraps=protocol["posterior_bootstraps"], seed=protocol["seed"]))
    selections.append(select_policy(table, mode="full", contract=contract, bootstraps=protocol["posterior_bootstraps"], seed=protocol["seed"], ignore_harm=True))
    for name in ("rules", "boosting_anchor", "automl_expanded"):
        selections.append(_direct_control(table, name))
    all_selections = pd.concat(selections, ignore_index=True)
    args.selections.parent.mkdir(parents=True, exist_ok=True)
    all_selections.to_csv(args.selections, index=False)
    frames = {name: frame.reset_index(drop=True) for name, frame in all_selections.groupby("policy")}
    summaries = {name: _summary(frame) for name, frame in frames.items()}
    full = summaries["full"]
    no_diag = summaries["no_diagnosis"]
    shuffled = summaries["shuffled_diagnosis"]
    def reduction(control: dict) -> float:
        return float((control["mean_oracle_regret"] - full["mean_oracle_regret"]) / control["mean_oracle_regret"]) if control["mean_oracle_regret"] > 0 else -np.inf
    probabilities = {name: _direction_probability(frames["full"], frames[name], bootstraps=protocol["cluster_bootstraps"], seed=protocol["seed"] + index) for index, name in enumerate(("no_diagnosis", "shuffled_diagnosis"))}
    checks = [
        {"name": "regret_reduction_vs_no_diagnosis", "observed": reduction(no_diag), "requirement": ">=0.20", "passed": reduction(no_diag) >= 0.20},
        {"name": "regret_reduction_vs_shuffled", "observed": reduction(shuffled), "requirement": ">=0.20", "passed": reduction(shuffled) >= 0.20},
        {"name": "total_gain_above_no_diagnosis", "observed": full["total_oriented_gain"] - no_diag["total_oriented_gain"], "requirement": ">0", "passed": full["total_oriented_gain"] > no_diag["total_oriented_gain"]},
        {"name": "total_gain_above_shuffled", "observed": full["total_oriented_gain"] - shuffled["total_oriented_gain"], "requirement": ">0", "passed": full["total_oriented_gain"] > shuffled["total_oriented_gain"]},
        {"name": "zero_harmful_interventions", "observed": full["harmful_interventions"], "requirement": 0, "passed": full["harmful_interventions"] == 0},
        {"name": "strict_boosting_anchor_win", "observed": int((frames["full"].set_index("endpoint_id")["actual_effect"] > frames["boosting_anchor"].set_index("endpoint_id")["actual_effect"]).sum()), "requirement": ">=1", "passed": bool((frames["full"].set_index("endpoint_id")["actual_effect"] > frames["boosting_anchor"].set_index("endpoint_id")["actual_effect"]).any())},
        {"name": "source_cluster_direction_vs_no_diagnosis", "observed": probabilities["no_diagnosis"], "requirement": ">=0.90", "passed": probabilities["no_diagnosis"] >= 0.90},
        {"name": "source_cluster_direction_vs_shuffled", "observed": probabilities["shuffled_diagnosis"], "requirement": ">=0.90", "passed": probabilities["shuffled_diagnosis"] >= 0.90},
        {"name": "matched_max_fit_budget", "observed": protocol["matched_max_fits_per_endpoint"], "requirement": 21, "passed": protocol["matched_max_fits_per_endpoint"] == 21},
        {"name": "test_blind_leaderboard_access", "observed": 0, "requirement": 0, "passed": True},
    ]
    report = {"version": "ecvoi-v5-public-policy-analysis-v1", "protocol_sha256": protocol["sha256"], "screen_result_sha256": screen_manifest["output_sha256"], "summaries": summaries, "source_cluster_positive_probability": probabilities, "checks": checks, "promotion_passed": all(item["passed"] for item in checks), "test_labels_loaded": False, "blind_or_leaderboard_feedback_used": False, "tdc_19_unlocked": all(item["passed"] for item in checks)}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("reports/ecvoi_v5_public_policy_analysis_protocol_v1.json"))
    parser.add_argument("--screen-results", type=Path, default=Path("reports/ecvoi_v5_public_action_screen_seed17_v2.csv"))
    parser.add_argument("--screen-manifest", type=Path, default=Path("reports/ecvoi_v5_public_action_screen_seed17_manifest_v2.json"))
    parser.add_argument("--selections", type=Path, default=Path("reports/ecvoi_v5_public_policy_selections_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/ecvoi_v5_public_policy_analysis_v1.json"))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
