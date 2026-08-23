"""Replay frozen v6 on the disclosed v5 failure cohort (development only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from admet_agent.ecvoi_v5_policy import EligibilityContract
from admet_agent.public_policy_analysis import endpoint_effect_table, select_policy
from admet_agent.public_policy_v6 import IdentifiabilityContract, select_policy_v6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    unbound = {key: value for key, value in payload.items() if key != "sha256"}
    expected = hashlib.sha256(
        (json.dumps(unbound, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if payload.get("sha256") != expected:
        raise RuntimeError("v6 mechanism protocol self-hash mismatch")
    for source, digest in payload["implementation_sha256"].items():
        if _sha256(Path(source)) != digest:
            raise RuntimeError(f"Frozen v6 implementation changed: {source}")
    return payload


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "endpoints": len(frame),
        "total_oriented_gain": float(frame["actual_effect"].sum()),
        "mean_oriented_gain": float(frame["actual_effect"].mean()),
        "mean_oracle_regret": float(frame["oracle_regret"].mean()),
        "harmful_interventions": int(frame["harmful_intervention"].sum()),
        "total_fit_cost": float(frame["realized_fit_cost"].sum()),
        "actions": frame["selected_action"].value_counts().sort_index().to_dict(),
    }


def main() -> None:
    protocol_path = Path("reports/ecvoi_v6_mechanism_protocol_v2.json")
    screen_path = Path("reports/ecvoi_v5_public_action_screen_seed17_v2.csv")
    protocol = _load_protocol(protocol_path)
    table = endpoint_effect_table(pd.read_csv(screen_path))
    primary = json.loads(
        Path("reports/ecvoi_v5_public_policy_analysis_protocol_v1.json").read_text()
    )["primary_endpoint_scope"]
    table = table.loc[table["endpoint_id"].isin(primary)].reset_index(drop=True)
    eligibility = EligibilityContract(**protocol["eligibility"])
    mechanism = protocol["mechanism"]
    identifiability = IdentifiabilityContract(
        min_training_sources=mechanism["minimum_training_sources"],
        min_source_win_fraction=mechanism["minimum_source_win_fraction"],
        min_median_loss_gain=mechanism["minimum_median_loss_gain"],
        ridge_alpha=4.0,
        full_weight_sources=mechanism["full_weight_sources"],
    )
    v6 = select_policy_v6(
        table,
        eligibility_contract=eligibility,
        identifiability_contract=identifiability,
        bootstraps=256,
        seed=20260823,
    )
    no_diagnosis = select_policy(
        table, mode="no_diagnosis", contract=eligibility,
        bootstraps=256, seed=20260823,
    )
    output_csv = Path("reports/ecvoi_v6_v5_development_replay_v1.csv")
    v6.to_csv(output_csv, index=False)
    v6_summary = _summary(v6)
    control_summary = _summary(no_diagnosis)
    report = {
        "version": "ecvoi-v6-v5-development-replay-v1",
        "mechanism_protocol_sha256": protocol["sha256"],
        "screen_result_sha256": _sha256(screen_path),
        "role": "disclosed development replay; not promotion or independent confirmation",
        "v6": v6_summary,
        "source_balanced_no_diagnosis": control_summary,
        "difference": {
            "total_gain": v6_summary["total_oriented_gain"] - control_summary["total_oriented_gain"],
            "mean_oracle_regret": v6_summary["mean_oracle_regret"] - control_summary["mean_oracle_regret"],
            "fit_cost": v6_summary["total_fit_cost"] - control_summary["total_fit_cost"],
        },
        "selected_diagnostics_identifiable": int(v6["diagnostics_identifiable"].sum()),
        "maximum_selected_shrinkage_weight": float(v6["diagnostic_shrinkage_weight"].max()),
        "fresh_confirmation_required": True,
        "promotion_claim_allowed": False,
        "test_or_leaderboard_feedback_used": False,
        "tdc_19_unlocked": False,
        "selection_output_sha256": _sha256(output_csv),
    }
    output = Path("reports/ecvoi_v6_v5_development_replay_v1.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
