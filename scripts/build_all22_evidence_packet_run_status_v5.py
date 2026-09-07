#!/usr/bin/env python3
"""Summarize all-22 data-axis execution without treating missing packets as zero trials."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packet_loop import CANONICAL_TASKS
from admet_agent.evidence_packet_multiseed import pooled_normalized_improvement
from admet_agent.evidence_packets import sha256_file

REPORT_ROOT = ROOT / "reports/evidence_packet_data_axis_v5"
OUTPUT = REPORT_ROOT / "all22_execution_status.json"

EXPERIMENTS = {
    "Caco2_Wang": (
        "caco2_peerj10_development.json",
        "caco2_peerj10_confirmation.json",
    ),
    "CYP3A4_Veith": (
        "cyp3a4_gonzalez2021_development.json",
        "cyp3a4_gonzalez2021_confirmation.json",
    ),
    "Pgp_Broccatelli": (
        "pgp_fda_inhibitors_v51_development.json",
        "pgp_fda_inhibitors_v51_confirmation.json",
    ),
    "Bioavailability_Ma": (
        "bioavailability_chembl4papers_v51_development.json",
        "bioavailability_chembl4papers_v51_confirmation.json",
    ),
    "CYP2C9_Substrate_CarbonMangels": (
        "cyp2c9_substrate_fda_v51_development.json",
        "cyp2c9_substrate_fda_v51_confirmation.json",
    ),
    "CYP2D6_Substrate_CarbonMangels": (
        "cyp2d6_substrate_fda_v51_development.json",
        "cyp2d6_substrate_fda_v51_confirmation.json",
    ),
    "CYP3A4_Substrate_CarbonMangels": (
        "cyp3a4_substrate_fda_v51_development.json",
        "cyp3a4_substrate_fda_v51_confirmation.json",
    ),
    "Half_Life_Obach": (
        "half_life_human_pk_3papers_v53_development.json",
        "half_life_human_pk_3papers_v53_confirmation.json",
    ),
    "Clearance_Hepatocyte_AZ": (
        "clearance_hepatocyte_jrc_v51_development.json",
        "clearance_hepatocyte_jrc_v51_confirmation.json",
    ),
    "hERG": ("herg_dorn2005_development.json", "herg_dorn2005_confirmation.json"),
    "DILI": (
        "dili_fda_dilirank2_new_strict_v51_development.json",
        "dili_fda_dilirank2_new_strict_v51_confirmation.json",
    ),
    "AMES": (
        "ames_chembl_papers_v51_development.json",
        "ames_chembl_papers_v51_confirmation.json",
    ),
    "Clearance_Microsome_AZ": (
        "clearance_microsome_granberg2024_v51_development.json",
        "clearance_microsome_granberg2024_v51_confirmation.json",
    ),
    "Lipophilicity_AstraZeneca": (
        "lipophilicity_lombardo2002_v51_development.json",
        "lipophilicity_lombardo2002_v51_confirmation.json",
    ),
    "Solubility_AqSolDB": (
        "solubility_granberg2024_v51_development.json",
        "solubility_granberg2024_v51_confirmation.json",
    ),
}

DEVELOPMENT_ONLY_EXPERIMENTS = {
    "PPBR_AZ": [
        "ppbr_sutherland2023_v51_development.json",
        "ppbr_sutherland2023_smit_drugcentral_v51_development.json",
        "ppbr_sutherland2023_drugcentral_v51_development.json",
        "ppbr_sutherland2023_smit_v51_development.json",
        "ppbr_jrc_v51_development.json",
    ]
}

REJECTED_CANDIDATES = {
    "VDss_Lombardo": {
        "packet_id": "korzekwa2017-vdss-table-s2",
        "reason": "all 63 candidate identities exactly overlap the native train/test name union",
    },
    "CYP2C9_Veith": {
        "packet_id": "gonzalez2021-cyp2c9-aid1645842",
        "reason": "test skeleton coverage 0.0570358 exceeded 0.05",
    },
    "CYP2D6_Veith": {
        "packet_id": "gonzalez2021-cyp2d6-aid1645840",
        "reason": "test skeleton coverage 0.0788668 exceeded 0.05",
    },
    "HIA_Hou": {
        "packet_id": "chembl37-plano2011-hia-values",
        "reason": "six near-100% values for newly synthesized antileishmanial compounds are computational/secondary ADME estimates, not direct human absorption assays",
        "audit_artifact": "reports/evidence_packet_data_axis_v5/remaining_hia_bbb_candidate_audit.json",
    },
    "BBB_Martins": {
        "packet_id": "b3db-r1-r50-aggregate",
        "reason": "B3DB rows retain secondary compilation IDs rather than molecule-level original direct in-vivo BBB assay locators",
        "audit_artifact": "reports/evidence_packet_data_axis_v5/remaining_hia_bbb_candidate_audit.json",
    },
    "LD50_Zhu": {
        "packet_id": "helman2019-rat-oral-ld50",
        "reason": "4,472 exact development overlaps, 1,112 exact test overlaps, and 70.8731% test-skeleton coverage establish same-source lineage",
        "audit_artifact": "reports/evidence_packet_data_axis_v5/ld50_helman2019_lineage_audit.json",
    },
}


def build() -> dict:
    registry_path = ROOT / "data/evidence_packets_v4/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    active_by_task: dict[str, list[str]] = {}
    for entry in registry.get("active_packets", []):
        active_by_task.setdefault(entry["task"], []).append(entry["packet_id"])
    rows = []
    for task in CANONICAL_TASKS:
        task_dir = ROOT / "data/admet_group" / task.lower()
        row = {
            "task": task,
            "native_train_val_present": (task_dir / "train_val.csv").is_file(),
            "native_test_present": (task_dir / "test.csv").is_file(),
            "active_incumbent_packet_ids": sorted(active_by_task.get(task, [])),
        }
        if task in EXPERIMENTS:
            development_path = REPORT_ROOT / EXPERIMENTS[task][0]
            confirmation_path = REPORT_ROOT / EXPERIMENTS[task][1]
            confirmation = json.loads(
                confirmation_path.read_text(encoding="utf-8")
            )
            summary = confirmation["confirmation_summary"]
            metric = confirmation["rows"][0]["metric"]
            pooled_mean, pooled_seed_values = pooled_normalized_improvement(
                metric,
                [item["baseline_score"] for item in confirmation["rows"]],
                [item["challenger_score"] for item in confirmation["rows"]],
            )
            pooled_positive_fraction = float((pooled_seed_values > 0).mean())
            pooled_passed = bool(
                pooled_mean > 0 and pooled_positive_fraction >= 2 / 3
            )
            if pooled_passed != summary["confirmation_passed"]:
                raise RuntimeError(
                    f"{task} pass/fail changes under pooled-score rescoring"
                )
            row.update(
                {
                    "external_data_axis_status": (
                        "confirmation_passed_active_incumbent"
                        if pooled_passed
                        else "confirmation_failed_rolled_back"
                    ),
                    "packet_ids_evaluated": confirmation["packet_ids"],
                    "development_selected_configuration": confirmation[
                        "selected_configuration"
                    ],
                    "development_artifact": str(
                        development_path.relative_to(ROOT)
                    ),
                    "development_artifact_sha256": sha256_file(development_path),
                    "confirmation_artifact": str(
                        confirmation_path.relative_to(ROOT)
                    ),
                    "confirmation_artifact_sha256": sha256_file(
                        confirmation_path
                    ),
                    "confirmation_mean_normalized_improvement": pooled_mean,
                    "confirmation_normalized_improvement_aggregation": (
                        "ratio_of_mean_scores"
                    ),
                    "confirmation_artifact_reported_mean_normalized_improvement": (
                        summary["mean_normalized_improvement"]
                    ),
                    "confirmation_positive_seed_fraction": pooled_positive_fraction,
                    "confirmation_passed": pooled_passed,
                }
            )
        elif task in DEVELOPMENT_ONLY_EXPERIMENTS:
            trials = []
            for filename in DEVELOPMENT_ONLY_EXPERIMENTS[task]:
                development_path = REPORT_ROOT / filename
                development = json.loads(
                    development_path.read_text(encoding="utf-8")
                )
                trials.append(
                    {
                        "packet_ids": development["packet_ids"],
                        "artifact": str(development_path.relative_to(ROOT)),
                        "artifact_sha256": sha256_file(development_path),
                        "selection": development["selection"],
                    }
                )
            row.update(
                {
                    "external_data_axis_status": (
                        "development_no_robust_positive_configuration_rolled_back"
                    ),
                    "development_trials": trials,
                    "confirmation_passed": False,
                }
            )
        elif task in REJECTED_CANDIDATES:
            row.update(
                {
                    "external_data_axis_status": "candidate_rejected_before_training",
                    "candidate_rejection": REJECTED_CANDIDATES[task],
                    "confirmation_passed": False,
                }
            )
        else:
            row.update(
                {
                    "external_data_axis_status": "acquisition_pending_no_admitted_packet",
                    "confirmation_passed": False,
                }
            )
        rows.append(row)
    evaluated_tasks = set(EXPERIMENTS) | set(DEVELOPMENT_ONLY_EXPERIMENTS)
    evaluated = [row for row in rows if row["task"] in evaluated_tasks]
    passed = [row for row in evaluated if row["confirmation_passed"]]
    return {
        "protocol": "admet-evidence-packet-all22-execution-status-v5.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": 22,
        "native_data_available_task_count": sum(
            row["native_train_val_present"] and row["native_test_present"]
            for row in rows
        ),
        "external_data_axis_evaluated_task_count": len(evaluated),
        "external_confirmation_passed_task_count": len(passed),
        "candidate_rejected_before_training_task_count": len(REJECTED_CANDIDATES),
        "acquisition_pending_task_count": sum(
            row["external_data_axis_status"]
            == "acquisition_pending_no_admitted_packet"
            for row in rows
        ),
        "all22_external_experiment_complete": len(evaluated) == 22,
        "all22_tasks_resolved": len(evaluated) + len(REJECTED_CANDIDATES) == 22,
        "why_not_run_missing_tasks": "A task without a direct, independent, firewall-admitted FDA/literature/regulatory-assay packet is not a zero-data trial and is not eligible for external-data training.",
        "active_incumbent_registry": str(registry_path.relative_to(ROOT)),
        "active_incumbent_registry_sha256": sha256_file(registry_path),
        "active_incumbent_packet_count": len(registry.get("active_packets", [])),
        "confirmed_all22_mean_increment_from_active_packets": sum(
            row["confirmation_mean_normalized_improvement"] for row in passed
        )
        / 22.0,
        "tasks": rows,
        "safety": {
            "best_three_seeds_used_for_primary_selection": False,
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
            "failed_trials_rolled_back": True,
        },
    }


def main() -> None:
    payload = build()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "evaluated": payload["external_data_axis_evaluated_task_count"],
                "passed": payload["external_confirmation_passed_task_count"],
                "pending": payload["acquisition_pending_task_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
