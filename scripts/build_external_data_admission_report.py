"""Build the frozen, pre-model external-data admission report for ADMET-22."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from admet_agent.external_data_admission import evaluate_external_data_admission
from admet_agent.tasks import TASKS

PROTOCOL = "admet22-external-data-quality-admission-v2-two-tier"

# These reviews summarize endpoint/source inspection only. They were frozen
# while the all-22 model experiment was paused and do not use model outcomes.
REVIEWS = {
    "Caco2_Wang": ("pending_review", "pending", "transport direction and pH still require row-level enforcement"),
    "HIA_Hou": ("approved_direct", "shared_or_unresolved", "endpoint is aligned but 27.6% exact overlap requires document holdout"),
    "Pgp_Broccatelli": ("repair_required", "shared_or_unresolved", "inhibition, transport, substrate and ATPase records are mixed"),
    "Bioavailability_Ma": ("repair_required", "pending", "species are pooled; only 4.3% of rows are human"),
    "Lipophilicity_AstraZeneca": ("pending_review", "pending", "LogD pH and method are not yet aligned to the endpoint contract"),
    "Solubility_AqSolDB": ("repair_required", "pending", "pH/protocol heterogeneity and off-endpoint assay descriptions"),
    "BBB_Martins": ("bridge_only", "shared_or_unresolved", "B3DB contains substantial Martins/TDC lineage overlap"),
    "PPBR_AZ": ("approved_direct", "pending", "human PPB percentage is aligned; document lineage remains pending"),
    "VDss_Lombardo": ("bridge_only", "shared_or_unresolved", "43.3% exact overlap and sampled median similarity 1.0"),
    "CYP2C9_Veith": ("repair_required", "shared_or_unresolved", "benchmark-lineage overlap and interval censoring"),
    "CYP2D6_Veith": ("repair_required", "shared_or_unresolved", "benchmark-lineage overlap and interval censoring"),
    "CYP3A4_Veith": ("repair_required", "shared_or_unresolved", "benchmark-lineage overlap, species/assay mixing and censoring"),
    "CYP2C9_Substrate_CarbonMangels": ("repair_required", "shared_or_unresolved", "too few identities, conflicts, and absence-as-negative risk"),
    "CYP2D6_Substrate_CarbonMangels": ("repair_required", "shared_or_unresolved", "too few identities, conflicts, and absence-as-negative risk"),
    "CYP3A4_Substrate_CarbonMangels": ("repair_required", "shared_or_unresolved", "conflicts and source-lineage overlap require rebuilding"),
    "Half_Life_Obach": ("repair_required", "pending", "terminal, plasma, hydrolysis and metabolic-stability half-lives are mixed"),
    "Clearance_Hepatocyte_AZ": ("repair_required", "pending", "only 17 identities and two non-equivalent normalization units"),
    "Clearance_Microsome_AZ": ("approved_direct", "pending", "assay semantics align, but coverage and document lineage must pass"),
    "LD50_Zhu": ("repair_required", "pending", "species and administration routes are pooled"),
    "hERG": ("bridge_only", "pending", "fixed-dose hERGCentral label is not endpoint-equivalent to direct merge"),
    "AMES": ("repair_required", "pending", "Ames/Adames keyword false matches and high label conflict"),
    "DILI": ("bridge_only", "shared_or_unresolved", "DILIrank severity mapping is a bridge sensitivity analysis"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict:
    expected = {spec.name for spec in TASKS.values()}
    if set(REVIEWS) != expected:
        raise RuntimeError("Manual admission review must cover exactly the canonical 22 tasks")
    decisions = {}
    for task in sorted(expected):
        path = args.audit_dir / f"{task}.json"
        audit = json.loads(path.read_text())
        review_status, lineage_status, rationale = REVIEWS[task]
        decision = evaluate_external_data_admission(
            audit,
            endpoint_review_status=review_status,
            lineage_status=lineage_status,
        )
        decision["review_rationale"] = rationale
        decision["audit_path"] = str(path.resolve())
        decision["audit_sha256"] = _sha256(path)
        decisions[task] = decision
    admitted = [
        task
        for task, decision in decisions.items()
        if decision["permitted_for_direct_augmentation"]
    ]
    bridge_admitted = [
        task
        for task, decision in decisions.items()
        if decision["permitted_for_controlled_bridge_augmentation"]
    ]
    report = {
        "protocol": PROTOCOL,
        "policy_timing": (
            "strict-direct tier frozen before all22 fitting; controlled-bridge "
            "tier introduced after the hERG developmental pilot"
        ),
        "model_scores_consulted": False,
        "prior_herg_pilot_used_to_motivate_controlled_bridge_tier": True,
        "controlled_bridge_claim_tier": "developmental_only",
        "test_labels_loaded": False,
        "task_count": len(decisions),
        "admitted_tasks": admitted,
        "admitted_task_count": len(admitted),
        "controlled_bridge_admitted_tasks": bridge_admitted,
        "controlled_bridge_admitted_task_count": len(bridge_admitted),
        "fail_closed_task_count": len(decisions) - len(admitted) - len(bridge_admitted),
        "decisions": decisions,
        "rule": (
            "Direct augmentation requires the strict three-layer gate. Controlled "
            "bridge augmentation uses relaxed coverage thresholds but remains "
            "developmental and requires disjoint confirmation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("reports/classic_ml_external_data_all22_audit_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    payload = build(parse_args())
    print(
        json.dumps(
            {
                "task_count": payload["task_count"],
                "admitted_task_count": payload["admitted_task_count"],
                "fail_closed_task_count": payload["fail_closed_task_count"],
                "controlled_bridge_admitted_task_count": payload[
                    "controlled_bridge_admitted_task_count"
                ],
            },
            indent=2,
        )
    )
