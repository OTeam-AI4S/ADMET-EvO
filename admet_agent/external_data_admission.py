"""Fail-closed quality admission for task-conditioned external ADMET data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdmissionThresholds:
    min_classification_identities: int = 100
    min_regression_identities: int = 30
    max_invalid_structure_fraction: float = 0.01
    max_conflicting_row_fraction: float = 0.02
    max_exact_overlap_fraction: float = 0.10
    max_classification_exact_disagreement: float = 0.10
    max_regression_exact_difference_iqr: float = 0.25
    max_target_distribution_shift: float = 0.50
    min_fraction_nearest_tanimoto_at_least_0_4: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledBridgeThresholds:
    """Relaxed thresholds for developmental, explicitly non-direct data."""

    min_classification_identities: int = 100
    min_regression_identities: int = 30
    max_invalid_structure_fraction: float = 0.01
    max_conflicting_row_fraction: float = 0.02
    max_exact_overlap_fraction: float = 0.05
    max_classification_exact_disagreement: float = 0.10
    max_regression_exact_difference_iqr: float = 0.25
    max_target_distribution_shift: float = 0.25
    min_fraction_nearest_tanimoto_at_least_0_4: float = 0.025

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VALID_REVIEW_STATUSES = {
    "approved_direct",
    "pending_review",
    "repair_required",
    "bridge_only",
}
VALID_LINEAGE_STATUSES = {"cleared_independent", "pending", "shared_or_unresolved"}


@dataclass(frozen=True)
class ExternalWeightPolicy:
    """Conservative training influence allowed for an admitted source."""

    external_mass_ratio: float
    external_row_weight_cap: float
    external_to_min_native_weight_ratio_cap: float
    source_tier: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DIRECT_WEIGHT_POLICY = ExternalWeightPolicy(
    external_mass_ratio=0.20,
    external_row_weight_cap=0.50,
    external_to_min_native_weight_ratio_cap=0.50,
    source_tier="endpoint_matched_cleared_independent",
)
CLEARED_BRIDGE_WEIGHT_POLICY = ExternalWeightPolicy(
    external_mass_ratio=0.05,
    external_row_weight_cap=0.25,
    external_to_min_native_weight_ratio_cap=0.50,
    source_tier="controlled_bridge_cleared_independent",
)
PENDING_BRIDGE_WEIGHT_POLICY = ExternalWeightPolicy(
    external_mass_ratio=0.02,
    external_row_weight_cap=0.10,
    external_to_min_native_weight_ratio_cap=0.50,
    source_tier="controlled_bridge_pending_lineage",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_external_data_admission(
    audit: dict[str, Any],
    *,
    endpoint_review_status: str,
    lineage_status: str,
    thresholds: AdmissionThresholds | None = None,
    bridge_thresholds: ControlledBridgeThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen audit without consulting downstream model scores."""

    if endpoint_review_status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"Unknown endpoint review status: {endpoint_review_status}")
    if lineage_status not in VALID_LINEAGE_STATUSES:
        raise ValueError(f"Unknown lineage status: {lineage_status}")
    if audit.get("run_mode") != "data_audit_only":
        raise ValueError("Admission requires a data_audit_only artifact")
    if audit.get("safety", {}).get("tdc_test_loaded") is not False:
        raise ValueError("Admission audit must certify that TDC test data was not loaded")

    policy = thresholds or AdmissionThresholds()
    bridge_policy = bridge_thresholds or ControlledBridgeThresholds()
    curation = audit["curation"]
    target = audit["target_alignment"]
    space = audit["chemical_space_alignment"]
    task_type = audit["task_type"]

    output_identities = int(curation["output_identities"])
    valid_rows = max(1, int(curation.get("valid_structure_rows", output_identities)))
    selected_rows = max(
        1,
        int(
            curation.get(
                "selected_before_standardization",
                curation.get("numeric_label_rows", curation.get("raw_rows", valid_rows)),
            )
        ),
    )
    invalid_fraction = float(curation.get("invalid_structure_rows", 0) / selected_rows)
    conflicting_fraction = float(
        curation.get("conflicting_identity_rows", 0) / valid_rows
    )
    overlap_fraction = float(target["exact_overlap_identities"] / max(1, output_identities))
    target_shift = target.get(
        "absolute_positive_rate_shift", target.get("normalized_median_shift")
    )
    if task_type == "classification":
        exact_conflict = target.get("exact_overlap_label_disagreement_fraction")
        exact_limit = policy.max_classification_exact_disagreement
        minimum = policy.min_classification_identities
        bridge_exact_limit = bridge_policy.max_classification_exact_disagreement
        bridge_minimum = bridge_policy.min_classification_identities
    elif task_type == "regression":
        exact_conflict = target.get(
            "exact_overlap_normalized_median_absolute_target_difference"
        )
        exact_limit = policy.max_regression_exact_difference_iqr
        minimum = policy.min_regression_identities
        bridge_exact_limit = bridge_policy.max_regression_exact_difference_iqr
        bridge_minimum = bridge_policy.min_regression_identities
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    checks = {
        "minimum_identity_count": output_identities >= minimum,
        "valid_structure_fraction": invalid_fraction
        <= policy.max_invalid_structure_fraction,
        "within_source_label_consistency": conflicting_fraction
        <= policy.max_conflicting_row_fraction,
        "source_lineage_overlap": overlap_fraction
        <= policy.max_exact_overlap_fraction,
        "exact_overlap_target_consistency": exact_conflict is None
        or float(exact_conflict) <= exact_limit,
        "target_distribution_alignment": target_shift is not None
        and float(target_shift) <= policy.max_target_distribution_shift,
        "chemical_space_coverage": float(
            space["fraction_nearest_at_least_0_4"]
        )
        >= policy.min_fraction_nearest_tanimoto_at_least_0_4,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    automated_quality_passed = not failed_checks
    direct_permitted = bool(
        automated_quality_passed
        and endpoint_review_status == "approved_direct"
        and lineage_status == "cleared_independent"
    )
    bridge_checks = {
        "minimum_identity_count": output_identities >= bridge_minimum,
        "valid_structure_fraction": invalid_fraction
        <= bridge_policy.max_invalid_structure_fraction,
        "within_source_label_consistency": conflicting_fraction
        <= bridge_policy.max_conflicting_row_fraction,
        "source_lineage_overlap": overlap_fraction
        <= bridge_policy.max_exact_overlap_fraction,
        "exact_overlap_target_consistency": exact_conflict is None
        or float(exact_conflict) <= bridge_exact_limit,
        "target_distribution_alignment": target_shift is not None
        and float(target_shift) <= bridge_policy.max_target_distribution_shift,
        "chemical_space_coverage": float(
            space["fraction_nearest_at_least_0_4"]
        )
        >= bridge_policy.min_fraction_nearest_tanimoto_at_least_0_4,
    }
    bridge_failed_checks = [
        name for name, passed in bridge_checks.items() if not passed
    ]
    bridge_permitted = bool(
        not bridge_failed_checks
        and endpoint_review_status == "bridge_only"
        and lineage_status in {"pending", "cleared_independent"}
    )
    return {
        "task": audit["task"],
        "task_type": task_type,
        "thresholds": policy.to_dict(),
        "controlled_bridge_thresholds": bridge_policy.to_dict(),
        "observed": {
            "output_identities": output_identities,
            "invalid_structure_fraction": invalid_fraction,
            "conflicting_row_fraction": conflicting_fraction,
            "exact_overlap_fraction": overlap_fraction,
            "exact_overlap_target_conflict": exact_conflict,
            "target_distribution_shift": target_shift,
            "fraction_nearest_tanimoto_at_least_0_4": space[
                "fraction_nearest_at_least_0_4"
            ],
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "automated_quality_passed": automated_quality_passed,
        "controlled_bridge_checks": bridge_checks,
        "controlled_bridge_failed_checks": bridge_failed_checks,
        "controlled_bridge_quality_passed": not bridge_failed_checks,
        "endpoint_review_status": endpoint_review_status,
        "lineage_status": lineage_status,
        "permitted_for_direct_augmentation": direct_permitted,
        "permitted_for_controlled_bridge_augmentation": bridge_permitted,
        "decision": (
            "admit_direct"
            if direct_permitted
            else "admit_controlled_bridge"
            if bridge_permitted
            else "fail_closed"
        ),
        "model_scores_consulted": False,
    }


def require_task_admission(
    report_path: Path, task: str, *, augmentation_mode: str = "direct"
) -> dict[str, Any]:
    """Return an admitted decision or fail before any external training rows load."""

    if not report_path.is_file():
        raise RuntimeError(f"External-data admission report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("model_scores_consulted") is not False:
        raise RuntimeError("Admission report must be independent of model scores")
    if report.get("test_labels_loaded") is not False:
        raise RuntimeError("Admission report must not use TDC test labels")
    decision = report.get("decisions", {}).get(task)
    if decision is None:
        raise RuntimeError(f"No external-data admission decision exists for {task}")
    if augmentation_mode == "auto":
        if decision.get("permitted_for_direct_augmentation") is True:
            augmentation_mode = "direct"
        elif decision.get("permitted_for_controlled_bridge_augmentation") is True:
            augmentation_mode = "controlled_bridge"
        else:
            direct_reasons = list(decision.get("failed_checks", []))
            bridge_reasons = list(
                decision.get("controlled_bridge_failed_checks", [])
            )
            raise RuntimeError(
                f"External augmentation for {task} failed closed in every mode: "
                f"direct={direct_reasons or ['not_permitted']}; "
                f"controlled_bridge={bridge_reasons or ['not_permitted']}; "
                f"endpoint_review={decision.get('endpoint_review_status')}; "
                f"lineage={decision.get('lineage_status')}"
            )
    if augmentation_mode == "direct":
        permission_key = "permitted_for_direct_augmentation"
        failures_key = "failed_checks"
        expected_endpoint_review = "approved_direct"
        lineage_allowed = {"cleared_independent"}
    elif augmentation_mode == "controlled_bridge":
        permission_key = "permitted_for_controlled_bridge_augmentation"
        failures_key = "controlled_bridge_failed_checks"
        expected_endpoint_review = "bridge_only"
        lineage_allowed = {"pending", "cleared_independent"}
    else:
        raise ValueError(f"Unknown augmentation mode: {augmentation_mode}")
    if decision.get(permission_key) is not True:
        reasons = list(decision.get(failures_key, []))
        if decision.get("endpoint_review_status") != expected_endpoint_review:
            reasons.append(f"endpoint_review={decision.get('endpoint_review_status')}")
        if decision.get("lineage_status") not in lineage_allowed:
            reasons.append(f"lineage={decision.get('lineage_status')}")
        raise RuntimeError(
            f"External {augmentation_mode} augmentation for {task} failed closed: "
            f"{', '.join(reasons)}"
        )
    admitted = dict(decision)
    admitted["resolved_augmentation_mode"] = augmentation_mode
    return admitted


def weight_policy_for_admission(
    decision: dict[str, Any],
) -> ExternalWeightPolicy:
    """Resolve conservative source influence from a verified admission."""

    mode = decision.get("resolved_augmentation_mode")
    if mode == "direct":
        if not (
            decision.get("permitted_for_direct_augmentation") is True
            and decision.get("endpoint_review_status") == "approved_direct"
            and decision.get("lineage_status") == "cleared_independent"
        ):
            raise RuntimeError("Direct weight policy requires cleared direct admission")
        return DIRECT_WEIGHT_POLICY
    if mode == "controlled_bridge":
        if not (
            decision.get("permitted_for_controlled_bridge_augmentation") is True
            and decision.get("endpoint_review_status") == "bridge_only"
        ):
            raise RuntimeError("Bridge weight policy requires controlled-bridge admission")
        if decision.get("lineage_status") == "cleared_independent":
            return CLEARED_BRIDGE_WEIGHT_POLICY
        if decision.get("lineage_status") == "pending":
            return PENDING_BRIDGE_WEIGHT_POLICY
        raise RuntimeError("Shared or unresolved lineage cannot receive training weight")
    raise RuntimeError("Admission decision has no resolved augmentation mode")


def require_bound_task_admission(
    report_path: Path,
    task: str,
    *,
    source_path: Path,
    source_name: str,
    augmentation_mode: str = "auto",
) -> dict[str, Any]:
    """Fail closed unless the exact source selected for fitting was audited."""

    report_path = report_path.resolve()
    decision = require_task_admission(
        report_path, task, augmentation_mode=augmentation_mode
    )
    audit_value = decision.get("audit_path")
    audit_digest = decision.get("audit_sha256")
    if not audit_value or not audit_digest:
        raise RuntimeError(f"Admission decision for {task} is not bound to an audit")
    audit_path = Path(str(audit_value))
    if not audit_path.is_absolute():
        audit_path = report_path.parent / audit_path
    audit_path = audit_path.resolve()
    if not audit_path.is_file():
        relocated_audit = (report_path.parent / audit_path.name).resolve()
        if relocated_audit.is_file():
            audit_path = relocated_audit
    if not audit_path.is_file():
        raise RuntimeError(f"Admission audit is missing for {task}: {audit_path}")
    if _sha256(audit_path) != str(audit_digest):
        raise RuntimeError(f"Admission audit SHA-256 mismatch for {task}")
    audit = json.loads(audit_path.read_text())
    if not (
        audit.get("task") == task
        and audit.get("run_mode") == "data_audit_only"
        and audit.get("safety", {}).get("tdc_test_loaded") is False
    ):
        raise RuntimeError(f"Invalid bound admission audit for {task}")
    audited_source = audit.get("source", {})
    resolved_source = source_path.resolve()
    if not resolved_source.is_file():
        raise RuntimeError(f"Admitted source is missing for {task}: {resolved_source}")
    if audited_source.get("name") != source_name:
        raise RuntimeError(f"Audited source name mismatch for {task}")
    audited_source_path = Path(str(audited_source.get("path", ""))).resolve()
    source_digest = _sha256(resolved_source)
    if audited_source.get("sha256") != source_digest:
        raise RuntimeError(f"Audited source SHA-256 mismatch for {task}")
    policy = weight_policy_for_admission(decision)
    return {
        **decision,
        "admission_report_path": str(report_path),
        "admission_report_sha256": _sha256(report_path),
        "audit_artifact_verified": True,
        "audit_path": str(audit_path),
        "audited_source_path": str(audited_source_path),
        "selected_source_path": str(resolved_source),
        "source_relocated_with_identical_sha256": (
            audited_source_path != resolved_source
        ),
        "audited_source_sha256": source_digest,
        "weight_policy": policy.to_dict(),
    }


__all__ = [
    "AdmissionThresholds",
    "ControlledBridgeThresholds",
    "ExternalWeightPolicy",
    "evaluate_external_data_admission",
    "require_bound_task_admission",
    "require_task_admission",
    "weight_policy_for_admission",
]
