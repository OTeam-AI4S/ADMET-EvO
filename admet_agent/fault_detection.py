"""Scientific fault detection and benchmarkable remediation suggestions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .data_governance import SourceManifest
from .endpoint_contracts import EndpointContract
from .hardness import HardnessCard


@dataclass(frozen=True)
class FaultFinding:
    fault_type: str
    severity: float
    confidence: float
    evidence: dict[str, Any]
    remediation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaultReport:
    task: str
    findings: tuple[FaultFinding, ...]
    detector_version: str = "scientific-fault-detector-v1"

    @property
    def detected_types(self) -> tuple[str, ...]:
        return tuple(item.fault_type for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "detector_version": self.detector_version,
            "detected_types": list(self.detected_types),
            "findings": [item.to_dict() for item in self.findings],
        }


class ScientificFaultDetector:
    """Convert discovery evidence into explicit, actionable fault findings."""

    def detect(
        self,
        card: HardnessCard,
        contract: EndpointContract,
        *,
        source_manifest: SourceManifest | None = None,
        reported_metric_direction: str | None = None,
        source_manifest_required: bool = False,
    ) -> FaultReport:
        findings: list[FaultFinding] = []
        quality = card.dimensions["data_quality"]
        audit = quality.evidence
        invalid = int(audit.get("invalid_smiles", 0))
        missing = int(audit.get("missing_targets", 0))
        non_finite = int(audit.get("non_finite_targets", 0))
        if invalid:
            findings.append(FaultFinding(
                "invalid_molecular_structure", min(1.0, invalid / max(1, audit["input_rows"])), 1.0,
                {"invalid_smiles": invalid}, "quarantine invalid structures before featurization",
            ))
        if missing or non_finite:
            findings.append(FaultFinding(
                "invalid_target", min(1.0, (missing + non_finite) / max(1, audit["input_rows"])), 1.0,
                {"missing_targets": missing, "non_finite_targets": non_finite},
                "remove invalid targets and retain a row-level audit trail",
            ))
        conflicts = int(audit.get("conflicting_structures", 0))
        if conflicts:
            fault = (
                "conflicting_class_labels"
                if contract.task_type == "classification"
                else "high_variance_replicate_measurements"
            )
            findings.append(FaultFinding(
                fault, quality.severity, quality.confidence,
                {
                    "structures": conflicts,
                    "replicate_rows": audit.get("replicate_rows", 0),
                    "relative_dispersion": audit.get("replicate_target_dispersion"),
                },
                (
                    "exclude ambiguous classification structures in a controlled data ablation"
                    if contract.task_type == "classification"
                    else "compare robust replicate aggregation against unaggregated training"
                ),
            ))
        elif contract.task_type == "regression" and int(audit.get("replicate_rows", 0)):
            findings.append(FaultFinding(
                "replicate_measurements", quality.severity, quality.confidence,
                {
                    "replicate_rows": audit.get("replicate_rows", 0),
                    "relative_dispersion": audit.get("replicate_target_dispersion"),
                }, "evaluate median replicate aggregation as a data action",
            ))

        leakage = card.dimensions["split_leakage_risk"]
        overlap = int(leakage.evidence.get("exact_train_valid_overlap_rows", 0))
        if overlap:
            findings.append(FaultFinding(
                "exact_split_overlap", leakage.severity, 1.0,
                {"overlap_rows": overlap},
                "remove overlapping training structures and recompute validation evidence",
            ))
        imbalance = card.dimensions["imbalance"]
        if imbalance.severity >= 0.5:
            findings.append(FaultFinding(
                "extreme_class_imbalance", imbalance.severity, imbalance.confidence,
                imbalance.evidence, "evaluate class weighting and balanced ensembles under PR-AUC",
            ))
        shift = card.dimensions["domain_shift"]
        if shift.severity >= 0.65:
            findings.append(FaultFinding(
                "chemical_domain_shift", shift.severity, shift.confidence,
                shift.evidence,
                "evaluate multi-view/pretrained representations and report applicability domain",
            ))
        roughness = card.dimensions["local_roughness"]
        if roughness.severity >= 0.35:
            findings.append(FaultFinding(
                "local_label_roughness", roughness.severity, roughness.confidence,
                roughness.evidence,
                "evaluate local kernels, robust ensembles and a cliff-subset metric",
            ))

        if source_manifest is not None:
            for issue in source_manifest.compatibility_issues(contract):
                issue_lower = issue.lower()
                fault_type = (
                    "species_mismatch" if "species" in issue_lower
                    else "assay_mismatch" if "assay" in issue_lower
                    else "unit_or_transform_mismatch"
                )
                findings.append(FaultFinding(
                    fault_type, 1.0, 1.0, {"issue": issue},
                    "keep the source quarantined until a source-held-out alignment study passes",
                ))
        elif source_manifest_required:
            findings.append(FaultFinding(
                "provenance_missing",
                1.0,
                1.0,
                {"required_manifest": True},
                "quarantine the source until a complete provenance manifest is supplied",
            ))
        if (
            reported_metric_direction is not None
            and reported_metric_direction != contract.metric_direction
        ):
            findings.append(FaultFinding(
                "wrong_metric_direction",
                1.0,
                1.0,
                {
                    "reported": reported_metric_direction,
                    "required": contract.metric_direction,
                },
                "orient model selection and all deltas using the endpoint contract",
            ))
        findings.sort(key=lambda item: (-item.severity, item.fault_type))
        return FaultReport(card.task, tuple(findings))
