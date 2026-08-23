"""Automatic external-source discovery with endpoint-semantic adjudication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from .endpoint_contracts import EndpointContract, endpoint_contract


@dataclass(frozen=True)
class ExternalDatasetCandidate:
    source_id: str
    endpoint_key: str
    property_name: str
    task_type: str
    label_rule: str
    species: str | None
    assay_system: str | None
    unit: str | None
    transform: str
    provenance: str
    license: str
    access_url: str
    source_family: str
    target_id: str | None = None
    source_overlap_risk: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EndpointJudgment:
    source_id: str
    decision: str
    hard_rejection_reasons: tuple[str, ...]
    bridge_requirements: tuple[str, ...]
    next_action: str

    def to_dict(self) -> dict:
        return asdict(self)


_ENDPOINT_ALIASES = {
    "hERG": {"herg", "herg_blockade", "kcnh2", "chembl240"},
}


def judge_endpoint(candidate: ExternalDatasetCandidate, contract: EndpointContract) -> EndpointJudgment:
    hard = []
    bridge = []
    aliases = _ENDPOINT_ALIASES.get(contract.tdc_name, {contract.endpoint_id.lower(), contract.tdc_name.lower()})
    observed_keys = {
        candidate.endpoint_key.lower(),
        candidate.property_name.lower().replace(" ", "_"),
        (candidate.target_id or "").lower(),
    }
    if not aliases.intersection(observed_keys):
        hard.append("endpoint_or_target_identity_mismatch")
    if candidate.task_type != contract.task_type:
        hard.append("task_type_mismatch")
    if contract.species and candidate.species and candidate.species.lower() != contract.species.lower():
        hard.append("species_mismatch")
    for field in ("label_rule", "transform", "provenance", "license", "access_url"):
        if not str(getattr(candidate, field)).strip():
            hard.append(f"missing_{field}")
    if not candidate.assay_system:
        hard.append("missing_assay_system")
    if contract.species and not candidate.species:
        hard.append("missing_species")
    if candidate.task_type == "regression" and not candidate.unit:
        hard.append("missing_regression_unit")
    if candidate.label_rule.strip().lower() in {"binary", "active", "inactive", "unspecified"}:
        hard.append("label_direction_or_threshold_unspecified")

    if not hard:
        if contract.assay_system and candidate.assay_system != contract.assay_system:
            bridge.append("assay_system_not_identical_to_tdc_contract")
        if contract.label_semantics not in candidate.label_rule:
            bridge.append("label_semantics_or_threshold_not_proven_identical")
        if candidate.source_family == contract.source_family:
            bridge.append("possible_same_source_lineage_requires_provenance_deduplication")
        if candidate.source_overlap_risk != "low":
            bridge.append("source_overlap_risk_requires_structure_firewall")

    if hard:
        decision = "reject"
        next_action = "quarantine_without_download_or_training"
    elif bridge:
        decision = "bridge_required"
        next_action = "run_structure_firewall_then_source_heldout_alignment_only"
    else:
        decision = "exact"
        next_action = "run_structure_firewall_then_preregistered_matched_data_experiment"
    return EndpointJudgment(
        source_id=candidate.source_id,
        decision=decision,
        hard_rejection_reasons=tuple(sorted(set(hard))),
        bridge_requirements=tuple(sorted(set(bridge))),
        next_action=next_action,
    )


class DataSourceConnector(Protocol):
    name: str

    def search(self, contract: EndpointContract) -> Sequence[ExternalDatasetCandidate]: ...


class ChEMBLConnector:
    name = "chembl"

    def search(self, contract: EndpointContract) -> Sequence[ExternalDatasetCandidate]:
        if contract.tdc_name != "hERG":
            return ()
        return (
            ExternalDatasetCandidate(
                source_id="chembl-current-CHEMBL240-IC50",
                endpoint_key="herg_blockade",
                property_name="hERG blockade",
                task_type="classification",
                label_rule="hERG blocker when human KCNH2 IC50 < 10000 nM",
                species="human",
                assay_system="functional/toxicity human KCNH2 IC50 assays",
                unit="nM",
                transform="standard_value_nm < 10000",
                provenance="ChEMBL activity records grouped by document_chembl_id",
                license="CC BY-SA 3.0 Unported",
                access_url="https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL240",
                source_family="ChEMBL",
                target_id="CHEMBL240",
                source_overlap_risk="medium",
            ),
        )


class TDCRelatedHergConnector:
    name = "tdc_related"

    def search(self, contract: EndpointContract) -> Sequence[ExternalDatasetCandidate]:
        if contract.tdc_name != "hERG":
            return ()
        common = {
            "endpoint_key": "herg_blockade",
            "property_name": "hERG blockade",
            "task_type": "classification",
            "species": "human",
            "unit": "uM",
            "license": "CC BY 4.0 as listed by TDC",
            "source_family": "TDC hERG",
            "source_overlap_risk": "high",
        }
        return (
            ExternalDatasetCandidate(
                **common,
                source_id="tdc-herg-karim",
                label_rule="hERG blocker when IC50 < 10 uM; non-blocker when IC50 >= 10 uM",
                assay_system="integrated DeepHIT/BindingDB/ChEMBL/literature hERG evidence",
                transform="published binary threshold at 10 uM",
                provenance="Karim et al. integrated hERG dataset",
                access_url="https://tdcommons.ai/single_pred_tasks/tox/",
            ),
            ExternalDatasetCandidate(
                **common,
                source_id="tdc-herg-central",
                label_rule="hERG blocker derived from hERG_at_10uM inhibition and IC50 < 10 uM",
                assay_system="hERGCentral heterogeneous channel interaction assays",
                transform="published 10 uM binary rule",
                provenance="hERGCentral records distributed through TDC",
                access_url="https://tdcommons.ai/single_pred_tasks/tox/",
            ),
        )


class ExternalDataScout:
    def __init__(self, connectors: Sequence[DataSourceConnector] | None = None):
        self.connectors = tuple(connectors or (ChEMBLConnector(), TDCRelatedHergConnector()))

    def discover(self, task_name: str) -> dict:
        contract = endpoint_contract(task_name)
        rows = []
        failures = []
        for connector in self.connectors:
            try:
                candidates = connector.search(contract)
            except Exception as error:  # connector failure must not abort other sources
                failures.append({"connector": connector.name, "error": type(error).__name__})
                continue
            for candidate in candidates:
                judgment = judge_endpoint(candidate, contract)
                rows.append({"candidate": candidate.to_dict(), "judgment": judgment.to_dict()})
        order = {"exact": 0, "bridge_required": 1, "reject": 2}
        rows.sort(key=lambda row: (order[row["judgment"]["decision"]], row["candidate"]["source_id"]))
        return {
            "task": contract.tdc_name,
            "contract": contract.to_dict(),
            "candidate_count": len(rows),
            "connector_failures": failures,
            "candidates": rows,
            "automatic_download_allowed": False,
            "reason": "endpoint judgment and structure firewall must pass before acquisition",
            "test_labels_loaded": False,
        }


__all__ = [
    "ChEMBLConnector",
    "DataSourceConnector",
    "EndpointJudgment",
    "ExternalDataScout",
    "ExternalDatasetCandidate",
    "TDCRelatedHergConnector",
    "judge_endpoint",
]
