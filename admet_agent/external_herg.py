"""Deterministic ChEMBL hERG curation and source-heldout isolation."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .data_governance import canonical_smiles, frame_sha256

HERG_TARGET = "CHEMBL240"
BLOCKER_THRESHOLD_NM = 10_000.0
HOLDOUT_SALT = "tdc22-herg-chembl-source-holdout-v1"


def _source_partition(document_id: str) -> str:
    digest = hashlib.sha256(f"{HOLDOUT_SALT}:{document_id}".encode()).digest()
    return "heldout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def curate_chembl_herg(
    records: Iterable[Mapping[str, Any]],
    *,
    tdc_train_smiles: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Filter exact human hERG IC50s and create molecule-disjoint document splits."""

    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for raw in records:
        counts["input_records"] += 1
        checks = (
            (raw.get("target_chembl_id") == HERG_TARGET, "wrong_target"),
            (raw.get("target_organism") == "Homo sapiens", "wrong_species"),
            (raw.get("standard_type") == "IC50", "wrong_type"),
            (raw.get("standard_units") == "nM", "wrong_unit"),
            (raw.get("standard_relation") == "=", "censored_or_approximate"),
            (raw.get("standard_flag") == 1, "nonstandard_record"),
            (raw.get("potential_duplicate") in {0, None}, "potential_duplicate"),
            (raw.get("data_validity_comment") in {None, ""}, "validity_flag"),
            (raw.get("assay_type") in {"F", "T"}, "nonfunctional_assay"),
        )
        failed = next((reason for passed, reason in checks if not passed), None)
        if failed:
            counts[f"rejected_{failed}"] += 1
            continue
        document = str(raw.get("document_chembl_id") or "").strip()
        smiles = canonical_smiles(raw.get("canonical_smiles"))
        try:
            value_nm = float(raw.get("standard_value"))
        except (TypeError, ValueError):
            value_nm = float("nan")
        if not document:
            counts["rejected_missing_document"] += 1
            continue
        if smiles is None:
            counts["rejected_invalid_smiles"] += 1
            continue
        if not np.isfinite(value_nm) or value_nm <= 0:
            counts["rejected_invalid_value"] += 1
            continue
        rows.append(
            {
                "Drug": smiles,
                "Y": int(value_nm < BLOCKER_THRESHOLD_NM),
                "standard_value_nm": value_nm,
                "document_chembl_id": document,
                "assay_chembl_id": str(raw.get("assay_chembl_id") or ""),
                "molecule_chembl_id": str(raw.get("molecule_chembl_id") or ""),
                "activity_id": int(raw["activity_id"]),
                "source_partition": _source_partition(document),
            }
        )
    accepted = pd.DataFrame(rows)
    if accepted.empty:
        raise ValueError("No ChEMBL hERG records survived the preregistered filters")

    partitions = accepted.groupby("Drug")["source_partition"].nunique()
    cross_partition = set(partitions[partitions > 1].index)
    counts["rejected_cross_partition_molecules"] = len(cross_partition)
    accepted = accepted[~accepted["Drug"].isin(cross_partition)].copy()

    tdc_overlap = set(accepted["Drug"]).intersection(tdc_train_smiles)
    counts["rejected_tdc_train_overlap_molecules"] = len(tdc_overlap)
    accepted = accepted[~accepted["Drug"].isin(tdc_overlap)].copy()

    label_counts = accepted.groupby(["source_partition", "Drug"])["Y"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    counts["rejected_conflicting_partition_molecules"] = len(conflicts)
    if conflicts:
        conflict_index = pd.MultiIndex.from_tuples(conflicts)
        accepted_index = pd.MultiIndex.from_frame(accepted[["source_partition", "Drug"]])
        accepted = accepted[~accepted_index.isin(conflict_index)].copy()

    accepted = accepted.sort_values(
        ["source_partition", "Drug", "standard_value_nm", "activity_id"]
    )
    molecule_rows = []
    for (partition, smiles), group in accepted.groupby(["source_partition", "Drug"], sort=True):
        molecule_rows.append(
            {
                "Drug": smiles,
                "Y": int(group["Y"].iloc[0]),
                "median_ic50_nm": float(group["standard_value_nm"].median()),
                "document_chembl_ids": ";".join(sorted(set(group["document_chembl_id"]))),
                "assay_count": int(group["assay_chembl_id"].nunique()),
                "measurement_count": int(len(group)),
                "source_partition": partition,
            }
        )
    molecules = pd.DataFrame(molecule_rows)
    train = molecules[molecules["source_partition"] == "train"].reset_index(drop=True)
    heldout = molecules[molecules["source_partition"] == "heldout"].reset_index(drop=True)
    if set(train["Drug"]).intersection(heldout["Drug"]):
        raise RuntimeError("Molecule leakage remains across source partitions")
    train_docs = set(";".join(train["document_chembl_ids"]).split(";"))
    heldout_docs = set(";".join(heldout["document_chembl_ids"]).split(";"))
    if train_docs.intersection(heldout_docs):
        raise RuntimeError("Document leakage remains across source partitions")

    audit = {
        "filter_counts": dict(sorted(counts.items())),
        "threshold_nm": BLOCKER_THRESHOLD_NM,
        "partition_salt_sha256": hashlib.sha256(HOLDOUT_SALT.encode()).hexdigest(),
        "external_train_rows": len(train),
        "source_heldout_rows": len(heldout),
        "external_train_documents": len(train_docs - {""}),
        "source_heldout_documents": len(heldout_docs - {""}),
        "external_train_positive_rate": float(train["Y"].mean()) if len(train) else None,
        "source_heldout_positive_rate": float(heldout["Y"].mean()) if len(heldout) else None,
        "external_train_sha256": frame_sha256(train),
        "source_heldout_sha256": frame_sha256(heldout),
        "molecule_overlap_between_partitions": 0,
        "document_overlap_between_partitions": 0,
        "tdc_test_accessed_by_curator": False,
    }
    return train, heldout, audit


def admission_decision(audit: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    if int(audit["external_train_rows"]) < 500:
        reasons.append("external_train_has_fewer_than_500_molecules")
    if int(audit["source_heldout_rows"]) < 100:
        reasons.append("source_heldout_has_fewer_than_100_molecules")
    for key in ("external_train_positive_rate", "source_heldout_positive_rate"):
        value = audit[key]
        if value is None or not 0.05 <= float(value) <= 0.95:
            reasons.append(f"{key}_outside_0.05_to_0.95")
    if audit["molecule_overlap_between_partitions"] or audit["document_overlap_between_partitions"]:
        reasons.append("source_holdout_leakage")
    return {
        "admitted_for_matched_external_evaluation": not reasons,
        "reasons": reasons,
        "tdc_recipe_mutation_allowed": False,
    }


__all__ = [
    "BLOCKER_THRESHOLD_NM",
    "HERG_TARGET",
    "admission_decision",
    "curate_chembl_herg",
]
