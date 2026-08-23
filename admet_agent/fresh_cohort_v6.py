"""Deterministic helpers for the preregistered EC-VOI v6 fresh cohort."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from admet_agent.cohort_governance import collapse_endpoint_records
from admet_agent.external_structure_firewall import _standard_molecule
from scripts.build_openadmet_cyp_split_manifest import _chemistry_group


CAP_SALT = "ecvoi-v6-fresh-cohort-cap-v1"


def standardize_structure_column(
    frame: pd.DataFrame,
    *,
    smiles_column: str,
    cache: dict[str, tuple[str, str, str] | None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standardize each distinct raw string once without examining labels."""
    if smiles_column not in frame:
        raise ValueError(f"Missing structure column {smiles_column!r}")
    cache = {} if cache is None else cache
    raw_values = frame[smiles_column].astype(str)
    for raw in sorted(set(raw_values) - set(cache)):
        molecule = _standard_molecule(raw)
        if molecule is None:
            cache[raw] = None
            continue
        try:
            group = _chemistry_group(molecule.smiles)
        except Exception:
            cache[raw] = None
            continue
        cache[raw] = (molecule.smiles, molecule.inchikey, group)
    mapped = raw_values.map(cache)
    valid = mapped.notna()
    output = frame.loc[valid].copy()
    values = mapped.loc[valid]
    output["standard_smiles"] = values.map(lambda item: item[0])
    output["inchikey"] = values.map(lambda item: item[1])
    output["chemistry_group"] = values.map(lambda item: item[2])
    return output, {
        "input_rows": len(frame),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
    }


def cap_endpoint(frame: pd.DataFrame, *, task_type: str) -> pd.DataFrame:
    """Apply the label-bounded, identity-hash compute cap from the registry."""
    work = frame.copy()
    work["_rank"] = work["inchikey"].map(
        lambda value: hashlib.sha256(f"{CAP_SALT}|{value}".encode()).hexdigest()
    )
    if task_type == "regression":
        return work.sort_values("_rank", kind="stable").head(30000).drop(columns="_rank")
    if task_type != "classification":
        raise ValueError(f"Unsupported task type: {task_type}")
    counts = work["target"].value_counts()
    if len(counts) != 2:
        return work.iloc[0:0].drop(columns="_rank")
    minority = counts.sort_values(kind="stable").index[0]
    minority_rows = work.loc[work["target"] == minority]
    majority_rows = (
        work.loc[work["target"] != minority]
        .sort_values("_rank", kind="stable")
        .head(30000)
    )
    return (
        pd.concat([minority_rows, majority_rows], ignore_index=True)
        .drop(columns="_rank")
        .sort_values("inchikey", kind="stable")
        .reset_index(drop=True)
    )


def prepare_endpoint(
    standardized: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = standardized[
        ["inchikey", "standard_smiles", "chemistry_group", target_column]
    ].rename(columns={target_column: "target"})
    collapsed, audit = collapse_endpoint_records(selected, task_type=task_type)
    capped = cap_endpoint(collapsed, task_type=task_type)
    audit = {**audit, "post_cap_identities": len(capped)}
    return capped, audit


def endpoint_admission(
    frame: pd.DataFrame,
    *,
    task_type: str,
    valid_smiles_fraction: float,
) -> tuple[bool, list[str]]:
    reasons = []
    if valid_smiles_fraction < 0.95:
        reasons.append("valid_smiles_fraction_below_0.95")
    if frame["inchikey"].nunique() < 400:
        reasons.append("fewer_than_400_identities")
    values = pd.to_numeric(frame["target"], errors="coerce")
    if task_type == "classification":
        counts = values.value_counts()
        if len(counts) != 2 or int(counts.min()) < 40:
            reasons.append("classification_minority_below_40")
        if "outer_fold" in frame and len(counts) == 2:
            table = frame.groupby(["outer_fold", "target"]).size().unstack(fill_value=0)
            if len(table) != 5 or table.shape[1] != 2 or int(table.min().min()) < 5:
                reasons.append("outer_fold_class_count_below_5")
    elif task_type == "regression":
        if float(values.notna().mean()) < 0.95:
            reasons.append("regression_finite_fraction_below_0.95")
        if not np.isfinite(values.var()) or float(values.var()) <= 0.0:
            reasons.append("regression_zero_variance")
    else:
        reasons.append("unsupported_task_type")
    return not reasons, reasons


__all__ = [
    "CAP_SALT", "cap_endpoint", "endpoint_admission", "prepare_endpoint",
    "standardize_structure_column",
]
