"""Governance utilities for independent, identity-locked molecular cohorts."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from admet_agent.external_structure_firewall import _standard_molecule
from scripts.build_openadmet_cyp_split_manifest import _chemistry_group


def standardize_records(frame: pd.DataFrame, *, smiles_column: str) -> tuple[pd.DataFrame, int]:
    """Standardize structures without interpreting labels."""
    if smiles_column not in frame:
        raise ValueError(f"Missing SMILES column {smiles_column!r}")
    working = frame.copy().reset_index(drop=True)
    working["_molecule"] = working[smiles_column].map(
        lambda value: _standard_molecule(str(value))
    )
    invalid = int(working["_molecule"].isna().sum())
    working = working.loc[working["_molecule"].notna()].copy()
    working["standard_smiles"] = working["_molecule"].map(lambda value: value.smiles)
    working["inchikey"] = working["_molecule"].map(lambda value: value.inchikey)
    working["chemistry_group"] = working["standard_smiles"].map(_chemistry_group)
    return working.drop(columns=["_molecule"]), invalid


def collapse_endpoint_records(
    frame: pd.DataFrame,
    *,
    task_type: str,
    target_column: str = "target",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse replicated identities and fail closed on classification conflicts."""
    required = {"inchikey", "standard_smiles", "chemistry_group", target_column}
    if missing := required - set(frame):
        raise ValueError(f"Missing endpoint columns: {sorted(missing)}")
    working = frame.copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    missing_target = int(working[target_column].isna().sum())
    working = working.loc[np.isfinite(working[target_column])].copy()
    conflicts: set[str] = set()
    if task_type == "classification":
        if set(working[target_column].unique()) - {0.0, 1.0}:
            raise ValueError("Classification labels must be binary")
        counts = working.groupby("inchikey")[target_column].nunique()
        conflicts = set(counts[counts > 1].index.astype(str))
        working = working.loc[~working["inchikey"].isin(conflicts)].copy()
    elif task_type != "regression":
        raise ValueError(f"Unsupported task type {task_type!r}")
    rows = []
    for identity, group in working.groupby("inchikey", sort=True):
        first = group.iloc[0].to_dict()
        first[target_column] = float(group[target_column].mean())
        first["replicate_count"] = len(group)
        rows.append(first)
    output = pd.DataFrame(rows)
    return output, {
        "input_rows": len(frame),
        "missing_or_nonfinite_targets": missing_target,
        "conflicting_identities_removed": len(conflicts),
        "replicate_rows_collapsed": int(len(working) - len(output)),
        "output_identities": len(output),
    }


def protected_structure_firewall(
    identities: pd.DataFrame,
    *,
    development_smiles: Sequence[str],
    protected_test_smiles: Sequence[str],
    tanimoto_threshold: float = 0.90,
) -> tuple[set[str], dict[str, Any]]:
    """Filter records without exposing protected structures or using counts for selection."""
    if not 0 <= tanimoto_threshold <= 1:
        raise ValueError("Tanimoto threshold must be in [0, 1]")
    required = {"inchikey", "standard_smiles"}
    if missing := required - set(identities):
        raise ValueError(f"Missing identity columns: {sorted(missing)}")
    development = {
        value.inchikey
        for raw in development_smiles
        if (value := _standard_molecule(str(raw))) is not None
    }
    protected = {
        value.inchikey: value
        for raw in protected_test_smiles
        if (value := _standard_molecule(str(raw))) is not None
    }
    if not protected:
        raise ValueError("Protected test structure set is empty")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    protected_fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(value.smiles))
        for value in protected.values()
    ]
    allowed: set[str] = set()
    exact_development = 0
    exact_test = 0
    near_test = 0
    for row in identities.drop_duplicates("inchikey").itertuples(index=False):
        identity = str(row.inchikey)
        if identity in development:
            exact_development += 1
            continue
        if identity in protected:
            exact_test += 1
            continue
        molecule = Chem.MolFromSmiles(str(row.standard_smiles))
        fingerprint = generator.GetFingerprint(molecule)
        maximum = max(
            DataStructs.BulkTanimotoSimilarity(fingerprint, protected_fingerprints)
        )
        if maximum >= tanimoto_threshold:
            near_test += 1
            continue
        allowed.add(identity)
    return allowed, {
        "candidate_identities": int(identities["inchikey"].nunique()),
        "protected_development_identities": len(development),
        "protected_test_identities": len(protected),
        "exact_development_overlap_removed": exact_development,
        "exact_test_overlap_removed": exact_test,
        "near_test_neighbor_removed": near_test,
        "tanimoto_threshold": tanimoto_threshold,
        "allowed_identities": len(allowed),
        "source_acceptance_depends_on_removal_counts": False,
        "test_labels_loaded": False,
        "protected_structures_exposed_to_policy": False,
    }


def assign_identity_locked_folds(
    identities: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 20260823,
) -> dict[str, int]:
    """Greedily balance indivisible chemistry groups across deterministic folds."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    mapping = identities[["inchikey", "chemistry_group"]].drop_duplicates()
    inconsistent = mapping.groupby("inchikey")["chemistry_group"].nunique()
    if (inconsistent != 1).any():
        raise ValueError("An identity maps to multiple chemistry groups")
    sizes = mapping.groupby("chemistry_group").size().to_dict()
    ordered = sorted(
        sizes,
        key=lambda group: (
            -int(sizes[group]),
            hashlib.sha256(f"v5-fold:{seed}:{group}".encode()).hexdigest(),
        ),
    )
    fold_sizes = [0] * folds
    group_fold = {}
    for group in ordered:
        fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
        group_fold[group] = fold
        fold_sizes[fold] += int(sizes[group])
    return {
        str(row.inchikey): int(group_fold[str(row.chemistry_group)])
        for row in mapping.itertuples(index=False)
    }


__all__ = [
    "assign_identity_locked_folds",
    "collapse_endpoint_records",
    "protected_structure_firewall",
    "standardize_records",
]
