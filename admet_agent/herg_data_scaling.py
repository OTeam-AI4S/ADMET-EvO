"""Fast, auditable helpers for the focused hERG external-data pilot.

The module deliberately keeps collection/curation separate from model fitting.
It never reads the TDC test table; callers provide discovery train/validation
structures and receive deterministic, content-addressable external cohorts.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from .external_structure_firewall import _standard_molecule

CAP_SALT = "herg-data-scaling-pilot-v1"


@dataclass(frozen=True)
class CurationAudit:
    input_rows: int
    selected_before_standardization: int
    valid_structure_rows: int
    invalid_structure_rows: int
    conflicting_identity_rows: int
    duplicate_identity_rows: int
    output_identities: int
    positive_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def deterministic_binary_cap(
    frame: pd.DataFrame,
    *,
    id_column: str,
    label_column: str,
    majority_cap: int,
) -> pd.DataFrame:
    """Keep every minority row and a stable hash-ranked majority sample."""

    if majority_cap < 1:
        raise ValueError("majority_cap must be positive")
    if id_column not in frame or label_column not in frame:
        raise ValueError("Binary cap columns are missing")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    work = frame.loc[labels.isin((0, 1))].copy()
    work[label_column] = labels.loc[work.index].astype(int)
    counts = work[label_column].value_counts()
    if len(counts) != 2:
        raise ValueError("Binary cap requires both classes")
    minority = counts.sort_values(kind="stable").index[0]
    minority_rows = work.loc[work[label_column] == minority]
    majority_rows = work.loc[work[label_column] != minority].copy()
    majority_rows["_rank"] = majority_rows[id_column].map(
        lambda value: hashlib.sha256(
            f"{CAP_SALT}|{value}".encode()
        ).hexdigest()
    )
    majority_rows = (
        majority_rows.sort_values("_rank", kind="stable")
        .head(majority_cap)
        .drop(columns="_rank")
    )
    return (
        pd.concat([minority_rows, majority_rows], ignore_index=True)
        .sort_values([label_column, id_column], kind="stable")
        .reset_index(drop=True)
    )


def curate_binary_structures(
    frame: pd.DataFrame,
    *,
    smiles_column: str,
    label_column: str,
    id_column: str,
    majority_cap: int,
) -> tuple[pd.DataFrame, CurationAudit]:
    """Cap, standardize and collapse an external binary molecular endpoint."""

    selected = deterministic_binary_cap(
        frame,
        id_column=id_column,
        label_column=label_column,
        majority_cap=majority_cap,
    )
    standardized = []
    invalid = 0
    for row in selected[[id_column, smiles_column, label_column]].itertuples(
        index=False, name=None
    ):
        record_id, raw_smiles, raw_label = row
        molecule = _standard_molecule(str(raw_smiles))
        if molecule is None:
            invalid += 1
            continue
        standardized.append(
            {
                "source_record_id": str(record_id),
                "standard_smiles": molecule.smiles,
                "standard_inchikey": molecule.inchikey,
                "Y": int(raw_label),
            }
        )
    work = pd.DataFrame(standardized)
    if work.empty:
        raise ValueError("No valid external structures survived standardization")
    label_counts = work.groupby("standard_inchikey")["Y"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    conflict_rows = int(work["standard_inchikey"].isin(conflicts).sum())
    work = work.loc[~work["standard_inchikey"].isin(conflicts)].copy()
    duplicate_rows = int(work.duplicated("standard_inchikey", keep=False).sum())
    work = (
        work.sort_values(["standard_inchikey", "source_record_id"], kind="stable")
        .drop_duplicates("standard_inchikey", keep="first")
        .reset_index(drop=True)
    )
    audit = CurationAudit(
        input_rows=len(frame),
        selected_before_standardization=len(selected),
        valid_structure_rows=len(standardized),
        invalid_structure_rows=invalid,
        conflicting_identity_rows=conflict_rows,
        duplicate_identity_rows=duplicate_rows,
        output_identities=len(work),
        positive_rate=float(work["Y"].mean()),
    )
    return work, audit


def standardize_smiles(values: list[str]) -> pd.DataFrame:
    """Return one row per valid input without consulting labels."""

    rows = []
    for index, value in enumerate(values):
        molecule = _standard_molecule(str(value))
        if molecule is not None:
            rows.append(
                {
                    "input_index": index,
                    "standard_smiles": molecule.smiles,
                    "standard_inchikey": molecule.inchikey,
                }
            )
    return pd.DataFrame(rows)


def morgan_fingerprints(smiles: list[str]) -> list:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError("Standardized SMILES unexpectedly failed RDKit parsing")
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def fingerprints_to_csr(fingerprints: list) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for row_index, fingerprint in enumerate(fingerprints):
        bits = list(fingerprint.GetOnBits())
        rows.extend([row_index] * len(bits))
        columns.extend(bits)
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(fingerprints), 2048),
        dtype=np.float32,
    )


def near_neighbor_keep_mask(
    candidate_fingerprints: list,
    protected_fingerprints: list,
    *,
    threshold: float = 0.80,
) -> np.ndarray:
    """Keep candidates below a fixed similarity threshold to protected rows."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be within (0, 1]")
    if not protected_fingerprints:
        return np.ones(len(candidate_fingerprints), dtype=bool)
    keep = np.ones(len(candidate_fingerprints), dtype=bool)
    for index, fingerprint in enumerate(candidate_fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, protected_fingerprints
        )
        keep[index] = max(similarities) < threshold
    return keep


__all__ = [
    "CAP_SALT",
    "CurationAudit",
    "curate_binary_structures",
    "deterministic_binary_cap",
    "fingerprints_to_csr",
    "morgan_fingerprints",
    "near_neighbor_keep_mask",
    "standardize_smiles",
]
