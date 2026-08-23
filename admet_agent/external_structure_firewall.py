"""Label-blind structural decontamination for external molecular datasets.

The planner never receives test labels or raw test structures.  A sealed
executor may use the test SMILES only to remove leakage-prone external records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from .data_governance import frame_sha256


@dataclass(frozen=True)
class StructureFirewallAudit:
    input_rows: int
    valid_external_rows: int
    invalid_external_smiles: int
    exact_train_overlap_rows: int
    exact_validation_overlap_rows: int
    exact_test_overlap_rows: int
    near_test_neighbor_rows: int
    test_molecule_count: int
    external_test_skeleton_coverage_count: int
    external_test_skeleton_coverage_fraction: float
    same_source_rejection_threshold: float
    tanimoto_rejection_threshold: float
    dataset_rejected: bool
    rejection_reason: str | None
    output_rows: int
    output_sha256: str
    test_structures_used_for_decontamination: bool = True
    test_labels_loaded: bool = False
    test_structures_exposed_to_planner: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_label_blind_test_structures(path: str | Path, *, smiles_col: str = "Drug") -> tuple[str, ...]:
    """Load only the public structure column from a benchmark test table."""

    source = Path(path).resolve()
    if source.name.lower() not in {"test.csv", "test.tsv"}:
        raise ValueError("Label-blind test loader accepts only an explicit test table")
    separator = "\t" if source.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(source, sep=separator, usecols=[smiles_col], dtype={smiles_col: "string"})
    return tuple(str(value) for value in frame[smiles_col].dropna())


@dataclass(frozen=True)
class _StandardMolecule:
    smiles: str
    inchikey: str
    skeleton: str


def _standard_molecule(smiles: str) -> _StandardMolecule | None:
    """Desalt, neutralise and canonicalise tautomers before InChI identity."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        smiles_out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not smiles_out or not inchikey or "-" not in inchikey:
        return None
    return _StandardMolecule(
        smiles=smiles_out,
        inchikey=inchikey,
        skeleton=inchikey.split("-", 1)[0],
    )


def _standard_unique(values: Sequence[str]) -> dict[str, _StandardMolecule]:
    output = {}
    for value in values:
        molecule = _standard_molecule(str(value))
        if molecule is not None:
            output[molecule.inchikey] = molecule
    return output


def apply_external_structure_firewall(
    external: pd.DataFrame,
    *,
    train_smiles: Sequence[str],
    validation_smiles: Sequence[str],
    test_smiles: Sequence[str],
    smiles_col: str = "Drug",
    same_source_test_overlap_limit: float = 0.05,
    test_tanimoto_limit: float = 0.90,
) -> tuple[pd.DataFrame, StructureFirewallAudit]:
    """Apply the three preregistered exclusions from the user-provided policy."""

    if smiles_col not in external:
        raise ValueError(f"Missing external SMILES column {smiles_col!r}")
    if not 0 <= same_source_test_overlap_limit <= 1:
        raise ValueError("same_source_test_overlap_limit must be in [0, 1]")
    if not 0 <= test_tanimoto_limit <= 1:
        raise ValueError("test_tanimoto_limit must be in [0, 1]")
    train = _standard_unique(train_smiles)
    validation = _standard_unique(validation_smiles)
    test = _standard_unique(test_smiles)
    if not test:
        raise ValueError("At least one valid label-blind test structure is required")

    working = external.copy().reset_index(drop=True)
    working["_standard_external"] = working[smiles_col].map(
        lambda value: _standard_molecule(str(value))
    )
    invalid = int(working["_standard_external"].isna().sum())
    valid = working[working["_standard_external"].notna()].copy()
    inchikeys = valid["_standard_external"].map(lambda value: value.inchikey)
    exact_train = inchikeys.isin(train)
    exact_validation = inchikeys.isin(validation)
    exact_test = inchikeys.isin(test)

    external_skeletons = {
        value.skeleton for value in valid["_standard_external"]
    }
    covered_test_molecules = sum(
        molecule.skeleton in external_skeletons for molecule in test.values()
    )
    coverage = covered_test_molecules / len(test)

    eligible_for_similarity = valid[~(exact_train | exact_validation | exact_test)].copy()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    test_fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(value.smiles))
        for value in test.values()
    ]
    near_mask = []
    for value in eligible_for_similarity["_standard_external"]:
        fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(value.smiles))
        maximum = max(DataStructs.BulkTanimotoSimilarity(fingerprint, test_fingerprints))
        near_mask.append(maximum >= test_tanimoto_limit)
    near_mask_array = np.asarray(near_mask, dtype=bool)

    rejected = coverage > same_source_test_overlap_limit
    reason = "external_dataset_has_more_than_allowed_test_inchikey_skeleton_overlap" if rejected else None
    if rejected:
        output = external.iloc[0:0].copy()
    else:
        output = eligible_for_similarity[~near_mask_array].copy()
        output[smiles_col] = output["_standard_external"].map(lambda value: value.smiles)
        output = output.drop(columns=["_standard_external"]).reset_index(drop=True)
    audit = StructureFirewallAudit(
        input_rows=len(working),
        valid_external_rows=len(valid),
        invalid_external_smiles=invalid,
        exact_train_overlap_rows=int(exact_train.sum()),
        exact_validation_overlap_rows=int(exact_validation.sum()),
        exact_test_overlap_rows=int(exact_test.sum()),
        near_test_neighbor_rows=int(near_mask_array.sum()),
        test_molecule_count=len(test),
        external_test_skeleton_coverage_count=covered_test_molecules,
        external_test_skeleton_coverage_fraction=coverage,
        same_source_rejection_threshold=same_source_test_overlap_limit,
        tanimoto_rejection_threshold=test_tanimoto_limit,
        dataset_rejected=rejected,
        rejection_reason=reason,
        output_rows=len(output),
        output_sha256=frame_sha256(output),
    )
    return output, audit


__all__ = [
    "StructureFirewallAudit",
    "apply_external_structure_firewall",
    "load_label_blind_test_structures",
]
