"""Structure-only firewall for hidden TDC test partitions.

This module deliberately reads only the molecular-structure column from a test
CSV.  Labels are never requested from pandas and are therefore unavailable to
the caller.  The resulting identity blocklist may be used to prevent an
external training source from containing exact hidden-test molecules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from rdkit import Chem

from admet_agent.external_structure_firewall import _standard_molecule


def load_structure_only_inchikey_blocklist(
    path: Path, *, smiles_column: str = "Drug"
) -> tuple[set[str], dict[str, object]]:
    """Load and standardize test structures without reading any label column."""

    frame = pd.read_csv(path, usecols=[smiles_column])
    if list(frame.columns) != [smiles_column]:
        raise RuntimeError("The test firewall loaded columns other than structure")

    identities: set[str] = set()
    fallback_count = 0
    invalid_count = 0
    for raw_value in frame[smiles_column]:
        raw_smiles = str(raw_value)
        molecule = _standard_molecule(raw_smiles)
        if molecule is not None and Chem.MolFromSmiles(molecule.smiles) is not None:
            identity = molecule.inchikey
        else:
            raw_molecule = Chem.MolFromSmiles(raw_smiles)
            if raw_molecule is None:
                invalid_count += 1
                continue
            identity = Chem.MolToInchiKey(raw_molecule)
            fallback_count += 1
        if identity:
            identities.add(identity)

    identity_digest = hashlib.sha256(
        ("\n".join(sorted(identities)) + "\n").encode("utf-8")
    ).hexdigest()
    audit: dict[str, object] = {
        "test_structure_path": str(path.resolve()),
        "test_structure_columns_loaded": [smiles_column],
        "test_label_columns_loaded": [],
        "test_structure_rows_loaded": len(frame),
        "test_structure_unique_inchikeys": len(identities),
        "test_structure_standardization_fallback_count": fallback_count,
        "test_structure_invalid_count": invalid_count,
        "test_structure_inchikey_blocklist_sha256": identity_digest,
    }
    return identities, audit


__all__ = ["load_structure_only_inchikey_blocklist"]
