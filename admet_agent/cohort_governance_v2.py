"""Chemistry-group parse compatibility adapter for cohort governance v1."""

from __future__ import annotations

import pandas as pd

from admet_agent.external_structure_firewall import _standard_molecule
from scripts.build_openadmet_cyp_split_manifest import _chemistry_group


def standardize_records(
    frame: pd.DataFrame, *, smiles_column: str
) -> tuple[pd.DataFrame, int]:
    """Count structures that cannot survive both standardization stages as invalid."""
    if smiles_column not in frame:
        raise ValueError(f"Missing SMILES column {smiles_column!r}")
    rows = []
    invalid = 0
    for record in frame.to_dict("records"):
        molecule = _standard_molecule(str(record[smiles_column]))
        if molecule is None:
            invalid += 1
            continue
        try:
            chemistry_group = _chemistry_group(molecule.smiles)
        except Exception:
            invalid += 1
            continue
        rows.append(
            {
                **record,
                "standard_smiles": molecule.smiles,
                "inchikey": molecule.inchikey,
                "chemistry_group": chemistry_group,
            }
        )
    return pd.DataFrame(rows), invalid


__all__ = ["standardize_records"]
