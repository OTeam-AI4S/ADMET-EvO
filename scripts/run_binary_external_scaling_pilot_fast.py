"""Fast, row-equivalent curation adapter for binary scaling runs."""

from __future__ import annotations

import pandas as pd
import run_binary_external_scaling_pilot as _base

from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.herg_data_scaling import CurationAudit, deterministic_binary_cap


def _curate_binary_structures(
    frame: pd.DataFrame,
    *,
    smiles_column: str,
    label_column: str,
    id_column: str,
    majority_cap: int,
) -> tuple[pd.DataFrame, CurationAudit]:
    selected = deterministic_binary_cap(
        frame,
        id_column=id_column,
        label_column=label_column,
        majority_cap=majority_cap,
    )
    selected[smiles_column] = selected[smiles_column].astype(str)
    mapping_rows = []
    for smiles in selected[smiles_column].drop_duplicates():
        molecule = _standard_molecule(smiles)
        if molecule is not None:
            mapping_rows.append(
                {
                    smiles_column: smiles,
                    "standard_smiles": molecule.smiles,
                    "standard_inchikey": molecule.inchikey,
                }
            )
    structure_map = pd.DataFrame(mapping_rows)
    if structure_map.empty:
        raise ValueError("No valid external structures survived standardization")
    work = selected[[id_column, smiles_column, label_column]].merge(
        structure_map, on=smiles_column, how="inner"
    )
    work = work.rename(columns={id_column: "source_record_id", label_column: "Y"})
    work["source_record_id"] = work["source_record_id"].astype(str)
    work["Y"] = work["Y"].astype(int)
    valid_rows = len(work)
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
        valid_structure_rows=valid_rows,
        invalid_structure_rows=len(selected) - valid_rows,
        conflicting_identity_rows=conflict_rows,
        duplicate_identity_rows=duplicate_rows,
        output_identities=len(work),
        positive_rate=float(work["Y"].mean()),
    )
    return work, audit


_base.curate_binary_structures = _curate_binary_structures
run = _base.run
