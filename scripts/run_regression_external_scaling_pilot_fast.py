"""Fast, measurement-equivalent curation adapter for regression scaling runs."""

from __future__ import annotations

import argparse

import pandas as pd
import run_regression_external_scaling_pilot as _base


def _curate(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    values = pd.to_numeric(frame[args.label_column], errors="coerce")
    work = frame.loc[values.notna(), [args.id_column, args.smiles_column]].copy()
    work["Y"] = values.loc[work.index].astype(float)
    work[args.smiles_column] = work[args.smiles_column].astype(str)

    # Assay replicates frequently repeat the submitted SMILES. Standardizing
    # each distinct string once is equivalent to per-row standardization; all
    # measurements are mapped back before the standardized-identity median.
    unique_smiles = work[args.smiles_column].drop_duplicates()
    mapping_rows = []
    for smiles in unique_smiles:
        molecule = _base._standard_molecule(smiles)
        if molecule is not None:
            mapping_rows.append(
                {
                    args.smiles_column: smiles,
                    "standard_smiles": molecule.smiles,
                    "standard_inchikey": molecule.inchikey,
                }
            )
    structure_map = pd.DataFrame(mapping_rows)
    if structure_map.empty:
        raise ValueError("No valid external structures")
    standardized = work.merge(structure_map, on=args.smiles_column, how="inner")
    spread = standardized.groupby("standard_inchikey")["Y"].agg(
        lambda values: float(values.max() - values.min())
    )
    collapsed = (
        standardized.groupby("standard_inchikey", as_index=False)
        .agg(
            standard_smiles=("standard_smiles", "first"),
            Y=("Y", "median"),
            replicate_count=("Y", "size"),
        )
        .reset_index(drop=True)
    )
    return collapsed, {
        "input_rows": len(frame),
        "numeric_label_rows": len(work),
        "unique_structure_inputs": len(unique_smiles),
        "valid_unique_structure_inputs": len(structure_map),
        "invalid_structure_rows": len(work) - len(standardized),
        "valid_structure_rows": len(standardized),
        "output_identities": len(collapsed),
        "replicate_rows": int((collapsed["replicate_count"] > 1).sum()),
        "median_within_identity_range": float(spread.median()),
    }


_base._curate = _curate
run = _base.run
