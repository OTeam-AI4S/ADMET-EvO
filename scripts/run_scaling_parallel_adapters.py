"""Parallel, row-equivalent curation adapters for the TDC22 screen."""

from __future__ import annotations

import multiprocessing as mp

import pandas as pd
import run_binary_external_scaling_pilot as _binary
import run_regression_external_scaling_pilot as _regression

from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.herg_data_scaling import CurationAudit, deterministic_binary_cap


def _standardize_one(smiles: str) -> tuple[str, str, str] | None:
    molecule = _standard_molecule(smiles)
    if molecule is None:
        return None
    return smiles, molecule.smiles, molecule.inchikey


def _parallel_structure_map(values: pd.Series, processes: int) -> list[dict]:
    inputs = values.astype(str).drop_duplicates().tolist()
    worker_count = max(1, min(int(processes), len(inputs)))
    chunksize = max(1, len(inputs) // max(1, worker_count * 16))
    if worker_count == 1:
        results = map(_standardize_one, inputs)
    else:
        # Linux fork keeps the loaded RDKit tables shared copy-on-write.
        with mp.get_context("fork").Pool(worker_count) as pool:
            results = pool.map(_standardize_one, inputs, chunksize=chunksize)
    return [
        {"raw_smiles": raw, "standard_smiles": smiles, "standard_inchikey": key}
        for result in results
        if result is not None
        for raw, smiles, key in [result]
    ]


def _curate_regression(frame: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    values = pd.to_numeric(frame[args.label_column], errors="coerce")
    work = frame.loc[values.notna(), [args.id_column, args.smiles_column]].copy()
    work["Y"] = values.loc[work.index].astype(float)
    work["raw_smiles"] = work[args.smiles_column].astype(str)
    unique_count = work["raw_smiles"].nunique()
    structure_map = pd.DataFrame(
        _parallel_structure_map(work["raw_smiles"], args.n_jobs)
    )
    if structure_map.empty:
        raise ValueError("No valid external structures")
    standardized = work.merge(structure_map, on="raw_smiles", how="inner")
    spread = standardized.groupby("standard_inchikey")["Y"].agg(
        lambda observations: float(observations.max() - observations.min())
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
        "unique_structure_inputs": unique_count,
        "valid_unique_structure_inputs": len(structure_map),
        "invalid_structure_rows": len(work) - len(standardized),
        "valid_structure_rows": len(standardized),
        "output_identities": len(collapsed),
        "replicate_rows": int((collapsed["replicate_count"] > 1).sum()),
        "median_within_identity_range": float(spread.median()),
    }


def _curate_binary(
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
    selected["raw_smiles"] = selected[smiles_column].astype(str)
    # Binary runner has no n_jobs curation argument, so use the screen's fixed
    # eight-process budget; model fitting remains controlled separately.
    structure_map = pd.DataFrame(_parallel_structure_map(selected["raw_smiles"], 8))
    if structure_map.empty:
        raise ValueError("No valid external structures survived standardization")
    work = selected[[id_column, "raw_smiles", label_column]].merge(
        structure_map, on="raw_smiles", how="inner"
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


_regression._curate = _curate_regression
_binary.curate_binary_structures = _curate_binary
run_regression = _regression.run
run_binary = _binary.run
