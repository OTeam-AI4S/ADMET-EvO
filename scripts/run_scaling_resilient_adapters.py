"""Resilient TDC feature adapter for rare non-roundtripping RDKit SMILES."""

from __future__ import annotations

import json

import pandas as pd
import run_regression_external_scaling_pilot as _regression
from rdkit import Chem
from run_scaling_parallel_adapters import run_regression as _parallel_run_regression

from admet_agent.external_structure_firewall import _standard_molecule

_FALLBACKS: list[dict] = []


def _standardized_tdc(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for input_index, (raw_smiles, target) in enumerate(
        frame[["Drug", "Y"]].itertuples(index=False, name=None)
    ):
        raw_smiles = str(raw_smiles)
        standardized = _standard_molecule(raw_smiles)
        if standardized is not None and Chem.MolFromSmiles(standardized.smiles) is not None:
            feature_smiles = standardized.smiles
            identity = standardized.inchikey
        else:
            raw_molecule = Chem.MolFromSmiles(raw_smiles)
            if raw_molecule is None:
                raise ValueError(f"TDC input structure is not fingerprintable at row {input_index}")
            feature_smiles = Chem.MolToSmiles(raw_molecule, canonical=True, isomericSmiles=True)
            identity = Chem.MolToInchiKey(raw_molecule)
            _FALLBACKS.append(
                {
                    "input_index": input_index,
                    "raw_smiles": raw_smiles,
                    "failed_standard_smiles": None if standardized is None else standardized.smiles,
                    "feature_smiles": feature_smiles,
                    "reason": "standardization_failed_or_non_roundtripping_smiles",
                }
            )
        rows.append(
            {
                "standard_smiles": feature_smiles,
                "standard_inchikey": identity,
                "Y": float(target),
            }
        )
    return pd.DataFrame(rows)


_regression._standardized_tdc = _standardized_tdc


def run_regression(args) -> dict:
    _FALLBACKS.clear()
    payload = _parallel_run_regression(args)
    unique = {
        (item["raw_smiles"], item["failed_standard_smiles"]): item
        for item in _FALLBACKS
    }
    payload["tdc_feature_fallbacks"] = list(unique.values())
    payload["tdc_feature_fallback_count"] = len(unique)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
