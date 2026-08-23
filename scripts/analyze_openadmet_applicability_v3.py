"""RDKit-API compatibility adapter for frozen applicability analysis v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scripts import analyze_openadmet_applicability_v1 as v1
from scripts import analyze_openadmet_applicability_v2 as v2


def nearest_train_similarity(frame: pd.DataFrame) -> np.ndarray:
    """Use the installed RDKit generator's molecule-based fingerprint API."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles_to_fp = {}
    for smiles in sorted(frame["standard_smiles"].astype(str).unique()):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES: {smiles}")
        smiles_to_fp[smiles] = generator.GetFingerprint(molecule)
    output = np.empty(len(frame), dtype=float)
    positions = {index: pos for pos, index in enumerate(frame.index)}
    for fold in sorted(frame["outer_fold_evaluated"].unique()):
        train_smiles = frame.loc[
            frame["outer_fold_evaluated"] != fold, "standard_smiles"
        ].astype(str).unique()
        evaluation = frame.loc[frame["outer_fold_evaluated"] == fold]
        train_fp = [smiles_to_fp[value] for value in train_smiles]
        if not train_fp:
            raise RuntimeError(f"No training molecules for outer fold {fold}")
        for index, smiles in evaluation["standard_smiles"].astype(str).items():
            output[positions[index]] = max(
                DataStructs.BulkTanimotoSimilarity(smiles_to_fp[smiles], train_fp)
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("reports/openadmet_cyp_applicability_protocol_v3.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_cyp_applicability_analysis_v3.json"),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    v1._nearest_train_similarity = nearest_train_similarity
    payload = v2.run(args)
    payload["version"] = "openadmet-cyp-applicability-v3"
    payload["rdkit_compatibility_adapter_scope"] = (
        "GetFingerprintFromSmiles_to_MolFromSmiles_plus_GetFingerprint_only"
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
