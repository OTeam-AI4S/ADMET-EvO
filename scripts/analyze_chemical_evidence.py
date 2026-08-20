"""Relate 3D expert gains to scaffold shift and molecular chemistry profiles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from scipy.stats import skew, spearmanr
from tdc.benchmark_group import admet_group

from admet_agent.runner import _columns
from admet_agent.tasks import resolve_task


def _profile(train_smiles: list[str], valid_smiles: list[str]) -> dict[str, float]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_mols = [Chem.MolFromSmiles(value) for value in train_smiles]
    valid_mols = [Chem.MolFromSmiles(value) for value in valid_smiles]
    train_fps = [generator.GetFingerprint(mol) for mol in train_mols if mol is not None]
    similarities = []
    rotatable = []
    fraction_sp3 = []
    molecular_weight = []
    tpsa = []
    stereo = []
    for smiles, mol in zip(valid_smiles, valid_mols, strict=True):
        if mol is None:
            continue
        fingerprint = generator.GetFingerprint(mol)
        similarities.append(max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)))
        rotatable.append(Lipinski.NumRotatableBonds(mol))
        fraction_sp3.append(Descriptors.FractionCSP3(mol))
        molecular_weight.append(Descriptors.MolWt(mol))
        tpsa.append(Descriptors.TPSA(mol))
        stereo.append(float("@" in smiles))
    return {
        "mean_nearest_train_tanimoto": float(np.mean(similarities)),
        "mean_rotatable_bonds": float(np.mean(rotatable)),
        "mean_fraction_csp3": float(np.mean(fraction_sp3)),
        "mean_molecular_weight": float(np.mean(molecular_weight)),
        "mean_tpsa": float(np.mean(tpsa)),
        "stereochemical_smiles_fraction": float(np.mean(stereo)),
    }


def main() -> None:
    root = Path("runs/unimol2-pilot-v22")
    group = admet_group(path="data")
    records = []
    for path in sorted(root.glob("*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        task = resolve_task(result["task"])
        for seed in range(1, 6):
            train, valid = group.get_train_valid_split(
                benchmark=task.name, split_type="default", seed=seed
            )
            smiles_col, y_col = _columns(train)
            scores = result["seeds"][str(seed)]["scores"]
            best_2d = (
                min(scores[name] for name in scores if name.startswith("2d_"))
                if task.metric == "mae"
                else max(scores[name] for name in scores if name.startswith("2d_"))
            )
            fine = scores["finetuned_unimol2"]
            gain = best_2d - fine if task.metric == "mae" else fine - best_2d
            chemistry = _profile(
                train[smiles_col].astype(str).tolist(),
                valid[smiles_col].astype(str).tolist(),
            )
            target = train[y_col].to_numpy()
            records.append(
                {
                    "task": task.name,
                    "seed": seed,
                    "metric": task.metric,
                    "n_train": len(train),
                    "target_skew": float(skew(target)),
                    "positive_fraction": (
                        float(np.mean(target))
                        if task.task_type == "classification"
                        else None
                    ),
                    "unimol2_gain_over_best_2d": gain,
                    **chemistry,
                }
            )
    numeric = (
        "n_train",
        "target_skew",
        "mean_nearest_train_tanimoto",
        "mean_rotatable_bonds",
        "mean_fraction_csp3",
        "mean_molecular_weight",
        "mean_tpsa",
        "stereochemical_smiles_fraction",
    )
    correlations = {}
    gains = np.asarray([record["unimol2_gain_over_best_2d"] for record in records])
    for name in numeric:
        values = np.asarray([record[name] for record in records])
        correlations[name] = float(spearmanr(values, gains).statistic)
    by_task = {}
    for task_name in sorted({record["task"] for record in records}):
        selected = [record for record in records if record["task"] == task_name]
        by_task[task_name] = {
            "mean_unimol2_gain_over_best_2d": float(
                np.mean([record["unimol2_gain_over_best_2d"] for record in selected])
            ),
            **{
                name: float(np.mean([record[name] for record in selected]))
                for name in numeric
            },
        }
    payload = {"records": records, "by_task": by_task, "correlations": correlations}
    (root / "chemical_evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"by_task": by_task, "correlations": correlations}, indent=2))


if __name__ == "__main__":
    main()
