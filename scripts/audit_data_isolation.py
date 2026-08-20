"""Audit train/validation isolation without loading the official test split."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc.benchmark_group import admet_group

from admet_agent.runner import _columns

TASKS = (
    "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels",
    "DILI",
    "Half_Life_Obach",
    "Lipophilicity_AstraZeneca",
    "hERG",
)


def _tdc_scaffolds(smiles_values: set[str]) -> set[str]:
    """Match PyTDC's outer split definition: Bemis-Murcko, no chirality."""
    scaffolds = set()
    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        scaffolds.add(
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule,
                includeChirality=False,
            )
        )
    return scaffolds


def main() -> None:
    group = admet_group(path="data")
    records = []
    validation_smiles_by_task: dict[str, list[set[str]]] = {}
    for task in TASKS:
        validation_smiles_by_task[task] = []
        for seed in range(1, 6):
            train, valid = group.get_train_valid_split(
                benchmark=task, split_type="default", seed=seed
            )
            smiles_col, _ = _columns(train)
            train_smiles = set(train[smiles_col].astype(str))
            valid_smiles = set(valid[smiles_col].astype(str))
            validation_smiles_by_task[task].append(valid_smiles)
            train_scaffolds = _tdc_scaffolds(train_smiles)
            valid_scaffolds = _tdc_scaffolds(valid_smiles)
            records.append(
                {
                    "task": task,
                    "seed": seed,
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "smiles_overlap": len(train_smiles & valid_smiles),
                    "bemis_murcko_scaffold_overlap": len(
                        train_scaffolds & valid_scaffolds
                    ),
                }
            )
    cross_seed_overlap = {}
    for task, validation_sets in validation_smiles_by_task.items():
        pairs = list(combinations(validation_sets, 2))
        intersections = [len(left & right) for left, right in pairs]
        jaccards = [len(left & right) / len(left | right) for left, right in pairs]
        validation_union = set().union(*validation_sets)
        appearances = {
            smiles: sum(smiles in values for values in validation_sets)
            for smiles in validation_union
        }
        cross_seed_overlap[task] = {
            "validation_size_min": min(map(len, validation_sets)),
            "validation_size_max": max(map(len, validation_sets)),
            "mean_pairwise_smiles_intersection": sum(intersections) / len(intersections),
            "mean_pairwise_smiles_jaccard": sum(jaccards) / len(jaccards),
            "unique_smiles_seen_in_any_validation": len(validation_union),
            "smiles_seen_in_exactly_one_validation_seed": sum(
                count == 1 for count in appearances.values()
            ),
            "smiles_seen_in_multiple_validation_seeds": sum(
                count > 1 for count in appearances.values()
            ),
        }
    report = {
        "source": "official TDC ADMET Group train_val.csv",
        "split": "TDC default 87.5/12.5 scaffold split",
        "seeds": [1, 2, 3, 4, 5],
        "official_test_loaded": False,
        "all_smiles_disjoint": all(row["smiles_overlap"] == 0 for row in records),
        "all_scaffolds_disjoint": all(
            row["bemis_murcko_scaffold_overlap"] == 0 for row in records
        ),
        "cross_seed_validation_overlap": cross_seed_overlap,
        "records": records,
    }
    output = Path("runs/v23-ablation-study/data_isolation_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
