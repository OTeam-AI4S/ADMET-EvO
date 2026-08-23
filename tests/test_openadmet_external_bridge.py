import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from admet_agent.external_structure_firewall import _standard_molecule
from scripts.build_openadmet_cyp_split_manifest import _chemistry_group
from scripts.build_openadmet_external_bridge import build


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bridge_excludes_certification_group_and_marks_outer_fold(tmp_path):
    ethanol = _standard_molecule("CCO")
    benzene = _standard_molecule("c1ccccc1")
    assert ethanol is not None and benzene is not None
    split_path = tmp_path / "split.csv"
    split = pd.DataFrame(
        [
            {
                "inchikey": ethanol.inchikey,
                "standard_smiles": ethanol.smiles,
                "chemistry_group": _chemistry_group(ethanol.smiles),
                "partition": "development",
                "outer_fold": 2,
            },
            {
                "inchikey": benzene.inchikey,
                "standard_smiles": benzene.smiles,
                "chemistry_group": _chemistry_group(benzene.smiles),
                "partition": "certification",
                "outer_fold": -1,
            },
        ]
    )
    split.to_csv(split_path, index=False)
    split_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "training_only": True,
                "output": {"sha256": _sha256(split_path)},
            }
        )
    )

    source_path = tmp_path / "cyp2d6_filtered.csv"
    amine = _standard_molecule("CCN")
    assert amine is not None
    pd.DataFrame(
        {
            "Drug_ID": [1, 2, 3],
            "Drug": [ethanol.smiles, benzene.smiles, amine.smiles],
            "Y": [1, 0, 1],
            "standard_inchikey": [
                ethanol.inchikey,
                benzene.inchikey,
                amine.inchikey,
            ],
        }
    ).to_csv(source_path, index=False)
    source_path.with_suffix(".firewall.json").write_text(
        json.dumps(
            {
                "test_labels_loaded": False,
                "dataset_rejected_from_blind_distribution": False,
                "filtered_external_table_sha256": _sha256(source_path),
            }
        )
    )
    output_path = tmp_path / "bridge.csv"
    result = build(
        argparse.Namespace(
            split=split_path,
            source=[f"CYP2D6={source_path}"],
            output=output_path,
        )
    )

    bridge = pd.read_csv(output_path, keep_default_na=False)
    assert len(bridge) == 2
    exact = bridge.loc[bridge["inchikey"] == ethanol.inchikey].iloc[0]
    assert str(exact["excluded_outer_folds"]) == "2"
    novel = bridge.loc[bridge["inchikey"] == amine.inchikey].iloc[0]
    assert novel["excluded_outer_folds"] == ""
    audit = result["source_audits"]["CYP2D6"]
    assert audit["internal_certification_group_exclusions"] == 1
    assert audit["outer_fold_group_exclusions"] == {"2": 1}
