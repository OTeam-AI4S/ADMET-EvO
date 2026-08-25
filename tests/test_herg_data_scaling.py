import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from admet_agent.herg_data_scaling import (
    curate_binary_structures,
    deterministic_binary_cap,
    near_neighbor_keep_mask,
)


def test_binary_cap_is_deterministic_and_keeps_every_minority_row():
    frame = pd.DataFrame(
        {
            "id": list(range(10)),
            "smiles": ["C"] * 10,
            "label": [1, 1] + [0] * 8,
        }
    )
    first = deterministic_binary_cap(
        frame, id_column="id", label_column="label", majority_cap=3
    )
    second = deterministic_binary_cap(
        frame.sample(frac=1, random_state=7),
        id_column="id",
        label_column="label",
        majority_cap=3,
    )
    assert first[["id", "label"]].to_dict("records") == second[
        ["id", "label"]
    ].to_dict("records")
    assert first["label"].value_counts().to_dict() == {0: 3, 1: 2}


def test_curation_drops_conflicting_and_duplicate_identities():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "smiles": ["CC", "CC", "CCC", "CCC", "CO"],
            "label": [0, 0, 0, 1, 1],
        }
    )
    curated, audit = curate_binary_structures(
        frame,
        smiles_column="smiles",
        label_column="label",
        id_column="id",
        majority_cap=10,
    )
    assert set(curated["standard_smiles"]) == {"CC", "CO"}
    assert audit.conflicting_identity_rows == 2
    assert audit.output_identities == 2


def test_near_neighbor_firewall_excludes_exact_and_close_candidates():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(smiles):
        return generator.GetFingerprint(Chem.MolFromSmiles(smiles))

    keep = near_neighbor_keep_mask(
        [fp("CCO"), fp("c1ccccc1")], [fp("CCO")], threshold=0.90
    )
    assert keep.tolist() == [False, True]
