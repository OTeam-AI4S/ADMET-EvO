import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.build_openadmet_embedding_union import build


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_embedding_union_is_label_free_deduplicated_and_hash_bound(tmp_path):
    main_path = tmp_path / "main.json"
    main_path.write_text(
        json.dumps(
            {
                "contains_labels": False,
                "official_test_loaded": False,
                "inchikeys": ["id-a", "id-b"],
                "smiles": ["CC", "CCC"],
            }
        )
    )
    bridge_path = tmp_path / "bridge.csv"
    pd.DataFrame(
        {
            "inchikey": ["id-b", "id-c"],
            "standard_smiles": ["CCC", "CCCC"],
        }
    ).to_csv(bridge_path, index=False)
    bridge_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "training_only": True,
                "blind_test_opened": False,
                "output_sha256": _sha256(bridge_path),
            }
        )
    )
    output_path = tmp_path / "union.json"
    result = build(
        argparse.Namespace(main=main_path, bridge=bridge_path, output=output_path)
    )

    assert result["contains_labels"] is False
    assert result["official_test_loaded"] is False
    assert result["inchikeys"] == ["id-a", "id-b", "id-c"]
    assert result["smiles"] == ["CC", "CCC", "CCCC"]
    assert result["parents"]["bridge"]["sha256"] == _sha256(bridge_path)
    assert (
        result["parents"]["bridge"][
            "shared_identity_representation_reconciliations"
        ]
        == 0
    )


def test_embedding_union_prefers_main_smiles_for_shared_frozen_identity(tmp_path):
    main_path = tmp_path / "main.json"
    main_path.write_text(
        json.dumps(
            {
                "contains_labels": False,
                "official_test_loaded": False,
                "inchikeys": ["shared-id"],
                "smiles": ["CC=CC"],
            }
        )
    )
    bridge_path = tmp_path / "bridge.csv"
    pd.DataFrame(
        {
            "inchikey": ["shared-id", "bridge-only"],
            "standard_smiles": ["C/C=C/C", "CCCC"],
        }
    ).to_csv(bridge_path, index=False)
    bridge_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "training_only": True,
                "blind_test_opened": False,
                "output_sha256": _sha256(bridge_path),
            }
        )
    )
    output_path = tmp_path / "union.json"

    result = build(
        argparse.Namespace(main=main_path, bridge=bridge_path, output=output_path)
    )

    assert result["inchikeys"] == ["bridge-only", "shared-id"]
    assert result["smiles"] == ["CCCC", "CC=CC"]
    assert (
        result["parents"]["bridge"][
            "shared_identity_representation_policy"
        ]
        == "prefer_main_training_smiles_for_equal_frozen_inchikey"
    )
    assert (
        result["parents"]["bridge"][
            "shared_identity_representation_reconciliations"
        ]
        == 1
    )
