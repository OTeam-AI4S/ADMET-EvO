import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.apply_openadmet_external_firewall_v2 import (
    _standardizer_hash,
    apply_firewall,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, implementation_hash: str | None = None):
    blind_path = tmp_path / "blind.csv"
    blind_path.write_text("Molecule_Name,SMILES\nblind-1,CC\n", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    manifest_path = reports / "manifest.json"
    manifest = {
        "blind_table": {
            "path": "blind.csv",
            "sha256": _sha256(blind_path),
            "ordered_columns": ["Molecule_Name", "SMILES"],
            "molecule_name_column": "Molecule_Name",
            "smiles_column": "SMILES",
            "expected_rows": 1,
            "contains_label_columns": False,
        },
        "record_exclusions": {"tanimoto_greater_than_or_equal": 0.9},
        "standardization": {
            "implementation_hash": implementation_hash or _standardizer_hash()
        },
        "test_labels_loaded": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_path = tmp_path / "external.csv"
    pd.DataFrame(
        {"Drug": ["CC", "c1ccccc1"], "Y": [1, 0], "source": ["x", "x"]}
    ).to_csv(source_path, index=False)
    return blind_path, manifest_path, source_path


def test_record_level_firewall_removes_exact_row_without_rejecting_source(tmp_path):
    _, manifest_path, source_path = _fixture(tmp_path)
    output_path = tmp_path / "filtered.csv"
    audit = apply_firewall(
        source_path=source_path,
        expected_source_sha256=_sha256(source_path),
        smiles_column="Drug",
        output_path=output_path,
        manifest_path=manifest_path,
    )

    output = pd.read_csv(output_path)
    assert len(output) == 1
    assert output.loc[0, "Drug"] == "c1ccccc1"
    assert audit["exact_blind_identity_exclusion_rows"] == 1
    assert audit["near_blind_neighbor_exclusion_rows"] == 0
    assert audit["dataset_rejected_from_blind_distribution"] is False
    assert audit["blind_distribution_used_for_source_or_model_selection"] is False
    assert audit["test_labels_loaded"] is False


def test_blind_schema_mismatch_fails_closed(tmp_path):
    blind_path, manifest_path, source_path = _fixture(tmp_path)
    blind_path.write_text("SMILES,Molecule_Name\nCC,blind-1\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["blind_table"]["sha256"] = _sha256(blind_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ordered schema mismatch"):
        apply_firewall(
            source_path=source_path,
            expected_source_sha256=_sha256(source_path),
            smiles_column="Drug",
            output_path=tmp_path / "filtered.csv",
            manifest_path=manifest_path,
        )


def test_unfrozen_standardizer_hash_fails_closed(tmp_path):
    _, manifest_path, source_path = _fixture(
        tmp_path, implementation_hash="pending_v2_source_freeze"
    )
    with pytest.raises(RuntimeError, match="has not been frozen"):
        apply_firewall(
            source_path=source_path,
            expected_source_sha256=_sha256(source_path),
            smiles_column="Drug",
            output_path=tmp_path / "filtered.csv",
            manifest_path=manifest_path,
        )
