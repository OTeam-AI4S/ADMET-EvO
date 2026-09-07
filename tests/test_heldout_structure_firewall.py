from pathlib import Path

import pandas as pd

from admet_agent import heldout_structure_firewall as firewall


def test_structure_blocklist_reads_only_smiles_column(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    def fake_read_csv(path, *, usecols):
        observed["path"] = path
        observed["usecols"] = usecols
        return pd.DataFrame({"Drug": ["CCO", "c1ccccc1"]})

    monkeypatch.setattr(firewall.pd, "read_csv", fake_read_csv)
    identities, audit = firewall.load_structure_only_inchikey_blocklist(
        tmp_path / "test.csv"
    )

    assert observed["usecols"] == ["Drug"]
    assert len(identities) == 2
    assert audit["test_structure_columns_loaded"] == ["Drug"]
    assert audit["test_label_columns_loaded"] == []


def test_structure_blocklist_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    frame = pd.DataFrame({"Drug": ["CCO", "CCO", "CCN"]})
    monkeypatch.setattr(
        firewall.pd,
        "read_csv",
        lambda path, *, usecols: frame.loc[:, usecols].copy(),
    )

    first, first_audit = firewall.load_structure_only_inchikey_blocklist(
        tmp_path / "test.csv"
    )
    second, second_audit = firewall.load_structure_only_inchikey_blocklist(
        tmp_path / "test.csv"
    )

    assert first == second
    assert len(first) == 2
    assert (
        first_audit["test_structure_inchikey_blocklist_sha256"]
        == second_audit["test_structure_inchikey_blocklist_sha256"]
    )
