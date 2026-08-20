import pandas as pd

from scripts.export_minimol_validation_data import build_export


class FakeGroup:
    def get_train_valid_split(self, benchmark, split_type, seed):
        assert benchmark == "CYP2C9_Substrate_CarbonMangels"
        assert split_type == "default"
        train = pd.DataFrame({"Drug": ["CC", "CCC"], "Y": [0, 1]})
        valid = pd.DataFrame({"Drug": ["CO"], "Y": [seed % 2]})
        return train, valid


def test_minimol_export_never_requests_official_test_and_separates_labels():
    smiles, splits = build_export(
        FakeGroup(), "CYP2C9_Substrate_CarbonMangels", [1, 2]
    )

    assert smiles["smiles"] == ["CC", "CCC", "CO"]
    assert smiles["contains_labels"] is False
    assert "splits" not in smiles
    assert splits["official_test_loaded"] is False
    assert splits["splits"]["1"]["valid_y"] == [1.0]
