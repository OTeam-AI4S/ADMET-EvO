from dataclasses import dataclass

import pandas as pd

from scripts import build_hardness_stability_input as module


@dataclass
class _Card:
    task: str

    def to_dict(self):
        return {
            "task": self.task,
            "overall": 0.4,
            "dimensions": {
                "scarcity": {"severity": 0.4, "confidence": 1.0, "evidence": {}}
            },
            "dominant_factors": ["scarcity"],
            "recommended_actions": ["replicate"],
            "discovery_only": True,
        }


class _Group:
    def get_train_valid_split(self, benchmark, split_type, seed):
        assert split_type == "default"
        train = pd.DataFrame({"Drug": ["CC", "CCC"], "Y": [0, 1]})
        valid = pd.DataFrame({"Drug": ["CCCC"], "Y": [seed % 2]})
        return train, valid


def test_builder_is_discovery_only_complete_and_deterministic(monkeypatch):
    monkeypatch.setattr(
        module,
        "profile_hardness",
        lambda train, valid, spec, smiles, target, sample_limit, seed: _Card(spec.name),
    )
    left = module.build(_Group(), ["HIA_Hou"], [2, 1], sample_limit=100)
    right = module.build(_Group(), ["HIA_Hou"], [2, 1], sample_limit=100)
    assert left == right
    assert left["discovery_only"] is True
    assert left["test_loaded"] is False
    assert left["model_fits"] == 0
    assert [(row["task"], row["seed"]) for row in left["cards"]] == [
        ("HIA_Hou", 1),
        ("HIA_Hou", 2),
    ]
    assert all(len(row["train_sha256"]) == 64 for row in left["split_manifest"])
