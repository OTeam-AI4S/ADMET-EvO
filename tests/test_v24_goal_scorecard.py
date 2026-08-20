import hashlib
import json

import pytest

from scripts.build_v24_goal_scorecard import _freeze_digest, _saved_training_seeds


def test_official_freeze_digest_is_canonical():
    freeze = {
        "protocol": "v1",
        "tasks": ["DILI"],
        "finalists": {"DILI": {"kind": "recipe"}},
    }
    payload = {
        "protocol": freeze["protocol"],
        "tasks": freeze["tasks"],
        "finalists": freeze["finalists"],
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert _freeze_digest(freeze) == expected


def test_saved_training_seed_reader_detects_incomplete_seed_pair(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 4\nsplit_seed: 4\nepochs: 20\n", encoding="utf-8")
    assert _saved_training_seeds(config) == {"seed": 4, "split_seed": 4}

    config.write_text("seed: 4\n", encoding="utf-8")
    assert _saved_training_seeds(config) != {"seed": 4, "split_seed": 4}


def test_missing_saved_training_config_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        _saved_training_seeds(tmp_path / "missing.yaml")
