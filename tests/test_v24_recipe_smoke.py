import json

import pytest

from scripts.smoke_v24_recipe import (
    load_frozen_config,
    read_training_seeds,
    sha256_file,
)


def test_smoke_loader_normalizes_cli_mapping_without_loading_data(tmp_path):
    mapping = tmp_path / "finalists.json"
    mapping.write_text(
        json.dumps(
            {
                "DILI": {
                    "kind": "recipe",
                    "recipe": "raw(2d_lightgbm@0.50+finetuned_unimol2@0.50)",
                }
            }
        ),
        encoding="utf-8",
    )

    frozen = load_frozen_config(mapping, "dili")

    assert frozen["task"] == "DILI"
    assert frozen["protocol"]["model_size"] == "84m"
    assert sha256_file(mapping)


def test_smoke_loader_rejects_unfrozen_task(tmp_path):
    mapping = tmp_path / "finalists.json"
    mapping.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="No frozen recipe"):
        load_frozen_config(mapping, "DILI")


def test_training_seed_reader_requires_both_runtime_seeds(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 3\nsplit_seed: 3\nepochs: 20\n", encoding="utf-8")
    assert read_training_seeds(config) == {"seed": 3, "split_seed": 3}

    config.write_text("seed: 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing Uni-Mol training seeds"):
        read_training_seeds(config)
