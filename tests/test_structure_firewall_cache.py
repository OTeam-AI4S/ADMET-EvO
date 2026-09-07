from __future__ import annotations

import json

from admet_agent.structure_firewall_cache import (
    prepare_structure_firewall_reference_cached,
)


def test_structure_reference_cache_is_input_bound_and_label_blind(tmp_path):
    arguments = {
        "train_smiles": ["CCO", "CCN"],
        "validation_smiles": [],
        "test_smiles": ["c1ccccc1"],
        "cache_root": tmp_path,
    }
    first = prepare_structure_firewall_reference_cached(**arguments)
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    assert payload["test_labels_loaded"] is False
    assert set(payload) == {
        "protocol",
        "train_input_sha256",
        "validation_input_sha256",
        "test_input_sha256",
        "train",
        "validation",
        "test",
        "test_labels_loaded",
    }
    second = prepare_structure_firewall_reference_cached(**arguments)
    assert second.test_input_sha256 == first.test_input_sha256
    assert set(second.test) == set(first.test)

    prepare_structure_firewall_reference_cached(
        **{**arguments, "test_smiles": ["c1ccncc1"]}
    )
    assert len(list(tmp_path.glob("*.json"))) == 2
