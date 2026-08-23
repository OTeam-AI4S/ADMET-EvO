from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from admet_agent import matched_state
from admet_agent.matched_state import construct_matched_state, matched_action_spec
from scripts.init_ecvoi_v4_matched_state_protocol import build_protocol
from scripts.run_ecvoi_v4_matched_states import load_protocol


def _frame(negative: int, positive: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Drug": ["CCO"] * negative + ["CCN"] * positive,
            "Y": [0] * negative + [1] * positive,
        }
    )


def test_imbalance_states_are_deterministic_and_ordered(monkeypatch):
    train = _frame(120, 60)
    valid = _frame(10, 10)
    monkeypatch.setattr(
        matched_state,
        "_max_validation_similarity",
        lambda train_smiles, valid_smiles: np.linspace(0.1, 0.9, len(train_smiles)),
    )

    mild, mild_audit = construct_matched_state(
        train, valid, state="imbalance_mild", seed=17
    )
    repeat, repeat_audit = construct_matched_state(
        train, valid, state="imbalance_mild", seed=17
    )
    severe, severe_audit = construct_matched_state(
        train, valid, state="imbalance_severe", seed=17
    )

    assert mild.equals(repeat)
    assert mild_audit == repeat_audit
    assert len(severe) < len(mild) < len(train)
    assert severe["Y"].mean() < mild["Y"].mean() < train["Y"].mean()
    assert not mild_audit.validation_labels_read_for_state


def test_domain_shift_removes_closest_structures_with_class_ratio_fixed(monkeypatch):
    train = _frame(50, 50)
    valid = _frame(10, 10)
    similarity = np.concatenate([np.linspace(0.1, 0.9, 50)] * 2)
    monkeypatch.setattr(
        matched_state,
        "_max_validation_similarity",
        lambda train_smiles, valid_smiles: similarity,
    )

    shifted, audit = construct_matched_state(
        train, valid, state="domain_shift_severe", seed=29
    )

    assert len(shifted) == 60
    assert shifted["Y"].mean() == pytest.approx(0.5)
    assert (
        audit.median_max_validation_similarity_after
        < audit.median_max_validation_similarity_before
    )
    assert audit.manipulation_family == "domain_shift"


def test_action_specs_change_one_declared_axis():
    baseline = matched_action_spec(
        "BBB_Martins", action="baseline", seed=17, workers=2, max_minutes=5
    )
    feature = matched_action_spec(
        "BBB_Martins", action="feature_rich", seed=17, workers=2, max_minutes=5
    )
    imbalance = matched_action_spec(
        "BBB_Martins", action="imbalance_aware", seed=17, workers=2, max_minutes=5
    )

    assert baseline.data_policy == feature.data_policy == imbalance.data_policy == "identity_safe"
    assert baseline.model == feature.model == imbalance.model == "extra_trees"
    assert baseline.representations != feature.representations
    assert baseline.imbalance == feature.imbalance == "none"
    assert imbalance.representations == baseline.representations
    assert imbalance.imbalance == "class_weight"


def test_protocol_is_hash_bound_and_keeps_confirmation_sealed(tmp_path):
    protocol = build_protocol()
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    loaded = load_protocol(path)

    assert loaded["sealed_confirmation_tasks"] == 19
    assert loaded["test_loaded"] is False
    assert loaded["leaderboard_loaded"] is False
    assert loaded["confirmation_unlock"].startswith("screen execution integrity")
    assert loaded["sha256"] == protocol["sha256"]

    tampered = dict(protocol)
    tampered["max_minutes_per_fit"] = 16
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_protocol(path)


def test_protocol_binds_every_implementation_file():
    protocol = build_protocol()
    for name, expected in protocol["implementation_sha256"].items():
        assert hashlib.sha256(open(name, "rb").read()).hexdigest() == expected
