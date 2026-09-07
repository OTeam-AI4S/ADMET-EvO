from pathlib import Path

import pandas as pd
import pytest

from admet_agent.evidence_packet_splits import (
    scaffold_partition_indices,
    scaffold_train_validation_split,
)


def test_reference_scaffold_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame(
        {
            "Drug": [
                "c1ccccc1O",
                "c1ccccc1N",
                "C1CCCCC1O",
                "C1CCCCC1N",
                "CCO",
                "CCN",
            ],
            "Y": range(6),
        }
    )
    first_train, first_validation, first_audit = scaffold_train_validation_split(
        frame, seed=42
    )
    second_train, second_validation, second_audit = scaffold_train_validation_split(
        frame, seed=42
    )
    assert first_train.equals(second_train)
    assert first_validation.equals(second_validation)
    assert first_audit["split_sha256"] == second_audit["split_sha256"]
    assert set(first_audit["train_indices"]).isdisjoint(
        first_audit["validation_indices"]
    )
    assert sorted(
        first_audit["train_indices"] + first_audit["validation_indices"]
    ) == list(range(len(frame)))


def test_ames_split_matches_audited_reference_counts():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/admet_group/ames/train_val.csv"
    if not source.is_file():
        pytest.skip("local TDC cache is not available")
    frame = pd.read_csv(source, usecols=["Drug", "Y"])
    train_indices, validation_indices = scaffold_partition_indices(
        frame["Drug"], fraction=0.20, seed=42
    )
    assert (len(train_indices), len(validation_indices)) == (3963, 1858)
