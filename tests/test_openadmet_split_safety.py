import pandas as pd
import pytest

from scripts.openadmet_split_safety import assert_disjoint_training_evaluation


def _frame(identities, groups, partition="development"):
    return pd.DataFrame(
        {
            "inchikey": identities,
            "chemistry_group": groups,
            "partition": [partition] * len(identities),
        }
    )


def test_disjoint_identity_and_chemistry_groups_pass():
    assert_disjoint_training_evaluation(
        _frame(["a", "b"], ["g1", "g2"]),
        _frame(["c"], ["g3"]),
        context="fixture",
    )


def test_external_identity_overlap_fails_even_if_group_id_is_wrong():
    training = pd.concat(
        [
            _frame(["a"], ["g1"]),
            _frame(["c"], ["external-wrong-group"], "external_bridge"),
        ],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="multiple chemistry groups"):
        assert_disjoint_training_evaluation(
            training,
            _frame(["c"], ["g3"]),
            context="outer-0",
        )


def test_group_overlap_fails_without_exact_identity_overlap():
    with pytest.raises(RuntimeError, match="chemistry groups cross"):
        assert_disjoint_training_evaluation(
            _frame(["a"], ["g1"]),
            _frame(["b"], ["g1"]),
            context="outer-0",
        )


def test_certification_partition_is_forbidden():
    with pytest.raises(RuntimeError, match="forbidden partitions"):
        assert_disjoint_training_evaluation(
            _frame(["a"], ["g1"]),
            _frame(["b"], ["g2"], "certification"),
            context="outer-0",
        )
