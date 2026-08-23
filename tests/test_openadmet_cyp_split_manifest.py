import pandas as pd

from scripts.build_openadmet_cyp_split_manifest import (
    _availability_by_partition,
    _freeze_partitions,
)


def test_training_only_partitions_keep_chemistry_groups_locked_and_balanced():
    frame = pd.DataFrame(
        {
            "inchikey": [f"identity-{index}" for index in range(60)],
            "chemistry_group": [f"group-{index // 3}" for index in range(60)],
        }
    )
    output = _freeze_partitions(
        frame,
        certification_fraction=0.15,
        seed=20260821,
        outer_folds=5,
        inner_folds=4,
    )

    assert output.groupby("chemistry_group")["partition"].nunique().max() == 1
    development = output[output["partition"] == "development"]
    assert development.groupby("chemistry_group")["outer_fold"].nunique().max() == 1
    assert set(development["outer_fold"]) == set(range(5))
    outer_counts = development["outer_fold"].value_counts()
    assert outer_counts.max() - outer_counts.min() <= 3

    for outer in range(5):
        column = f"inner_fold_when_outer_{outer}"
        eligible = development[development["outer_fold"] != outer]
        assert eligible.groupby("chemistry_group")[column].nunique().max() == 1
        assert set(eligible[column]) == set(range(4))
        assert (output.loc[output["outer_fold"] == outer, column] == -1).all()
    assert (
        output.loc[output["partition"] == "certification", "outer_fold"] == -1
    ).all()


def test_partition_assignment_is_deterministic_under_row_reordering():
    frame = pd.DataFrame(
        {
            "inchikey": [f"identity-{index}" for index in range(80)],
            "chemistry_group": [f"group-{index // 4}" for index in range(80)],
        }
    )
    left = _freeze_partitions(
        frame,
        certification_fraction=0.15,
        seed=17,
        outer_folds=5,
        inner_folds=4,
    ).set_index("inchikey")
    right = _freeze_partitions(
        frame.sample(frac=1.0, random_state=9),
        certification_fraction=0.15,
        seed=17,
        outer_folds=5,
        inner_folds=4,
    ).set_index("inchikey")
    columns = ["partition", "outer_fold"] + [
        f"inner_fold_when_outer_{outer}" for outer in range(5)
    ]
    pd.testing.assert_frame_equal(
        left[columns].sort_index(), right[columns].sort_index()
    )


def test_oversized_chemistry_group_is_not_split_or_allowed_to_break_holdout_cap():
    frame = pd.DataFrame(
        {
            "inchikey": [f"identity-{index}" for index in range(100)],
            "chemistry_group": [
                "oversized-group" if index < 40 else f"small-group-{index}"
                for index in range(100)
            ],
        }
    )
    output = _freeze_partitions(
        frame,
        certification_fraction=0.125,
        seed=20260821,
        outer_folds=5,
        inner_folds=4,
    )

    certification = output[output["partition"] == "certification"]
    assert 0.10 <= len(certification) / len(output) <= 0.15
    assert "oversized-group" not in set(certification["chemistry_group"])
    assert output.groupby("chemistry_group")["partition"].nunique().max() == 1


def test_label_availability_covers_primary_and_auxiliary_modalities():
    split = pd.DataFrame(
        {
            "inchikey": ["identity-a", "identity-b"],
            "partition": ["certification", "development"],
            "outer_fold": [-1, 2],
        }
    )
    frames = {
        "direct": pd.DataFrame(
            {
                "SMILES": ["raw-a", "raw-b"],
                "CYP1A2_pIC50_direct_inhibition": [5.0, 6.0],
            }
        ),
        "tdi": pd.DataFrame(
            {
                "SMILES": ["raw-a", "raw-b"],
                "CYP2D6_is_TDI": [1, 0],
                "CYP2D6_pIC50_TDI_condition": [5.5, 6.5],
                "CYP1A2_pIC50_direct_inhibition": [5.1, 6.1],
            }
        ),
        "emax": pd.DataFrame(
            {
                "SMILES": ["raw-a", "raw-b"],
                "CYP2D6_EmaxVsPosCtrl_TDI_condition": [0.2, 0.4],
            }
        ),
        "single": pd.DataFrame(
            {
                "SMILES": ["raw-a", "raw-b"],
                "enzyme": ["CYP2D6", "CYP3A4"],
                "log2fc_estimate": [-1.0, -2.0],
            }
        ),
    }
    result = _availability_by_partition(
        frames,
        {"raw-a": "identity-a", "raw-b": "identity-b"},
        split,
    )

    for target in (
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2D6_is_TDI",
        "CYP2D6_pIC50_TDI_condition",
        "CYP2D6_EmaxVsPosCtrl_TDI_condition",
    ):
        assert result[target] == {"certification": 1, "outer_2": 1}
    assert result[
        "tdi_paired_direct:CYP1A2_pIC50_direct_inhibition"
    ] == {"certification": 1, "outer_2": 1}
    assert result["CYP2D6_single_concentration_log2fc_estimate"] == {
        "certification": 1
    }
    assert result["CYP3A4_single_concentration_log2fc_estimate"] == {"outer_2": 1}
