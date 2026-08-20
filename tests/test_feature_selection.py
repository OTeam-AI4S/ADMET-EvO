import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from admet_agent.feature_selection import (
    nested_permutation_select,
    scaffold_groups,
)


def _factory(seed: int):
    return ExtraTreesRegressor(
        n_estimators=40,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )


def test_scaffold_groups_is_deterministic_and_handles_invalid_smiles():
    smiles = ["CCOc1ccccc1", "CCNc1ccccc1", "not-smiles"]
    first = scaffold_groups(smiles)
    second = scaffold_groups(smiles)
    assert np.array_equal(first, second)
    assert first[0] == first[1]
    assert first[2] == "__invalid_2"


def test_nested_selection_keeps_every_evaluation_scaffold_isolated():
    rng = np.random.default_rng(7)
    # Distinct ring systems provide enough deterministic scaffold groups.
    rings = [
        "c1ccccc1",
        "C1CCCCC1",
        "c1ccncc1",
        "c1ccoc1",
        "c1ccsc1",
        "C1CCCC1",
        "C1CCNC1",
        "C1CCOC1",
        "c1ccc2ccccc2c1",
        "c1ccc2[nH]ccc2c1",
        "O=C1CCCCC1",
        "O=C1CCCC1",
    ]
    smiles = [f"{prefix}{ring}" for ring in rings for prefix in ("C", "CC", "CCC")]
    features = rng.normal(size=(len(smiles), 12))
    targets = 2.5 * features[:, 0] - 1.5 * features[:, 1] + rng.normal(
        scale=0.05, size=len(smiles)
    )
    result = nested_permutation_select(
        features,
        targets,
        smiles,
        _factory,
        candidate_feature_counts=(2, 4, 8),
        feature_names=[f"d{index}" for index in range(features.shape[1])],
        inner_splits=3,
        permutation_repeats=2,
        random_state=9,
    )

    assert result.chosen_feature_count in {2, 4, 8, 12}
    assert len(result.selected_indices) == result.chosen_feature_count
    assert len(set(result.selected_indices)) == result.chosen_feature_count
    assert list(result.selected_indices) == sorted(result.selected_indices)
    assert set(result.selected_feature_names) <= {f"d{i}" for i in range(12)}
    assert result.final_selector_split["selector_scaffold_overlap"] == 0
    assert 0.0 <= result.mean_selected_set_jaccard <= 1.0
    for fold in result.inner_folds:
        assert fold["model_selection_scaffold_overlap"] == 0
        assert fold["selector_scaffold_overlap"] == 0
        assert set(fold["development_indices"]).isdisjoint(
            fold["validation_indices"]
        )


def test_nested_selection_rejects_bad_contracts():
    features = np.ones((12, 3))
    targets = np.ones(12)
    smiles = [f"C1{'C' * index}CCCC1" for index in range(12)]
    with pytest.raises(ValueError, match="feature_names"):
        nested_permutation_select(
            features,
            targets,
            smiles,
            _factory,
            feature_names=("a", "b"),
        )
    with pytest.raises(TypeError, match="counts"):
        nested_permutation_select(
            features,
            targets,
            smiles,
            _factory,
            candidate_feature_counts=(True,),
        )
