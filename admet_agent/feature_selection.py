"""Leak-safe nested feature selection for scaffold-validation experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

EstimatorFactory = Callable[[int], Any]


@dataclass(frozen=True)
class NestedSelectionResult:
    """Serializable evidence from nested permutation feature selection."""

    chosen_feature_count: int
    selected_indices: tuple[int, ...]
    selected_feature_names: tuple[str, ...]
    feature_importances: tuple[float, ...]
    candidate_scores: dict[str, tuple[float, ...]]
    candidate_means: dict[str, float]
    inner_folds: tuple[dict[str, Any], ...]
    final_selector_split: dict[str, Any]
    mean_selected_set_jaccard: float
    protocol: str = (
        "outer-train-only nested scaffold CV; permutation ranking uses an "
        "inner-development scaffold holdout"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scaffold_groups(smiles: Sequence[str]) -> np.ndarray:
    """Return Bemis-Murcko scaffold labels matching the TDC split convention."""
    groups = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            groups.append(f"__invalid_{index}")
            continue
        groups.append(
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule,
                includeChirality=False,
            )
        )
    return np.asarray(groups, dtype=np.str_)


def _validate_inputs(
    features: np.ndarray,
    targets: np.ndarray,
    smiles: Sequence[str],
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    x = np.asarray(features)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if y.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if len(x) != len(y) or len(x) != len(smiles):
        raise ValueError("features, targets, and smiles must have equal row counts")
    if len(x) < 10 or x.shape[1] < 2:
        raise ValueError("nested selection requires at least 10 rows and 2 features")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    if feature_names is None:
        names = tuple(f"feature_{index}" for index in range(x.shape[1]))
    else:
        names = tuple(str(value) for value in feature_names)
        if len(names) != x.shape[1]:
            raise ValueError("feature_names must match the feature count")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
    return x, y, names


def _normalize_counts(counts: Sequence[int], width: int) -> tuple[int, ...]:
    values = []
    for raw in counts:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError("candidate feature counts must be integers")
        if raw < 1:
            raise ValueError("candidate feature counts must be positive")
        values.append(min(raw, width))
    values.append(width)
    return tuple(sorted(set(values)))


def _group_holdout(
    groups: np.ndarray,
    fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("at least three distinct scaffolds are required")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=fraction,
        random_state=random_state,
    )
    fit_indices, importance_indices = next(
        splitter.split(np.zeros(len(groups)), groups=groups)
    )
    return fit_indices, importance_indices


def _permutation_rank(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: EstimatorFactory,
    random_state: int,
    holdout_fraction: float,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit_indices, importance_indices = _group_holdout(
        groups,
        holdout_fraction,
        random_state,
    )
    estimator = estimator_factory(random_state)
    estimator.fit(x[fit_indices], y[fit_indices])
    importance = permutation_importance(
        estimator,
        x[importance_indices],
        y[importance_indices],
        scoring="neg_mean_absolute_error",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=1,
    ).importances_mean
    # Stable index tie-breaks make the selected recipe reconstructable.
    ranking = np.lexsort((np.arange(x.shape[1]), -importance))
    fit_groups = set(groups[fit_indices])
    importance_groups = set(groups[importance_indices])
    audit = {
        "n_selector_fit": len(fit_indices),
        "n_selector_importance": len(importance_indices),
        "selector_scaffold_overlap": len(fit_groups & importance_groups),
        "selector_fit_indices": fit_indices.astype(int).tolist(),
        "selector_importance_indices": importance_indices.astype(int).tolist(),
    }
    if audit["selector_scaffold_overlap"]:
        raise RuntimeError("selector scaffold holdout is not isolated")
    return ranking, importance, audit


def _mean_pairwise_jaccard(sets: Sequence[set[int]]) -> float:
    values = []
    for left_index, left in enumerate(sets):
        for right in sets[left_index + 1 :]:
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else 1.0


def nested_permutation_select(
    features: np.ndarray,
    targets: np.ndarray,
    smiles: Sequence[str],
    estimator_factory: EstimatorFactory,
    *,
    candidate_feature_counts: Sequence[int] = (32, 64, 96, 128),
    feature_names: Sequence[str] | None = None,
    inner_splits: int = 5,
    selector_holdout_fraction: float = 0.2,
    permutation_repeats: int = 3,
    random_state: int = 1,
) -> NestedSelectionResult:
    """Choose a feature count without exposing outer-validation labels.

    Each model-selection fold ranks features using a second scaffold holdout
    carved only from that fold's development rows. Candidate counts are scored
    on the untouched model-selection fold. A final ranking is then fitted using
    only the complete outer-training set.
    """
    x, y, names = _validate_inputs(features, targets, smiles, feature_names)
    if not 2 <= inner_splits <= 10:
        raise ValueError("inner_splits must be between 2 and 10")
    if not 0.1 <= selector_holdout_fraction <= 0.4:
        raise ValueError("selector_holdout_fraction must be between 0.1 and 0.4")
    if not 1 <= permutation_repeats <= 20:
        raise ValueError("permutation_repeats must be between 1 and 20")
    counts = _normalize_counts(candidate_feature_counts, x.shape[1])
    groups = scaffold_groups(smiles)
    if len(np.unique(groups)) < inner_splits:
        raise ValueError("distinct scaffolds must be at least inner_splits")

    # Do not use GroupKFold's newer shuffle arguments so the project remains
    # compatible with its declared scikit-learn >=1.4 floor.
    splitter = GroupKFold(n_splits=inner_splits)
    scores: dict[int, list[float]] = {count: [] for count in counts}
    fold_records = []
    fold_rankings = []
    for fold, (development, validation) in enumerate(
        splitter.split(x, y, groups),
        start=1,
    ):
        fold_seed = random_state * 10_000 + fold
        ranking, _, selector_audit = _permutation_rank(
            x[development],
            y[development],
            groups[development],
            estimator_factory,
            fold_seed,
            selector_holdout_fraction,
            permutation_repeats,
        )
        fold_rankings.append(ranking)
        fold_scores = {}
        for count in counts:
            # Preserve original column order after choosing the top set. This
            # prevents stochastic column subsampling from treating an "all
            # features" condition as a different learner merely because the
            # columns were permuted by importance rank.
            selected = np.sort(ranking[:count])
            estimator = estimator_factory(fold_seed + count)
            estimator.fit(x[development][:, selected], y[development])
            predictions = np.asarray(estimator.predict(x[validation][:, selected]))
            value = float(mean_absolute_error(y[validation], predictions))
            scores[count].append(value)
            fold_scores[str(count)] = value
        development_groups = set(groups[development])
        validation_groups = set(groups[validation])
        if development_groups & validation_groups:
            raise RuntimeError("inner model-selection scaffold folds overlap")
        fold_records.append(
            {
                "fold": fold,
                "n_development": len(development),
                "n_validation": len(validation),
                "model_selection_scaffold_overlap": 0,
                "development_indices": development.astype(int).tolist(),
                "validation_indices": validation.astype(int).tolist(),
                "candidate_scores": fold_scores,
                **selector_audit,
            }
        )

    means = {count: float(np.mean(values)) for count, values in scores.items()}
    chosen_count = min(counts, key=lambda count: (means[count], count))
    final_ranking, final_importance, final_audit = _permutation_rank(
        x,
        y,
        groups,
        estimator_factory,
        random_state * 10_000 + 999,
        selector_holdout_fraction,
        permutation_repeats,
    )
    selected = tuple(sorted(int(value) for value in final_ranking[:chosen_count]))
    chosen_sets = [set(map(int, ranking[:chosen_count])) for ranking in fold_rankings]
    return NestedSelectionResult(
        chosen_feature_count=chosen_count,
        selected_indices=selected,
        selected_feature_names=tuple(names[index] for index in selected),
        feature_importances=tuple(float(value) for value in final_importance),
        candidate_scores={
            str(count): tuple(float(value) for value in scores[count])
            for count in counts
        },
        candidate_means={str(count): means[count] for count in counts},
        inner_folds=tuple(fold_records),
        final_selector_split=final_audit,
        mean_selected_set_jaccard=_mean_pairwise_jaccard(chosen_sets),
    )
