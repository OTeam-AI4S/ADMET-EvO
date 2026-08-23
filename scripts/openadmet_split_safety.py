"""Fail-closed identity and chemistry-group checks for OpenADMET CV."""

from __future__ import annotations

import pandas as pd

ALLOWED_TRAINING_PARTITIONS = {"development", "external_bridge"}


def assert_disjoint_training_evaluation(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    context: str,
) -> None:
    if training.empty or evaluation.empty:
        raise RuntimeError(f"{context}: training and evaluation must both be non-empty")
    required = {"inchikey", "chemistry_group", "partition"}
    for name, frame in (("training", training), ("evaluation", evaluation)):
        missing = required - set(frame)
        if missing:
            raise RuntimeError(f"{context}: {name} frame missing {sorted(missing)}")
        partitions = set(frame["partition"].dropna().astype(str))
        if not partitions.issubset(ALLOWED_TRAINING_PARTITIONS):
            raise RuntimeError(
                f"{context}: {name} contains forbidden partitions {sorted(partitions - ALLOWED_TRAINING_PARTITIONS)}"
            )
        if frame[list(required)].isna().any().any():
            raise RuntimeError(f"{context}: {name} has missing split-safety identifiers")
    combined = pd.concat(
        [
            training[["inchikey", "chemistry_group"]],
            evaluation[["inchikey", "chemistry_group"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    groups_per_identity = combined.groupby("inchikey")["chemistry_group"].nunique()
    inconsistent = sorted(groups_per_identity[groups_per_identity != 1].index.astype(str))
    if inconsistent:
        raise RuntimeError(
            f"{context}: identities map to multiple chemistry groups: {inconsistent[:5]}"
        )
    training_identity = set(training["inchikey"].astype(str))
    evaluation_identity = set(evaluation["inchikey"].astype(str))
    identity_overlap = sorted(training_identity & evaluation_identity)
    if identity_overlap:
        raise RuntimeError(
            f"{context}: molecular identities cross train/evaluation: {identity_overlap[:5]}"
        )
    training_groups = set(training["chemistry_group"].astype(str))
    evaluation_groups = set(evaluation["chemistry_group"].astype(str))
    group_overlap = sorted(training_groups & evaluation_groups)
    if group_overlap:
        raise RuntimeError(
            f"{context}: chemistry groups cross train/evaluation: {group_overlap[:5]}"
        )


__all__ = ["assert_disjoint_training_evaluation"]
