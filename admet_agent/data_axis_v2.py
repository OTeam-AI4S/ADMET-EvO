"""Train-only data curation and valuation primitives for three-axis ablation v2.

The functions in this module never inspect validation or test labels.  They
operate only on the training partition supplied by the caller and return an
explicit audit packet alongside the selected rows and sample weights.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .adaptive_external_data import molecular_scaffold


NATIVE_POLICIES = frozenset({"official", "curated"})
WEIGHT_POLICIES = frozenset(
    {"none", "scaffold_balance", "target_balance", "combined"}
)


def _normalized_clipped_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or not len(weights):
        raise ValueError("sample weights must be a non-empty vector")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample weights must be finite and positive")
    weights = weights / float(weights.mean())
    weights = np.clip(weights, 0.25, 4.0)
    return weights / float(weights.mean())


def _target_weights(targets: np.ndarray, task_type: str) -> np.ndarray:
    values = np.asarray(targets)
    if task_type == "classification":
        labels, counts = np.unique(values, return_counts=True)
        if set(labels.tolist()) != {0, 1}:
            return np.ones(len(values), dtype=float)
        count_by_label = dict(zip(labels.tolist(), counts.tolist(), strict=True))
        return np.asarray(
            [1.0 / math.sqrt(float(count_by_label[int(value)])) for value in values],
            dtype=float,
        )
    if task_type != "regression":
        raise ValueError(f"Unsupported task type {task_type!r}")
    unique = np.unique(values.astype(float))
    bins = min(5, max(2, len(unique)))
    try:
        labels = pd.qcut(values.astype(float), q=bins, labels=False, duplicates="drop")
    except ValueError:
        return np.ones(len(values), dtype=float)
    labels = np.asarray(labels, dtype=float)
    if not np.isfinite(labels).all():
        return np.ones(len(values), dtype=float)
    _, counts = np.unique(labels, return_counts=True)
    count_by_bin = dict(zip(np.unique(labels).tolist(), counts.tolist(), strict=True))
    return np.asarray(
        [1.0 / math.sqrt(float(count_by_bin[float(value)])) for value in labels],
        dtype=float,
    )


def native_training_plan(
    frame: pd.DataFrame,
    *,
    task_type: str,
    native_policy: str,
    weight_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return source positions, curated labels, normalized weights and audit.

    Classification identities with conflicting labels are removed.  Exact
    replicates are collapsed to one identity.  Regression replicates are
    collapsed with their median target.  Optional valuation weights reward
    rare scaffolds and/or underrepresented target regions without using any
    held-out outcomes.
    """

    if native_policy not in NATIVE_POLICIES:
        raise ValueError(f"Unknown native policy {native_policy!r}")
    if weight_policy not in WEIGHT_POLICIES:
        raise ValueError(f"Unknown weight policy {weight_policy!r}")
    required = {"standard_smiles", "standard_inchikey", "Y"}
    if missing := required - set(frame):
        raise ValueError(f"Missing native training columns: {sorted(missing)}")
    work = frame.reset_index(drop=True).copy()
    work["_position"] = np.arange(len(work), dtype=int)
    work["Y"] = pd.to_numeric(work["Y"], errors="raise")
    input_rows = len(work)
    duplicate_rows = int(input_rows - work["standard_inchikey"].nunique())
    conflicting_identities = 0

    if native_policy == "curated":
        grouped = work.groupby("standard_inchikey", sort=True)["Y"]
        if task_type == "classification":
            conflicts = set(grouped.nunique()[lambda value: value > 1].index)
            conflicting_identities = len(conflicts)
            if conflicts:
                work = work.loc[~work["standard_inchikey"].isin(conflicts)].copy()
            work = work.drop_duplicates("standard_inchikey", keep="first")
        elif task_type == "regression":
            medians = grouped.median()
            work = work.drop_duplicates("standard_inchikey", keep="first").copy()
            work["Y"] = work["standard_inchikey"].map(medians).astype(float)
        else:
            raise ValueError(f"Unsupported task type {task_type!r}")

    positions = work["_position"].to_numpy(dtype=int)
    targets = work["Y"].to_numpy(
        dtype=int if task_type == "classification" else float
    )
    weights = np.ones(len(work), dtype=float)
    scaffold_count = 0
    if weight_policy in {"scaffold_balance", "combined"}:
        scaffolds = work["standard_smiles"].astype(str).map(molecular_scaffold)
        counts = scaffolds.value_counts()
        scaffold_count = int(len(counts))
        weights *= np.asarray(
            [1.0 / math.sqrt(float(counts[value])) for value in scaffolds],
            dtype=float,
        )
    if weight_policy in {"target_balance", "combined"}:
        weights *= _target_weights(targets, task_type)
    weights = _normalized_clipped_weights(weights)
    audit = {
        "native_policy": native_policy,
        "weight_policy": weight_policy,
        "input_rows": input_rows,
        "selected_rows": len(work),
        "duplicate_rows_collapsed": (
            duplicate_rows if native_policy == "curated" else 0
        ),
        "conflicting_identities_removed": conflicting_identities,
        "unique_identities": int(work["standard_inchikey"].nunique()),
        "unique_scaffolds": scaffold_count,
        "sample_weight_min": float(weights.min()),
        "sample_weight_max": float(weights.max()),
        "sample_weight_mean": float(weights.mean()),
        "heldout_labels_used": False,
    }
    return positions, targets, weights, audit


__all__ = [
    "NATIVE_POLICIES",
    "WEIGHT_POLICIES",
    "native_training_plan",
]
