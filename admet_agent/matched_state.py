"""Deterministic development-only states for causal research-action studies.

The transformations in this module operate on discovery train/validation
frames.  Validation structures may define a label-blind domain-shift state;
validation targets are never read while constructing a state.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .data_governance import canonical_smiles, frame_sha256
from .schemas import Budget, ExperimentSpec

RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")
MATCHED_STATE_NAMES = (
    "native",
    "imbalance_mild",
    "imbalance_severe",
    "domain_shift_mild",
    "domain_shift_severe",
)
ACTION_NAMES = ("baseline", "feature_rich", "imbalance_aware")


def _stable_order(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _canonical_series(frame: pd.DataFrame, smiles_col: str) -> pd.Series:
    values = frame[smiles_col].map(canonical_smiles)
    if values.isna().any():
        raise ValueError("Matched-state construction requires valid SMILES")
    return values


def _max_validation_similarity(
    train_smiles: pd.Series, valid_smiles: pd.Series
) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smiles: str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid canonical SMILES in matched-state input")
        return generator.GetFingerprint(molecule)

    valid_fingerprints = [fingerprint(value) for value in valid_smiles]
    if not valid_fingerprints:
        raise ValueError("Validation frame cannot be empty")
    values = []
    for value in train_smiles:
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint(value), valid_fingerprints
        )
        values.append(max(similarities))
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class MatchedStateAudit:
    state: str
    manipulation_family: str
    severity: float
    input_rows: int
    output_rows: int
    retained_fraction: float
    input_positive_rate: float
    output_positive_rate: float
    median_max_validation_similarity_before: float
    median_max_validation_similarity_after: float
    validation_labels_read_for_state: bool
    input_sha256: str
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def construct_matched_state(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    state: str,
    seed: int,
    smiles_col: str = "Drug",
    target_col: str = "Y",
) -> tuple[pd.DataFrame, MatchedStateAudit]:
    """Return a deterministic classification state and its immutable audit."""

    if state not in MATCHED_STATE_NAMES:
        raise ValueError(f"Unknown matched state: {state}")
    if set(pd.unique(train[target_col])) - {0, 1, 0.0, 1.0}:
        raise ValueError("Matched-state v1 supports binary classification only")
    working = train.copy().reset_index(drop=True)
    train_canonical = _canonical_series(working, smiles_col)
    valid_canonical = _canonical_series(valid, smiles_col)
    similarities = _max_validation_similarity(train_canonical, valid_canonical)
    before_rate = float(pd.to_numeric(working[target_col]).mean())

    family = "native"
    severity = 0.0
    keep = np.ones(len(working), dtype=bool)
    if state.startswith("imbalance_"):
        family = "imbalance"
        retention = 0.5 if state.endswith("mild") else 0.25
        severity = 1.0 - retention
        labels = pd.to_numeric(working[target_col]).to_numpy()
        classes, counts = np.unique(labels, return_counts=True)
        minority = classes[int(np.argmin(counts))]
        minority_indices = np.flatnonzero(labels == minority)
        ordered = sorted(
            minority_indices,
            key=lambda index: _stable_order(train_canonical.iloc[index], seed),
        )
        retain_count = max(20, int(round(len(ordered) * retention)))
        retain = set(ordered[: min(retain_count, len(ordered))])
        keep[minority_indices] = False
        keep[list(retain)] = True
    elif state.startswith("domain_shift_"):
        family = "domain_shift"
        removal = 0.2 if state.endswith("mild") else 0.4
        severity = removal
        labels = pd.to_numeric(working[target_col]).to_numpy()
        # Remove the train molecules closest to validation within each class so
        # the manipulation changes structural support without confounding the
        # class ratio by construction.
        for label in sorted(np.unique(labels)):
            indices = np.flatnonzero(labels == label)
            removal_count = min(
                max(0, int(round(len(indices) * removal))),
                max(0, len(indices) - 20),
            )
            ordered = sorted(
                indices,
                key=lambda index: (
                    -similarities[index],
                    _stable_order(train_canonical.iloc[index], seed),
                ),
            )
            keep[ordered[:removal_count]] = False

    output = working.loc[keep].reset_index(drop=True)
    output_canonical = train_canonical.loc[keep].reset_index(drop=True)
    after_similarity = similarities[keep]
    audit = MatchedStateAudit(
        state=state,
        manipulation_family=family,
        severity=severity,
        input_rows=len(working),
        output_rows=len(output),
        retained_fraction=float(len(output) / len(working)),
        input_positive_rate=before_rate,
        output_positive_rate=float(pd.to_numeric(output[target_col]).mean()),
        median_max_validation_similarity_before=float(np.median(similarities)),
        median_max_validation_similarity_after=float(np.median(after_similarity)),
        validation_labels_read_for_state=False,
        input_sha256=frame_sha256(working),
        output_sha256=frame_sha256(output),
    )
    # Bind the audit to the exact structures used, not only the frame hash.
    if len(output_canonical) != audit.output_rows:
        raise RuntimeError("Matched-state audit row accounting failed")
    return output, audit


def matched_action_spec(
    task: str,
    *,
    action: str,
    seed: int,
    workers: int,
    max_minutes: int,
) -> ExperimentSpec:
    """Create one action from a shared Extra Trees intervention baseline."""

    if action not in ACTION_NAMES:
        raise ValueError(f"Unknown matched action: {action}")
    representations = ("morgan",)
    imbalance = "none"
    if action == "feature_rich":
        representations = RICH_2D
    elif action == "imbalance_aware":
        imbalance = "class_weight"
    return ExperimentSpec(
        task=task,
        seed=seed,
        representations=representations,
        model="extra_trees",
        imbalance=imbalance,
        data_policy="identity_safe",
        device="cpu",
        n_jobs=workers,
        hyperparameters={"n_estimators": 300, "min_samples_leaf": 2},
        budget=Budget(max_candidates=1, max_minutes=max_minutes),
        hypothesis=f"v4 matched-state action={action}",
    )


__all__ = [
    "ACTION_NAMES",
    "MATCHED_STATE_NAMES",
    "MatchedStateAudit",
    "construct_matched_state",
    "matched_action_spec",
]
