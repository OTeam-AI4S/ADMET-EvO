"""Frozen seed-42 scaffold split used by the Evidence Packet data-only loop.

The grouping, shuffle and greedy-fill algorithm is intentionally identical to
agent_core.harness.splits.scaffold_partition_indices at audited
Auto-Research-Molprop commit 91bf92a6f0f8f8bab5823a7018349a2e0db2a92a.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

REFERENCE_SPLIT_SEED = 42
REFERENCE_VALIDATION_FRACTION = 0.20


def scaffold_partition_indices(
    smiles: Sequence[Any],
    fraction: float = REFERENCE_VALIDATION_FRACTION,
    seed: int = REFERENCE_SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Return (train positions, validation positions) without scaffold overlap."""

    if not 0 < fraction < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    n_rows = len(smiles)
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        generator = np.random.default_rng(seed)
        scaffolds: dict[str, list[int]] = {}
        for index, raw_smiles in enumerate(smiles):
            try:
                molecule = Chem.MolFromSmiles(str(raw_smiles))
                scaffold = (
                    MurckoScaffold.MurckoScaffoldSmiles(
                        mol=molecule, includeChirality=False
                    )
                    if molecule
                    else ""
                )
            except Exception:  # noqa: BLE001 - byte-identical reference fallback
                scaffold = ""
            scaffolds.setdefault(scaffold, []).append(index)

        groups = list(scaffolds.values())
        generator.shuffle(groups)
        requested_validation_rows = int(n_rows * fraction)
        validation: list[int] = []
        train: list[int] = []
        for group in groups:
            if len(validation) < requested_validation_rows:
                validation.extend(group)
            else:
                train.extend(group)
        return train, validation
    except Exception:  # noqa: BLE001 - byte-identical reference fallback
        generator = np.random.default_rng(seed)
        indices = generator.permutation(n_rows)
        requested_validation_rows = int(n_rows * fraction)
        return (
            list(indices[requested_validation_rows:]),
            list(indices[:requested_validation_rows]),
        )


def scaffold_train_validation_split(
    frame: pd.DataFrame,
    *,
    fraction: float = REFERENCE_VALIDATION_FRACTION,
    seed: int = REFERENCE_SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if "Drug" not in frame:
        raise ValueError("scaffold split requires a Drug column")
    train_indices, validation_indices = scaffold_partition_indices(
        frame["Drug"], fraction=fraction, seed=seed
    )
    payload = {
        "algorithm": "auto-research-molprop-random-scaffold-greedy-fill",
        "reference_commit": "91bf92a6f0f8f8bab5823a7018349a2e0db2a92a",
        "seed": int(seed),
        "validation_fraction": float(fraction),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    split_sha256 = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    audit = {**payload, "split_sha256": split_sha256}
    return (
        frame.iloc[train_indices].reset_index(drop=True),
        frame.iloc[validation_indices].reset_index(drop=True),
        audit,
    )


__all__ = [
    "REFERENCE_SPLIT_SEED",
    "REFERENCE_VALIDATION_FRACTION",
    "scaffold_partition_indices",
    "scaffold_train_validation_split",
]
