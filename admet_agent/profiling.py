"""Task diagnostics computed strictly from train and validation data."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .schemas import TaskProfile
from .tasks import TaskSpec


def _scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)


def profile_task(train, valid, spec: TaskSpec, smiles_col: str = "Drug", target_col: str = "Y") -> TaskProfile:
    train_scaffolds = [_scaffold(value) for value in train[smiles_col].tolist()]
    valid_scaffolds = [_scaffold(value) for value in valid[smiles_col].tolist()]
    known = {value for value in train_scaffolds if value is not None}
    valid_known = [value for value in valid_scaffolds if value is not None]
    unseen_rate = (
        sum(value not in known for value in valid_known) / len(valid_known) if valid_known else 0.0
    )
    targets = train[target_col].to_numpy(dtype=float)
    positive_rate = float(np.mean(targets)) if spec.task_type == "classification" else None
    return TaskProfile(
        task=spec.name,
        task_type=spec.task_type,
        metric=spec.metric,
        n_train=len(train),
        n_valid=len(valid),
        positive_rate_train=positive_rate,
        target_mean=float(np.mean(targets)),
        target_std=float(np.std(targets)),
        invalid_smiles_train=sum(value is None for value in train_scaffolds),
        unique_scaffolds_train=len(known),
        unseen_valid_scaffold_rate=float(unseen_rate),
    )
