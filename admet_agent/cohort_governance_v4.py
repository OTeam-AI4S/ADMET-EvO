"""Identity/scaffold component locking for public-cohort fold assignment."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from admet_agent.cohort_governance import assign_identity_locked_folds as _assign


def assign_identity_locked_folds(
    identities: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 20260823,
) -> dict[str, int]:
    """Keep every identity and every scaffold-connected component indivisible."""
    required = {"inchikey", "chemistry_group"}
    if missing := required - set(identities):
        raise ValueError(f"Missing identity/group columns: {sorted(missing)}")
    pairs = identities[["inchikey", "chemistry_group"]].astype(str).drop_duplicates()
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        parent[merge] = keep

    by_identity = defaultdict(list)
    for row in pairs.itertuples(index=False):
        by_identity[str(row.inchikey)].append(str(row.chemistry_group))
    for groups in by_identity.values():
        for group in groups[1:]:
            union(groups[0], group)
    component_rows = [
        {"inchikey": identity, "chemistry_group": f"component:{find(groups[0])}"}
        for identity, groups in sorted(by_identity.items())
    ]
    return _assign(pd.DataFrame(component_rows), folds=folds, seed=seed)


__all__ = ["assign_identity_locked_folds"]
