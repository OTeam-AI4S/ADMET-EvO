"""Deterministic quality/quantity controls for task-conditioned external data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def molecular_scaffold(smiles: str) -> str:
    """Return a deterministic Bemis-Murcko scaffold or the molecule itself."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Coreset input contains an invalid standardized structure")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=False
    )
    return scaffold or Chem.MolToSmiles(molecule, canonical=True)


def select_relevance_diverse_binary_coreset(
    frame: pd.DataFrame,
    relevance: np.ndarray,
    *,
    total_rows: int,
    target_positive_rate: float,
    minimum_relevance: float,
) -> tuple[np.ndarray, dict]:
    """Select a class-matched, scaffold-diverse coreset without validation labels.

    Candidates first pass the relevance threshold. Within each label, the best
    member of every scaffold is considered before a second member of any
    scaffold, while higher train-set relevance breaks ties.
    """

    required = {"Y", "standard_smiles", "standard_inchikey"}
    if not required.issubset(frame):
        raise ValueError(f"Missing coreset columns: {sorted(required - set(frame))}")
    relevance = np.asarray(relevance, dtype=float)
    if len(frame) != len(relevance):
        raise ValueError("Relevance length does not match candidate rows")
    if total_rows < 1:
        raise ValueError("total_rows must be positive")
    if not 0.0 <= target_positive_rate <= 1.0:
        raise ValueError("target_positive_rate must lie in [0, 1]")
    if not 0.0 <= minimum_relevance <= 1.0:
        raise ValueError("minimum_relevance must lie in [0, 1]")

    work = frame.reset_index(drop=True).copy()
    work["_position"] = np.arange(len(work), dtype=int)
    work["_relevance"] = relevance
    if "scaffold" not in work:
        work["scaffold"] = work["standard_smiles"].map(molecular_scaffold)
    eligible = work.loc[
        np.isfinite(work["_relevance"])
        & (work["_relevance"] >= minimum_relevance)
        & work["Y"].isin((0, 1))
    ].copy()
    if eligible.empty:
        raise RuntimeError("No external rows passed the relevance floor")
    budget = min(int(total_rows), len(eligible))
    positive_quota = int(round(budget * target_positive_rate))
    quotas = {1: positive_quota, 0: budget - positive_quota}

    ranked_by_label: dict[int, pd.DataFrame] = {}
    chosen: list[int] = []
    for label in (0, 1):
        label_rows = eligible.loc[eligible["Y"] == label].copy()
        label_rows = label_rows.sort_values(
            ["_relevance", "standard_inchikey"],
            ascending=[False, True],
            kind="stable",
        )
        label_rows["_scaffold_round"] = label_rows.groupby(
            "scaffold", sort=False
        ).cumcount()
        label_rows = label_rows.sort_values(
            ["_scaffold_round", "_relevance", "standard_inchikey"],
            ascending=[True, False, True],
            kind="stable",
        )
        ranked_by_label[label] = label_rows
        chosen.extend(label_rows.head(quotas[label])["_position"].astype(int))

    # If one class cannot fill its quota, fill deterministically from the other.
    chosen_set = set(chosen)
    if len(chosen) < budget:
        remaining = pd.concat(ranked_by_label.values(), ignore_index=True)
        remaining = remaining.loc[~remaining["_position"].isin(chosen_set)]
        remaining = remaining.sort_values(
            ["_scaffold_round", "_relevance", "standard_inchikey"],
            ascending=[True, False, True],
            kind="stable",
        )
        chosen.extend(
            remaining.head(budget - len(chosen))["_position"].astype(int)
        )
    selected = np.asarray(sorted(chosen[:budget]), dtype=int)
    selected_frame = work.iloc[selected]
    selected_relevance = relevance[selected]
    return selected, {
        "requested_rows": int(total_rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "target_positive_rate": float(target_positive_rate),
        "selected_positive_rate": float(selected_frame["Y"].mean()),
        "unique_scaffolds": int(selected_frame["scaffold"].nunique()),
        "mean_train_relevance": float(selected_relevance.mean()),
        "minimum_train_relevance": float(selected_relevance.min()),
    }


def external_mass_capped_weights(
    tdc_rows: int, external_rows: int, *, external_mass_ratio: float = 1.0
) -> np.ndarray:
    """Weight external rows so their total mass is capped relative to TDC."""

    if tdc_rows < 1 or external_rows < 1:
        raise ValueError("Both TDC and external row counts must be positive")
    if not math.isfinite(external_mass_ratio) or external_mass_ratio <= 0:
        raise ValueError("external_mass_ratio must be finite and positive")
    external_weight = external_mass_ratio * tdc_rows / external_rows
    return np.concatenate(
        [
            np.ones(tdc_rows, dtype=float),
            np.full(external_rows, external_weight, dtype=float),
        ]
    )


def select_relevance_diverse_regression_coreset(
    frame: pd.DataFrame,
    relevance: np.ndarray,
    *,
    total_rows: int,
    reference_targets: np.ndarray,
    minimum_relevance: float,
    quantile_bins: int = 5,
) -> tuple[np.ndarray, dict]:
    """Select a scaffold-diverse coreset matched to TDC target quantiles."""

    required = {"Y", "standard_smiles", "standard_inchikey"}
    if not required.issubset(frame):
        raise ValueError(f"Missing coreset columns: {sorted(required - set(frame))}")
    relevance = np.asarray(relevance, dtype=float)
    reference = np.asarray(reference_targets, dtype=float)
    if len(frame) != len(relevance) or len(reference) < 2:
        raise ValueError("Invalid regression coreset inputs")
    if total_rows < 1 or quantile_bins < 2:
        raise ValueError("Regression budget and quantile bin count must be positive")
    edges = np.unique(
        np.quantile(reference, np.linspace(0.0, 1.0, quantile_bins + 1))
    )
    if len(edges) < 3:
        raise RuntimeError("Reference targets do not define multiple quantile bins")
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)

    work = frame.reset_index(drop=True).copy()
    work["_position"] = np.arange(len(work), dtype=int)
    work["_relevance"] = relevance
    work["_target_bin"] = pd.cut(work["Y"], bins=edges, labels=False)
    if "scaffold" not in work:
        work["scaffold"] = work["standard_smiles"].map(molecular_scaffold)
    eligible = work.loc[
        np.isfinite(work["_relevance"])
        & (work["_relevance"] >= minimum_relevance)
        & work["_target_bin"].notna()
    ].copy()
    if eligible.empty:
        raise RuntimeError("No external regression rows passed target/relevance matching")
    budget = min(int(total_rows), len(eligible))
    reference_bins = pd.cut(reference, bins=edges, labels=False)
    proportions = pd.Series(reference_bins).value_counts(normalize=True).sort_index()
    quotas = {
        int(bin_index): int(round(budget * proportion))
        for bin_index, proportion in proportions.items()
    }
    # Rounding is reconciled deterministically in the fill step below.
    ranked_bins: list[pd.DataFrame] = []
    chosen: list[int] = []
    for bin_index in range(len(edges) - 1):
        bin_rows = eligible.loc[eligible["_target_bin"] == bin_index].copy()
        bin_rows = bin_rows.sort_values(
            ["_relevance", "standard_inchikey"],
            ascending=[False, True],
            kind="stable",
        )
        bin_rows["_scaffold_round"] = bin_rows.groupby(
            "scaffold", sort=False
        ).cumcount()
        bin_rows = bin_rows.sort_values(
            ["_scaffold_round", "_relevance", "standard_inchikey"],
            ascending=[True, False, True],
            kind="stable",
        )
        ranked_bins.append(bin_rows)
        chosen.extend(
            bin_rows.head(quotas.get(bin_index, 0))["_position"].astype(int)
        )
    chosen_set = set(chosen)
    if len(chosen) < budget:
        remaining = pd.concat(ranked_bins, ignore_index=True)
        remaining = remaining.loc[~remaining["_position"].isin(chosen_set)]
        remaining = remaining.sort_values(
            ["_scaffold_round", "_relevance", "standard_inchikey"],
            ascending=[True, False, True],
            kind="stable",
        )
        chosen.extend(
            remaining.head(budget - len(chosen))["_position"].astype(int)
        )
    selected = np.asarray(sorted(chosen[:budget]), dtype=int)
    selected_frame = work.iloc[selected]
    selected_relevance = relevance[selected]
    reference_iqr = float(np.quantile(reference, 0.75) - np.quantile(reference, 0.25))
    selected_iqr = float(
        selected_frame["Y"].quantile(0.75) - selected_frame["Y"].quantile(0.25)
    )
    return selected, {
        "requested_rows": int(total_rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "unique_scaffolds": int(selected_frame["scaffold"].nunique()),
        "mean_train_relevance": float(selected_relevance.mean()),
        "minimum_train_relevance": float(selected_relevance.min()),
        "reference_target_median": float(np.median(reference)),
        "selected_target_median": float(selected_frame["Y"].median()),
        "reference_target_iqr": reference_iqr,
        "selected_target_iqr": selected_iqr,
    }


__all__ = [
    "external_mass_capped_weights",
    "molecular_scaffold",
    "select_relevance_diverse_binary_coreset",
    "select_relevance_diverse_regression_coreset",
]
