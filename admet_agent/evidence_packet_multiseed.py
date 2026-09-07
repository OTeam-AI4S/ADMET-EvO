"""Selection and summaries for multi-seed Evidence Packet experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

from .adaptive_external_data import (
    molecular_scaffold,
    select_relevance_diverse_binary_coreset,
    select_relevance_diverse_regression_coreset,
)

DEVELOPMENT_SEEDS = (11, 23, 42, 67, 89)
CONFIRMATION_SEEDS = (101, 131, 151)
REPAIR_CONFIRMATION_SEEDS = (181, 211, 241)
WEIGHT_GRID = (0.25, 0.50, 0.75, 1.00)
QUANTITY_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def _standard_fields(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.reset_index(drop=True).copy()
    molecules = [Chem.MolFromSmiles(str(value)) for value in work["Drug"]]
    if any(molecule is None for molecule in molecules):
        raise ValueError("prepared Evidence Packet contains an invalid structure")
    work["standard_smiles"] = [
        Chem.MolToSmiles(molecule, canonical=True) for molecule in molecules
    ]
    work["standard_inchikey"] = [
        Chem.MolToInchiKey(molecule) for molecule in molecules
    ]
    work["scaffold"] = work["standard_smiles"].map(molecular_scaffold)
    return work


def quantity_rows(total_rows: int, fraction: float) -> int:
    if total_rows < 1:
        raise ValueError("total_rows must be positive")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("quantity fraction must lie in (0, 1]")
    return min(total_rows, max(1, round(total_rows * fraction)))


def reconcile_external_records(
    frame: pd.DataFrame, *, task_type: str
) -> tuple[pd.DataFrame, dict]:
    """Collapse cross-paper duplicate identities and remove label conflicts."""

    required = {
        "Drug",
        "Y",
        "source_record_id",
        "evidence_packet_id",
        "evidence_base_weight",
    }
    if not required.issubset(frame):
        raise ValueError(f"missing external columns: {sorted(required - set(frame))}")
    work = _standard_fields(frame)
    weights = pd.to_numeric(work["evidence_base_weight"], errors="coerce")
    if (
        weights.isna().any()
        or not weights.between(0.0, 1.0, inclusive="neither").all()
    ):
        raise ValueError("evidence_base_weight must be finite and lie in (0, 1)")
    work["evidence_base_weight"] = weights.astype(float)
    work = work.sort_values(
        ["standard_inchikey", "evidence_packet_id", "source_record_id"],
        kind="stable",
    ).reset_index(drop=True)
    conflicts: set[str] = set()
    conflicting_rows_removed = 0
    if task_type == "classification":
        counts = work.groupby("standard_inchikey")["Y"].nunique()
        conflicts = set(counts[counts > 1].index)
        conflicting_rows_removed = int(
            work["standard_inchikey"].isin(conflicts).sum()
        )
        safe = work.loc[~work["standard_inchikey"].isin(conflicts)].copy()
        output = safe.drop_duplicates("standard_inchikey", keep="first").copy()
        maximum_weights = safe.groupby("standard_inchikey")[
            "evidence_base_weight"
        ].max()
        output["evidence_base_weight"] = output["standard_inchikey"].map(
            maximum_weights
        )
    elif task_type == "regression":
        rows = []
        for _, group in work.groupby("standard_inchikey", sort=True):
            row = group.iloc[0].copy()
            row["Y"] = float(group["Y"].median())
            row["source_record_id"] = "|".join(
                sorted(group["source_record_id"].astype(str))
            )
            row["evidence_packet_id"] = "|".join(
                sorted(set(group["evidence_packet_id"].astype(str)))
            )
            row["evidence_base_weight"] = float(
                group["evidence_base_weight"].max()
            )
            rows.append(row)
        output = pd.DataFrame(rows, columns=work.columns)
    else:
        raise ValueError(f"unsupported task_type: {task_type}")
    output = output.reset_index(drop=True)
    return output, {
        "input_rows": len(frame),
        "cross_packet_conflicting_identity_count": len(conflicts),
        "conflicting_identity_rows_removed": conflicting_rows_removed,
        "duplicate_identity_rows_collapsed": int(
            len(frame) - conflicting_rows_removed - len(output)
        ),
        "output_rows": len(output),
    }


def effective_external_sample_weights(
    frame: pd.DataFrame, positions: np.ndarray, multiplier: float
) -> np.ndarray:
    """Combine packet reliability with a searched global multiplier."""

    if "evidence_base_weight" not in frame:
        raise ValueError("external frame is missing evidence_base_weight")
    if not math.isfinite(multiplier) or not 0 < multiplier <= 1:
        raise ValueError("external weight multiplier must lie in (0, 1]")
    selected = pd.to_numeric(
        frame.iloc[np.asarray(positions, dtype=int)]["evidence_base_weight"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(selected).all()
        or np.any(selected <= 0)
        or np.any(selected >= 1)
    ):
        raise ValueError("evidence_base_weight must be finite and lie in (0, 1)")
    return selected * float(multiplier)


def select_external_positions(
    frame: pd.DataFrame,
    *,
    native_targets: np.ndarray,
    task_type: str,
    total_rows: int,
) -> tuple[np.ndarray, dict]:
    """Select a deterministic scaffold-diverse subset without validation labels."""

    work = _standard_fields(frame)
    relevance = (
        pd.to_numeric(work["evidence_base_weight"], errors="coerce").to_numpy(
            dtype=float
        )
        if "evidence_base_weight" in work
        else np.ones(len(work), dtype=float)
    )
    native = np.asarray(native_targets)
    if task_type == "classification":
        selected, audit = select_relevance_diverse_binary_coreset(
            work,
            relevance,
            total_rows=total_rows,
            target_positive_rate=float(native.mean()),
            minimum_relevance=0.0,
        )
    elif task_type == "regression":
        selected, audit = select_relevance_diverse_regression_coreset(
            work,
            relevance,
            total_rows=total_rows,
            reference_targets=native.astype(float),
            minimum_relevance=0.0,
        )
    else:
        raise ValueError(f"unsupported task_type: {task_type}")
    return selected, {
        **audit,
        "selection_uses_validation_labels": False,
        "selection_uses_test_labels": False,
        "selection_rule": "class/quantile-matched scaffold-diverse fixed ranking",
    }


def pooled_normalized_improvement(
    metric: str, baseline_scores, challenger_scores
) -> tuple[float, np.ndarray]:
    """Normalize the ratio of mean scores, never the mean of seed-wise ratios."""

    baseline = np.asarray(baseline_scores, dtype=float)
    challenger = np.asarray(challenger_scores, dtype=float)
    if baseline.shape != challenger.shape or baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("score arrays must be nonempty one-dimensional peers")
    if not np.isfinite(baseline).all() or not np.isfinite(challenger).all():
        raise ValueError("score arrays must contain only finite values")
    directional = baseline - challenger if metric == "mae" else challenger - baseline
    denominator = max(abs(float(baseline.mean())), 1e-12)
    normalized_seed_deltas = directional / denominator
    return float(normalized_seed_deltas.mean()), normalized_seed_deltas


def summarize_grid(
    rows: Iterable[dict],
    *,
    minimum_positive_seed_fraction: float = 0.60,
) -> tuple[list[dict], dict]:
    """Summarize all fixed seeds and choose a low-mass one-SE configuration."""

    frame = pd.DataFrame(list(rows))
    required = {"weight", "quantity_fraction", "normalized_improvement", "seed"}
    if frame.empty or not required.issubset(frame):
        raise ValueError("grid rows are empty or incomplete")
    summaries: list[dict] = []
    for (weight, fraction), group in frame.groupby(
        ["weight", "quantity_fraction"], sort=True
    ):
        score_columns = {"metric", "baseline_score", "challenger_score"}
        if score_columns.issubset(group.columns):
            metrics = set(group["metric"].astype(str))
            if len(metrics) != 1:
                raise ValueError("each configuration must use one metric")
            pooled_mean, values = pooled_normalized_improvement(
                metrics.pop(),
                group["baseline_score"].to_numpy(dtype=float),
                group["challenger_score"].to_numpy(dtype=float),
            )
            aggregation = "ratio_of_mean_scores"
        else:
            values = group["normalized_improvement"].to_numpy(dtype=float)
            pooled_mean = float(values.mean())
            aggregation = "precomputed_seed_values"
        ordered = np.sort(values)[::-1]
        top = ordered[: min(3, len(ordered))]
        effective_mass = (
            float(group["external_to_native_mass_ratio"].mean())
            if "external_to_native_mass_ratio" in group
            else float(weight * fraction)
        )
        summaries.append(
            {
                "weight": float(weight),
                "quantity_fraction": float(fraction),
                "seed_count": len(values),
                "mean_normalized_improvement": pooled_mean,
                "normalized_improvement_aggregation": aggregation,
                "std_normalized_improvement": float(values.std(ddof=0)),
                "standard_error": float(values.std(ddof=0) / math.sqrt(len(values))),
                "positive_seed_fraction": float((values > 0).mean()),
                "top3_mean_diagnostic_only": float(top.mean()),
                "top3_is_primary_selection_statistic": False,
                "effective_external_mass_fraction": effective_mass,
            }
        )
    eligible = [
        item
        for item in summaries
        if item["mean_normalized_improvement"] > 0
        and item["positive_seed_fraction"] >= minimum_positive_seed_fraction
    ]
    if not eligible:
        return summaries, {
            "status": "no_robust_positive_configuration",
            "selected": None,
            "primary_statistic": "mean over every preregistered development seed",
            "top3_role": "diagnostic_only",
        }
    best = max(eligible, key=lambda item: item["mean_normalized_improvement"])
    floor = best["mean_normalized_improvement"] - best["standard_error"]
    stability_floor = max(
        minimum_positive_seed_fraction,
        best["positive_seed_fraction"],
    )
    within_one_se = [
        item
        for item in eligible
        if item["mean_normalized_improvement"] >= floor
        and item["positive_seed_fraction"] >= stability_floor
    ]
    selected = min(
        within_one_se,
        key=lambda item: (
            item["effective_external_mass_fraction"],
            item["quantity_fraction"],
            item["weight"],
        ),
    )
    return summaries, {
        "status": "selected",
        "selected": {
            "weight": selected["weight"],
            "quantity_fraction": selected["quantity_fraction"],
        },
        "best_mean_configuration": {
            "weight": best["weight"],
            "quantity_fraction": best["quantity_fraction"],
            "mean_normalized_improvement": best["mean_normalized_improvement"],
        },
        "one_standard_error_floor": float(floor),
        "minimum_positive_seed_fraction": float(minimum_positive_seed_fraction),
        "selection_positive_seed_fraction_floor": float(stability_floor),
        "primary_statistic": "mean over every preregistered development seed",
        "top3_role": "diagnostic_only",
    }


__all__ = [
    "CONFIRMATION_SEEDS",
    "DEVELOPMENT_SEEDS",
    "QUANTITY_FRACTIONS",
    "REPAIR_CONFIRMATION_SEEDS",
    "WEIGHT_GRID",
    "effective_external_sample_weights",
    "pooled_normalized_improvement",
    "quantity_rows",
    "reconcile_external_records",
    "select_external_positions",
    "summarize_grid",
]
