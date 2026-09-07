"""Pure policy helpers for the Evidence Packet v5.4 data-axis experiment."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def cap_external_weights(
    raw_weights: np.ndarray,
    *,
    native_rows: int,
    maximum_mass_ratio: float,
) -> tuple[np.ndarray, dict]:
    """Keep native rows at weight one and cap total external influence."""

    values = np.asarray(raw_weights, dtype=float)
    if native_rows < 1:
        raise ValueError("native_rows must be positive")
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("raw external weights must be a finite nonempty vector")
    if np.any(values <= 0) or np.any(values >= 1):
        raise ValueError("every external row must have weight in (0, 1)")
    if not math.isfinite(maximum_mass_ratio) or not 0 < maximum_mass_ratio <= 1:
        raise ValueError("maximum_mass_ratio must lie in (0, 1]")
    raw_mass = float(values.sum())
    cap = float(native_rows * maximum_mass_ratio)
    scale = min(1.0, cap / raw_mass)
    capped = values * scale
    return capped, {
        "native_rows": int(native_rows),
        "native_total_weight": float(native_rows),
        "raw_external_total_weight": raw_mass,
        "external_total_weight": float(capped.sum()),
        "maximum_external_to_native_mass_ratio": float(maximum_mass_ratio),
        "observed_external_to_native_mass_ratio": float(capped.sum() / native_rows),
        "mass_cap_applied": bool(scale < 1.0),
        "mass_cap_scale": float(scale),
        "maximum_external_row_weight": float(capped.max()),
        "native_rows_have_strictly_greater_weight": bool(capped.max() < 1.0),
    }


def summarize_source_grid(
    rows: Iterable[dict],
    *,
    minimum_positive_seed_fraction: float = 0.80,
) -> tuple[list[dict], dict]:
    """Choose a stable low-mass source/quantity/weight configuration."""

    frame = pd.DataFrame(list(rows))
    required = {
        "source_set_id",
        "weight",
        "quantity_fraction",
        "normalized_improvement",
        "external_to_native_mass_ratio",
        "seed",
    }
    if frame.empty or not required.issubset(frame):
        raise ValueError("development rows are empty or incomplete")
    summaries: list[dict] = []
    keys = ["source_set_id", "weight", "quantity_fraction"]
    for (source_set_id, weight, fraction), group in frame.groupby(keys, sort=True):
        values = group["normalized_improvement"].to_numpy(dtype=float)
        if len(set(group["seed"])) != len(values):
            raise ValueError("a configuration contains duplicate seeds")
        summaries.append(
            {
                "source_set_id": str(source_set_id),
                "weight": float(weight),
                "quantity_fraction": float(fraction),
                "seed_count": int(len(values)),
                "mean_normalized_improvement": float(values.mean()),
                "std_normalized_improvement": float(values.std(ddof=0)),
                "standard_error": float(values.std(ddof=0) / math.sqrt(len(values))),
                "positive_seed_fraction": float((values > 0).mean()),
                "effective_external_mass_fraction": float(
                    group["external_to_native_mass_ratio"].mean()
                ),
                "top3_mean_diagnostic_only": float(
                    np.sort(values)[::-1][: min(3, len(values))].mean()
                ),
                "top3_is_primary_selection_statistic": False,
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
            "minimum_positive_seed_fraction": float(minimum_positive_seed_fraction),
            "primary_statistic": "mean over every preregistered development seed",
            "top3_role": "diagnostic_only",
        }
    best = max(eligible, key=lambda item: item["mean_normalized_improvement"])
    floor = best["mean_normalized_improvement"] - best["standard_error"]
    stable = [
        item
        for item in eligible
        if item["mean_normalized_improvement"] >= floor
        and item["positive_seed_fraction"] >= best["positive_seed_fraction"]
    ]
    selected = min(
        stable,
        key=lambda item: (
            item["effective_external_mass_fraction"],
            item["quantity_fraction"],
            item["weight"],
            item["source_set_id"],
        ),
    )
    return summaries, {
        "status": "selected",
        "selected": {
            "source_set_id": selected["source_set_id"],
            "weight": selected["weight"],
            "quantity_fraction": selected["quantity_fraction"],
        },
        "best_mean_configuration": {
            "source_set_id": best["source_set_id"],
            "weight": best["weight"],
            "quantity_fraction": best["quantity_fraction"],
            "mean_normalized_improvement": best["mean_normalized_improvement"],
        },
        "one_standard_error_floor": float(floor),
        "minimum_positive_seed_fraction": float(minimum_positive_seed_fraction),
        "primary_statistic": "mean over every preregistered development seed",
        "top3_role": "diagnostic_only",
    }


def independent_gate(
    values: Iterable[float],
    *,
    minimum_mean: float,
    minimum_positive_fraction: float,
) -> dict:
    observed = np.asarray(list(values), dtype=float)
    if observed.ndim != 1 or observed.size < 1 or not np.isfinite(observed).all():
        raise ValueError("gate values must be a finite nonempty vector")
    mean = float(observed.mean())
    fraction = float((observed > 0).mean())
    return {
        "mean_normalized_improvement": mean,
        "std_normalized_improvement": float(observed.std(ddof=0)),
        "positive_seed_fraction": fraction,
        "minimum_mean_normalized_improvement": float(minimum_mean),
        "minimum_positive_seed_fraction": float(minimum_positive_fraction),
        "passed": bool(mean >= minimum_mean and fraction >= minimum_positive_fraction),
    }


__all__ = ["cap_external_weights", "independent_gate", "summarize_source_grid"]
