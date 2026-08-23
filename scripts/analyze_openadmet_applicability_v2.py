"""Record-key compatibility adapter for frozen applicability analysis v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import analyze_openadmet_applicability_v1 as v1


def collapse_predictions(
    predictions: pd.DataFrame, expected_seeds: list[int]
) -> pd.DataFrame:
    """Preserve distinct source records before collapsing training seeds."""
    keys = [
        "inchikey",
        "endpoint_key",
        "source_table",
        "source_row",
        "standard_smiles",
        "chemistry_group",
        "task_type",
        "target",
        "credible_low",
        "credible_high",
        "outer_fold_evaluated",
    ]
    records = []
    for values, group in predictions.groupby(keys, dropna=False, sort=True):
        observed = sorted(group["training_seed"].astype(int).unique().tolist())
        if observed != expected_seeds or len(group) != len(expected_seeds):
            raise RuntimeError(f"Incomplete seed schedule for {values[:4]}: {observed}")
        row = dict(zip(keys, values, strict=True))
        values_array = group["prediction"].to_numpy(dtype=float)
        row.update(
            prediction=float(np.mean(values_array)),
            disagreement=float(np.std(values_array, ddof=1)),
        )
        records.append(row)
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("reports/openadmet_cyp_applicability_protocol_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_cyp_applicability_analysis_v2.json"),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    v1._collapse_predictions = collapse_predictions
    payload = v1.run(args)
    payload["version"] = "openadmet-cyp-applicability-v2"
    payload["compatibility_adapter_scope"] = (
        "add_source_table_and_source_row_to_seed_collapse_key_only"
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
