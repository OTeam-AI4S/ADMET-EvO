import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.analyze_openadmet_cyp_cv import _ece, analyze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ece_is_zero_for_perfect_binary_probabilities():
    assert _ece(
        target=pd.Series([0, 1]).to_numpy(),
        probability=pd.Series([0.0, 1.0]).to_numpy(),
    ) == 0.0


def test_cv_analysis_collapses_seeds_and_reports_cluster_uq(tmp_path):
    rows = []
    for identity_index in range(12):
        fold = identity_index % 3
        for seed in (1, 2):
            base = {
                "inchikey": f"id-{identity_index}",
                "source_table": "direct",
                "source_row": identity_index,
                "outer_fold_evaluated": fold,
                "chemistry_group": f"group-{identity_index}",
                "training_seed": seed,
                "predictive_std": 0.2,
            }
            target = 4.0 + identity_index / 10
            rows.append(
                {
                    **base,
                    "endpoint_key": "CYP2D6|direct_inhibition|pIC50|primary_direct",
                    "fidelity": "primary_direct",
                    "task_type": "regression",
                    "target": target,
                    "credible_low": target - 0.1,
                    "credible_high": target + 0.1,
                    "prediction": target + (0.05 if seed == 1 else -0.05),
                }
            )
            label = identity_index % 2
            rows.append(
                {
                    **base,
                    "endpoint_key": "CYP2D6|TDI_shift|is_TDI|primary_classification",
                    "fidelity": "primary_classification",
                    "task_type": "classification",
                    "target": label,
                    "credible_low": float("nan"),
                    "credible_high": float("nan"),
                    "prediction": 0.9 if label else 0.1,
                }
            )
    prediction_path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "training_only": True,
                "blind_test_opened": False,
                "prediction_path": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
            }
        )
    )

    output = analyze("fixture", result_path, bootstrap_samples=10, seed=7)
    assert output["collapsed_rows"] == 24
    classification = output["endpoints"][
        "CYP2D6|TDI_shift|is_TDI|primary_classification"
    ]
    assert classification["mcc"] == 1.0
    regression_key = "CYP2D6|direct_inhibition|pIC50|primary_direct"
    assert output["endpoints"][regression_key]["st_rae"] == 0.0
    assert regression_key in output["cluster_bootstrap"]
    assert output["cross_conformal_90"][regression_key]["pooled_coverage"] == 1.0


def test_cluster_bootstrap_handles_unequal_group_sizes_without_frame_concat(tmp_path):
    prediction_path = tmp_path / "predictions.csv"
    rows = []
    for index, group in enumerate(("large", "large", "large", "small")):
        rows.append(
            {
                "inchikey": f"id-{index}",
                "endpoint_key": "CYP2D6|TDI_shift|is_TDI|primary_classification",
                "source_table": "tdi",
                "source_row": index,
                "outer_fold_evaluated": index % 2,
                "chemistry_group": group,
                "fidelity": "primary_classification",
                "task_type": "classification",
                "target": index % 2,
                "credible_low": float("nan"),
                "credible_high": float("nan"),
                "prediction": 0.8 if index % 2 else 0.2,
                "predictive_std": 0.1,
                "training_seed": 1,
            }
        )
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "training_only": True,
                "blind_test_opened": False,
                "prediction_path": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
            }
        )
    )
    output = analyze("fixture", result_path, bootstrap_samples=100, seed=11)
    endpoint = "CYP2D6|TDI_shift|is_TDI|primary_classification"
    assert output["cluster_bootstrap"][endpoint]["mcc"]["ci95"][0] <= 1.0
