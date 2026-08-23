import hashlib
import json

import numpy as np
import pandas as pd

from scripts.fit_openadmet_cyp_oof_stack import fit_stack


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component(tmp_path, label, offset):
    rows = []
    source_row = 0
    for fold in range(5):
        for endpoint, task_type in (("reg", "regression"), ("cls", "classification")):
            for sample in range(4):
                target = float(sample) if task_type == "regression" else float(sample % 2)
                ideal = target if task_type == "regression" else (0.9 if target else 0.1)
                for seed in (1, 2, 3):
                    rows.append(
                        {
                            "inchikey": f"id-{fold}-{endpoint}-{sample}",
                            "endpoint_key": endpoint,
                            "source_table": endpoint,
                            "source_row": source_row,
                            "outer_fold_evaluated": fold,
                            "chemistry_group": f"g-{fold}-{endpoint}-{sample}",
                            "fidelity": "primary_direct" if task_type == "regression" else "primary_classification",
                            "task_type": task_type,
                            "target": target,
                            "credible_low": target,
                            "credible_high": target,
                            "training_seed": seed,
                            "prediction": ideal + offset,
                            "predictive_std": 0.1,
                        }
                    )
                source_row += 1
    prediction = tmp_path / f"{label}.csv"
    pd.DataFrame(rows).to_csv(prediction, index=False)
    result = tmp_path / f"{label}.json"
    result.write_text(
        json.dumps(
            {
                "training_only": True,
                "blind_test_opened": False,
                "sealed_certification_opened": False,
                "outer_folds": [0, 1, 2, 3, 4],
                "seeds": [1, 2, 3],
                "prediction_path": str(prediction),
                "prediction_sha256": _sha256(prediction),
            }
        )
    )
    return result


def test_nested_oof_stack_favors_better_frozen_component(tmp_path):
    weak = _component(tmp_path, "B1", 0.4)
    strong = _component(tmp_path, "M2", 0.0)
    result = fit_stack({"B1": weak, "M2": strong}, tmp_path / "stack")
    assert result["training_only"] is True
    assert result["blind_test_opened"] is False
    assert len(result["nested_fold_results"]) == 10
    regression = result["final_all_oof_weights_for_future_frozen_refit"]["regression"]
    assert regression["M2"] > regression["B1"]
    classification = result["final_all_oof_weights_for_future_frozen_refit"]["classification"]
    assert classification["M2"] >= classification["B1"]
    assert np.isclose(regression["B1"] + regression["M2"], 1.0)
