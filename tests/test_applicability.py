import numpy as np
import pandas as pd

from admet_agent.applicability import (
    combined_uncertainty,
    cross_conformal_by_similarity,
    error_detection,
    interval_aware_loss,
    percentile_rank,
    risk_coverage,
    similarity_quartiles,
)
from scripts.analyze_openadmet_applicability_v2 import collapse_predictions
from scripts.analyze_openadmet_applicability_v3 import nearest_train_similarity


def test_interval_aware_loss_respects_bounds():
    loss = interval_aware_loss(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 3.0, 1.0]),
        np.array([0.9, 1.8, 2.8]),
        np.array([1.2, 2.2, 3.2]),
    )
    assert np.allclose(loss, [0.0, 0.8, 1.8])


def test_combined_uncertainty_is_rank_scaled_and_label_free():
    output = combined_uncertainty(
        distance=np.array([0.9, 0.1, 0.5]),
        disagreement=np.array([0.8, 0.2, 0.4]),
    )
    assert np.allclose(output, [1.0, 0.0, 0.5])
    assert np.allclose(percentile_rank(np.ones(3)), [0.5, 0.5, 0.5])


def test_risk_coverage_and_detection_reward_correct_order():
    loss = np.array([0.0, 0.1, 0.8, 1.0])
    uncertainty = np.array([0.0, 0.2, 0.7, 1.0])
    risk = risk_coverage(loss, uncertainty, coverages=(0.5, 1.0))
    assert risk["aurc"] < risk["full_risk"]
    detection = error_detection(loss, uncertainty)
    assert detection["auroc"] == 1.0


def test_similarity_quartiles_and_cross_conformal():
    assert similarity_quartiles(np.arange(8)).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    frame = pd.DataFrame(
        {
            "target": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "prediction": [0.1, 0.9, 2.1, 2.9, 4.1, 4.9],
            "outer_fold_evaluated": [0, 0, 1, 1, 2, 2],
            "similarity_quartile": [0, 1, 2, 3, 0, 3],
            "scale": [1.0] * 6,
        }
    )
    result = cross_conformal_by_similarity(frame, scale_column="scale")
    assert result["pooled_coverage"] == 1.0
    assert result["similarity_quartiles"]["0"]["rows"] == 2


def test_record_key_adapter_preserves_distinct_source_rows():
    rows = []
    for source_row in (10, 11):
        for seed in (1, 2, 3):
            rows.append(
                {
                    "inchikey": "KEY",
                    "endpoint_key": "EP",
                    "source_table": "direct",
                    "source_row": source_row,
                    "standard_smiles": "CC",
                    "chemistry_group": "g",
                    "task_type": "regression",
                    "target": 1.0,
                    "credible_low": 0.9,
                    "credible_high": 1.1,
                    "outer_fold_evaluated": 0,
                    "training_seed": seed,
                    "prediction": 1.0 + seed / 100,
                }
            )
    collapsed = collapse_predictions(pd.DataFrame(rows), [1, 2, 3])
    assert len(collapsed) == 2
    assert sorted(collapsed["source_row"].tolist()) == [10, 11]


def test_rdkit_adapter_uses_other_outer_folds_only():
    frame = pd.DataFrame(
        {
            "standard_smiles": ["CC", "CCC", "c1ccccc1", "c1ccncc1"],
            "outer_fold_evaluated": [0, 0, 1, 1],
        }
    )
    similarity = nearest_train_similarity(frame)
    assert similarity.shape == (4,)
    assert np.isfinite(similarity).all()
    assert (similarity >= 0).all() and (similarity <= 1).all()
