import json

import numpy as np

from scripts.prepare_cyp2c9_focal_confirmation import (
    _digest,
    preregistration,
    verify_preregistration,
)
from scripts.train_cyp2c9_focal_confirmation import percentile_ranks


def test_preregistration_is_content_addressed(tmp_path, monkeypatch):
    root = tmp_path
    task_root = root / "runs/v24-minimol-validation/CYP2C9_Substrate_CarbonMangels"
    task_root.mkdir(parents=True)
    (task_root / "rank_heads.json").write_text("{}", encoding="utf-8")
    (task_root / "rank_head_predictions.npz").write_bytes(b"predictions")
    hard = root / "runs/hard-task-ablation/results.json"
    hard.parent.mkdir(parents=True)
    hard.write_text("{}", encoding="utf-8")

    value = preregistration(root)
    payload = {key: item for key, item in value.items() if key != "preregistration_sha256"}
    assert value["preregistration_sha256"] == _digest(payload)
    assert value["candidate"]["focal_minimol_weight"] == 0.2
    assert value["fresh_confirmation_seeds"] == [6, 7, 8, 9, 10]

    path = root / "preregistration.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    verify_preregistration(path, value)


def test_percentile_rank_fusion_is_scale_invariant():
    values = np.array([10.0, -2.0, 5.0, 5.0])
    assert np.allclose(
        percentile_ranks(values), percentile_ranks(7.0 * values + 3.0)
    )
