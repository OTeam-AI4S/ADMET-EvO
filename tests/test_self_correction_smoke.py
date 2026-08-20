from scripts.run_self_correction_smoke import _specs


def _plan():
    action = {
        "action_id": "model-imbalance-aware--correct-1",
        "axis": "model",
        "payload": {"candidate_model": "balanced_extra_trees"},
        "corrects_action_id": "old-action",
    }
    return {"selected_action_id": action["action_id"], "actions": [action]}


def test_corrective_smoke_materializes_one_axis_2d_pair_only():
    treatment, control = _specs("hERG", _plan(), seed=2, imbalance="none", n_jobs=4)
    assert treatment.model == "balanced_extra_trees"
    assert control.model == "extra_trees"
    assert treatment.representations == control.representations
    assert treatment.representations == ("morgan", "avalon", "erg", "rdkit2d")
    assert treatment.hyperparameters == control.hyperparameters == {}
    assert treatment.data_policy == control.data_policy == "official"
    assert treatment.seed == control.seed == 2
    assert treatment.imbalance == control.imbalance == "none"
    assert "unimol2" not in treatment.representations
