import numpy as np
import pytest

from admet_agent.graph_gated_tree import OOFGraphGatedTreeClassifier


def _data():
    rng = np.random.default_rng(4)
    y = np.array([0, 1] * 30)
    return {
        "two_d": np.column_stack([y + rng.normal(0, 0.25, len(y)), rng.normal(size=len(y))]),
        "gin": np.column_stack([rng.normal(size=len(y)), y + rng.normal(0, 0.35, len(y))]),
        "mpnn": rng.normal(size=(len(y), 3)),
    }, y


def test_oof_gate_accepts_three_named_views_and_returns_probabilities():
    views, y = _data()
    model = OOFGraphGatedTreeClassifier(n_estimators=20, folds=3, n_jobs=1).fit(views, y)
    prediction = model.predict_proba(views)
    assert prediction.shape == (len(y), 2)
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert set(model.gate_decision(views)) <= {"two_d", "gin", "mpnn"}
    assert model.audit_.gate_trained_on_oof_predictions is True
    assert model.audit_.test_labels_loaded is False


def test_gate_rejects_missing_or_changed_views():
    views, y = _data()
    with pytest.raises(ValueError, match="at least two"):
        OOFGraphGatedTreeClassifier(n_estimators=10, n_jobs=1).fit({"two_d": views["two_d"]}, y)
    model = OOFGraphGatedTreeClassifier(n_estimators=10, folds=2, n_jobs=1).fit(views, y)
    with pytest.raises(ValueError, match="differ"):
        model.predict_proba({"two_d": views["two_d"], "gin": views["gin"]})


def test_gate_rejects_nonfinite_or_nonbinary_inputs():
    views, y = _data()
    broken = dict(views)
    broken["gin"] = broken["gin"].copy()
    broken["gin"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        OOFGraphGatedTreeClassifier(n_estimators=10, n_jobs=1).fit(broken, y)
    with pytest.raises(ValueError, match="binary"):
        OOFGraphGatedTreeClassifier(n_estimators=10, n_jobs=1).fit(views, np.zeros_like(y))
