import numpy as np
import pytest

from admet_agent.guard import ProtocolViolation, validate_experiment
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict
from admet_agent.tasks import resolve_task


@pytest.mark.parametrize("model", ["xgboost", "lightgbm", "svm_rbf"])
@pytest.mark.parametrize(
    "task_name",
    ["Caco2_Wang", "HIA_Hou"],
)
def test_v21_models_fit_and_predict(model, task_name):
    task = resolve_task(task_name)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 12)).astype(np.float32)
    y = (
        np.linspace(-1, 1, 30)
        if task.task_type == "regression"
        else np.asarray([0, 1] * 15)
    )
    spec = ExperimentSpec(
        task=task.name,
        seed=1,
        model=model,
        representations=("morgan",),
        imbalance="class_weight" if task.task_type == "classification" else "none",
        hyperparameters=(
            {"n_estimators": 100, "max_depth": 3}
            if model == "xgboost"
            else {"n_estimators": 100, "num_leaves": 15}
            if model == "lightgbm"
            else {"C": 1.0, "gamma": "scale", **({"epsilon": 0.1} if task.task_type == "regression" else {})}
        ),
    )
    validate_experiment(spec)
    fitted = build_model(spec, task, np.asarray(y)).fit(x[:24], y[:24])
    values = predict(fitted, x[24:], task)
    assert values.shape == (6,)
    assert np.isfinite(values).all()


def test_v21_guard_rejects_wrong_booster_parameters():
    with pytest.raises(ProtocolViolation, match="invalid xgboost"):
        validate_experiment(ExperimentSpec(
            task="Caco2_Wang", seed=1, model="xgboost",
            hyperparameters={"num_leaves": 31},
        ))
    with pytest.raises(ProtocolViolation, match="regression-only"):
        validate_experiment(ExperimentSpec(
            task="HIA_Hou", seed=1, model="svm_rbf",
            hyperparameters={"epsilon": 0.1},
        ))


def test_balanced_extra_trees_fits_imbalanced_classification():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(80, 16)).astype(np.float32)
    y = np.asarray([0] * 64 + [1] * 16)
    spec = ExperimentSpec(
        task="CYP2C9_Substrate_CarbonMangels",
        seed=3,
        model="balanced_extra_trees",
        representations=("morgan",),
        imbalance="class_weight",
        hyperparameters={
            "n_estimators": 100,
            "n_members": 3,
            "majority_ratio": 1.0,
        },
    )
    task = resolve_task(spec.task)
    validate_experiment(spec)
    fitted = build_model(spec, task, y).fit(x, y)
    values = predict(fitted, x[:7], task)
    assert values.shape == (7,)
    assert np.logical_and(values >= 0.0, values <= 1.0).all()


def test_balanced_extra_trees_is_classification_only():
    with pytest.raises(ProtocolViolation, match="does not support regression"):
        validate_experiment(ExperimentSpec(
            task="Caco2_Wang",
            seed=1,
            model="balanced_extra_trees",
        ))
