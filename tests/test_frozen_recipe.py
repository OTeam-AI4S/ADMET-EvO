import numpy as np
import pandas as pd
import pytest

from admet_agent.frozen_executor import FrozenRecipeExecutor, _set_unimol_training_seed
from admet_agent.frozen_recipe import (
    aggregate_predictions,
    build_execution_plan,
    parse_recipe,
)
from admet_agent.tasks import resolve_task


def test_parse_recipe_normalizes_rounded_weights():
    recipe = parse_recipe(
        "raw(2d_lightgbm@0.33+2d_extra_trees@0.33+finetuned_unimol2@0.33)"
    )

    assert recipe.aggregation == "raw"
    assert recipe.expert_names == (
        "2d_lightgbm",
        "2d_extra_trees",
        "finetuned_unimol2",
    )
    assert [member.weight for member in recipe.members] == pytest.approx([1 / 3] * 3)


def test_rank_aggregation_uses_average_ties():
    recipe = parse_recipe("rank(2d_svm@0.50+finetuned_unimol2@0.50)")
    result = aggregate_predictions(
        recipe,
        {
            "2d_svm": np.asarray([3.0, 1.0, 3.0, 2.0]),
            "finetuned_unimol2": np.asarray([4.0, 3.0, 2.0, 1.0]),
        },
    )

    assert result.tolist() == pytest.approx([0.9375, 0.5, 0.6875, 0.375])


def test_execution_plan_exposes_expensive_and_transformed_experts():
    plan = build_execution_plan(
        resolve_task("Half_Life_Obach"),
        "rank(finetuned_unimol2@0.33+fixed_rank_top3@0.33+rank_target_xgboost@0.33)",
    )

    assert plan["experts"]["finetuned_unimol2"]["requires_inner_oof"] is True
    assert plan["experts"]["fixed_rank_top3"]["backend"] == "dynamic_oof_ensemble"
    assert plan["experts"]["rank_target_xgboost"]["target_transform"] == (
        "percentile_rank"
    )

    kernel = build_execution_plan(
        resolve_task("Bioavailability_Ma"), "minimol_morgan_kernel"
    )
    assert kernel["experts"]["minimol_morgan_kernel"] == {
        "backend": "precomputed_kernel",
        "representations": ["morgan", "morgan_r3", "minimol"],
        "model": "svc_c10_balanced",
        "target_transform": "identity",
        "requires_inner_oof": False,
    }
    focal = build_execution_plan(
        resolve_task("CYP2C9_Substrate_CarbonMangels"),
        "rank(2d_lightgbm@0.80+focal_minimol_16@0.20)",
    )
    assert focal["experts"]["focal_minimol_16"] == {
        "backend": "isolated_torch_head",
        "representations": ["minimol"],
        "model": "focal_mlp_16x100",
        "target_transform": "identity",
        "requires_inner_oof": False,
    }
    maplight_gnn = build_execution_plan(
        resolve_task("Bioavailability_Ma"), "maplight_gnn"
    )
    assert maplight_gnn["experts"]["maplight_gnn"] == {
        "backend": "isolated_maplight_gnn",
        "representations": [
            "morgan_count",
            "avalon_count",
            "erg",
            "rdkit2d",
            "gin_supervised_masking",
        ],
        "model": "catboost_fixed",
        "target_transform": "identity",
        "requires_inner_oof": False,
    }


def test_execution_plan_rejects_rank_for_mae_and_unknown_experts():
    with pytest.raises(ValueError, match="not valid for MAE"):
        build_execution_plan(
            resolve_task("Lipophilicity_AstraZeneca"),
            "rank(2d_lightgbm@0.50+finetuned_unimol2@0.50)",
        )
    with pytest.raises(ValueError, match="unknown experts"):
        build_execution_plan(resolve_task("DILI"), "raw(magic@1.00)")


def test_aggregation_rejects_missing_or_nonfinite_predictions():
    recipe = parse_recipe("raw(2d_lightgbm@0.50+finetuned_unimol2@0.50)")
    with pytest.raises(ValueError, match="Missing"):
        aggregate_predictions(recipe, {"2d_lightgbm": np.ones(3)})
    with pytest.raises(ValueError, match="finite"):
        aggregate_predictions(
            recipe,
            {
                "2d_lightgbm": np.ones(3),
                "finetuned_unimol2": np.asarray([0.1, np.nan, 0.2]),
            },
        )


def test_finetune_only_executor_dry_run_never_receives_prediction_labels(tmp_path):
    calls = {}

    def fake_backend(
        task,
        fit_frame,
        predict_smiles,
        smiles_col,
        y_col,
        model_dir,
        protocol,
        seed,
    ):
        calls["fit_columns"] = fit_frame.columns.tolist()
        calls["predict_smiles"] = predict_smiles
        calls["seed"] = seed
        return np.linspace(0.1, 0.9, len(fit_frame)), np.asarray([0.2, 0.8])

    executor = FrozenRecipeExecutor(
        cache=None,
        artifact_dir=tmp_path / "artifacts",
        device="cpu",
        n_jobs=1,
        finetune_backend=fake_backend,
    )
    fit = pd.DataFrame({"Drug": ["CC", "CCC", "CCCC"], "Y": [0, 1, 0]})
    result = executor.predict(
        resolve_task("DILI"),
        "finetuned_unimol2",
        7,
        fit,
        ["CO", "CN"],
        "Drug",
        "Y",
        {
            "model_size": "84m",
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 16,
            "inner_folds": 5,
        },
    )

    assert result.tolist() == pytest.approx([0.2, 0.8])
    assert calls == {
        "fit_columns": ["Drug", "Y"],
        "predict_smiles": ["CO", "CN"],
        "seed": 7,
    }
    audit = (tmp_path / "artifacts" / "execution_audit.json").read_text()
    assert '"test_labels_passed_to_executor": false' in audit


def test_unimol_runtime_seed_is_written_to_mutable_config():
    class FakeTrainer:
        def __init__(self):
            self.config = {"seed": 42, "split_seed": 42}

    trainer = FakeTrainer()
    _set_unimol_training_seed(trainer, 7)

    assert trainer.config == {"seed": 7, "split_seed": 7}


def test_minimol_morgan_kernel_executor_never_receives_prediction_labels(tmp_path):
    class FakeCache:
        def get_or_compute(self, smiles, representations):
            lookup = {
                "CC": [1, 0, 0, 1],
                "CCC": [1, 1, 0, 0],
                "CO": [0, 1, 1, 0],
                "CN": [0, 0, 1, 1],
                "CCCC": [1, 1, 1, 0],
            }
            values = np.asarray([lookup[value] for value in smiles], dtype=np.float32)
            if representations == ("minimol",):
                return np.pad(values, ((0, 0), (0, 508)))
            return values

    executor = FrozenRecipeExecutor(
        cache=FakeCache(),
        artifact_dir=tmp_path / "artifacts",
        device="cpu",
        n_jobs=1,
    )
    fit = pd.DataFrame(
        {"Drug": ["CC", "CCC", "CO", "CN"], "Y": [0, 0, 1, 1]}
    )
    result = executor.predict(
        resolve_task("Bioavailability_Ma"),
        "minimol_morgan_kernel",
        3,
        fit,
        ["CCCC"],
        "Drug",
        "Y",
        {
            "model_size": "84m",
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 16,
            "inner_folds": 5,
        },
    )

    assert result.shape == (1,)
    audit = (tmp_path / "artifacts" / "execution_audit.json").read_text()
    assert '"test_labels_passed_to_executor": false' in audit
    assert '"chemical_weight": 0.5' in audit


def test_focal_minimol_executor_passes_only_fit_labels_to_backend(tmp_path):
    calls = {}

    class FakeCache:
        def get_or_compute(self, smiles, representations):
            width = 512 if representations == ("minimol",) else 4
            return np.arange(len(smiles) * width, dtype=np.float32).reshape(
                len(smiles), width
            )

    def fake_focal(task, x_fit, y_fit, x_predict, seed, artifact_dir):
        calls.update(
            task=task.name,
            fit_shape=x_fit.shape,
            labels=y_fit.tolist(),
            predict_shape=x_predict.shape,
            seed=seed,
        )
        return np.asarray([0.3, 0.7])

    executor = FrozenRecipeExecutor(
        cache=FakeCache(),
        artifact_dir=tmp_path / "artifacts",
        device="cpu",
        n_jobs=1,
        focal_head_backend=fake_focal,
    )
    fit = pd.DataFrame({"Drug": ["CC", "CCC", "CO"], "Y": [0, 1, 0]})
    result = executor.predict(
        resolve_task("CYP2C9_Substrate_CarbonMangels"),
        "focal_minimol_16",
        4,
        fit,
        ["CN", "CCCC"],
        "Drug",
        "Y",
        {"inner_folds": 5},
    )
    assert result.tolist() == pytest.approx([0.3, 0.7])
    assert calls == {
        "task": "CYP2C9_Substrate_CarbonMangels",
        "fit_shape": (3, 512),
        "labels": [0, 1, 0],
        "predict_shape": (2, 512),
        "seed": 4,
    }
    audit = (tmp_path / "artifacts" / "execution_audit.json").read_text()
    assert '"predict_labels_passed": false' in audit


def test_maplight_gnn_executor_passes_no_prediction_labels(tmp_path):
    calls = {}

    def fake_maplight(
        task,
        fit_smiles,
        y_fit,
        predict_smiles,
        seed,
        workers,
        artifact_dir,
    ):
        calls.update(
            task=task.name,
            fit_smiles=fit_smiles,
            labels=y_fit.tolist(),
            predict_smiles=predict_smiles,
            seed=seed,
            workers=workers,
        )
        return np.asarray([0.25, 0.75])

    executor = FrozenRecipeExecutor(
        cache=None,
        artifact_dir=tmp_path / "artifacts",
        device="cpu",
        n_jobs=3,
        maplight_gnn_backend=fake_maplight,
    )
    fit = pd.DataFrame({"Drug": ["CC", "CCC", "CO"], "Y": [0, 1, 0]})
    result = executor.predict(
        resolve_task("Bioavailability_Ma"),
        "maplight_gnn",
        4,
        fit,
        ["CN", "CCCC"],
        "Drug",
        "Y",
        {"inner_folds": 5},
    )
    assert result.tolist() == pytest.approx([0.25, 0.75])
    assert calls == {
        "task": "Bioavailability_Ma",
        "fit_smiles": ["CC", "CCC", "CO"],
        "labels": [0, 1, 0],
        "predict_smiles": ["CN", "CCCC"],
        "seed": 4,
        "workers": 3,
    }
    audit = (tmp_path / "artifacts" / "execution_audit.json").read_text()
    assert '"predict_labels_passed": false' in audit
    assert '"offline_content_addressed_checkpoint": true' in audit
