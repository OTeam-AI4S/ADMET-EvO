"""Full-data execution backend for preregistered heterogeneous recipes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

from .features import FeatureCache
from .frozen_recipe import EXPERTS, aggregate_predictions, parse_recipe
from .schemas import ExperimentSpec
from .selection import build_model, predict, score
from .tasks import TaskSpec

FineTuneBackend = Callable[
    [TaskSpec, pd.DataFrame, list[str], str, str, Path, dict, int],
    tuple[np.ndarray, np.ndarray],
]
FocalHeadBackend = Callable[
    [TaskSpec, np.ndarray, np.ndarray, np.ndarray, int, Path], np.ndarray
]
MapLightGNNBackend = Callable[
    [TaskSpec, list[str], np.ndarray, list[str], int, int, Path], np.ndarray
]

RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")
BASE_TABULAR = {
    "2d_lightgbm": ("lightgbm", RICH_2D, {}),
    "2d_extra_trees": ("extra_trees", RICH_2D, {}),
    "2d_svm": ("svm_rbf", RICH_2D, {"C": 3.0, "gamma": "scale"}),
    "frozen_unimol2_lightgbm": ("lightgbm", ("unimol2",), {}),
    "frozen_unimol2_extra_trees": ("extra_trees", ("unimol2",), {}),
    "hybrid_unimol2_2d_lightgbm": (
        "lightgbm",
        ("unimol2", *RICH_2D),
        {},
    ),
}
FIXED_TOP3_POOL = (*BASE_TABULAR, "finetuned_unimol2")


def _cross_tanimoto(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = left @ right.T
    union = (
        left.sum(axis=1)[:, None]
        + right.sum(axis=1)[None, :]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float32),
        where=norms > 0,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_focal_head_backend(
    task: TaskSpec,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_predict: np.ndarray,
    seed: int,
    artifact_dir: Path,
) -> np.ndarray:
    """Run the locked 16-head focal MiniMol model in its isolated Torch env."""
    if task.task_type != "classification" or task.metric != "auprc":
        raise ValueError("The focal MiniMol expert is frozen for AUPRC classification")
    repo_root = Path(__file__).resolve().parents[1]
    python = repo_root / ".venvs/minimol/bin/python"
    script = repo_root / "scripts/predict_focal_minimol.py"
    if not python.exists() or not script.exists():
        raise RuntimeError("The isolated focal MiniMol runtime is unavailable")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / "input.npz"
    output_path = artifact_dir / "predictions.npy"
    report_path = artifact_dir / "runtime_report.json"
    np.savez_compressed(
        input_path,
        x_fit=np.asarray(x_fit, dtype=np.float32),
        y_fit=np.asarray(y_fit, dtype=np.float32),
        x_predict=np.asarray(x_predict, dtype=np.float32),
    )
    manifest = {
        "task": task.name,
        "metric": task.metric,
        "seed": int(seed),
        "fit_rows": len(x_fit),
        "predict_rows": len(x_predict),
        "embedding_width": 512,
        "members": 16,
        "epochs": 100,
        "batch_size": 32,
        "loss": "positive-weighted focal BCE",
        "focal_gamma": 2.0,
        "head_seed_formula": "outer_seed*10000+500+member",
        "fit_labels_present": True,
        "predict_labels_present": False,
        "input_sha256": _sha256(input_path),
    }
    manifest_path = artifact_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    completed = subprocess.run(
        [
            str(python),
            str(script),
            "--input",
            str(input_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Isolated focal MiniMol execution failed: " + completed.stderr[-2000:]
        )
    values = np.load(output_path, allow_pickle=False)
    if values.shape != (len(x_predict),) or not np.isfinite(values).all():
        raise RuntimeError("Isolated focal MiniMol returned invalid predictions")
    return np.asarray(values, dtype=float)


def _default_maplight_gnn_backend(
    task: TaskSpec,
    fit_smiles: list[str],
    y_fit: np.ndarray,
    predict_smiles: list[str],
    seed: int,
    workers: int,
    artifact_dir: Path,
) -> np.ndarray:
    """Run the content-addressed MapLight+GNN expert in its isolated env."""
    if task.task_type != "classification":
        raise ValueError("The MapLight+GNN frozen expert supports classification")
    repo_root = Path(__file__).resolve().parents[1]
    python = repo_root / ".venvs/maplight-gnn/bin/python"
    script = repo_root / "scripts/predict_maplight_gnn.py"
    if not python.exists() or not script.exists():
        raise RuntimeError("The isolated MapLight+GNN runtime is unavailable")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / "input.json"
    output_path = artifact_dir / "predictions.npy"
    report_path = artifact_dir / "runtime_report.json"
    input_path.write_text(
        json.dumps(
            {
                "task": task.name,
                "seed": int(seed),
                "workers": int(workers),
                "fit_smiles": fit_smiles,
                "fit_labels": np.asarray(y_fit, dtype=int).tolist(),
                "predict_smiles": predict_smiles,
                "predict_labels_present": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "DGLBACKEND": "pytorch",
            "LD_LIBRARY_PATH": str(repo_root / ".venvs/maplight-gnn/lib"),
            "OMP_NUM_THREADS": str(max(1, workers)),
            "OPENBLAS_NUM_THREADS": str(max(1, workers)),
            "MKL_NUM_THREADS": str(max(1, workers)),
        }
    )
    completed = subprocess.run(
        [
            str(python),
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Isolated MapLight+GNN execution failed: " + completed.stderr[-4000:]
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("predict_labels_present") is not False:
        raise RuntimeError("MapLight+GNN runtime label firewall failed")
    if report.get("input_sha256") != _sha256(input_path):
        raise RuntimeError("MapLight+GNN runtime input hash mismatch")
    values = np.load(output_path, allow_pickle=False)
    if values.shape != (len(predict_smiles),) or not np.isfinite(values).all():
        raise RuntimeError("Isolated MapLight+GNN returned invalid predictions")
    if report.get("predictions_sha256") != _sha256(output_path):
        raise RuntimeError("MapLight+GNN runtime prediction hash mismatch")
    return np.asarray(values, dtype=float)


def _set_unimol_training_seed(trainer: object, seed: int) -> None:
    """Set seeds on MolTrain's config, whose ``**params`` are not persisted.

    unimol-tools 0.1.5 accepts arbitrary keyword arguments in ``MolTrain`` but
    does not copy them into its runtime config.  Mutating the public config
    before ``fit`` is therefore necessary for genuine multi-seed training and
    scaffold splitting.
    """
    config = getattr(trainer, "config", None)
    if config is None:
        raise RuntimeError("MolTrain does not expose a mutable runtime config")
    config["seed"] = int(seed)
    config["split_seed"] = int(seed)


def _scaffolds(smiles: list[str]) -> np.ndarray:
    values = []
    for value in smiles:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=str(value), includeChirality=True
            )
        except (RuntimeError, ValueError):
            scaffold = str(value)
        values.append(scaffold or str(value))
    return np.asarray(values)


def _default_finetune_backend(
    task: TaskSpec,
    fit_frame: pd.DataFrame,
    predict_smiles: list[str],
    smiles_col: str,
    y_col: str,
    model_dir: Path,
    protocol: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from unimol_tools import MolPredict, MolTrain
    except ImportError as exc:
        raise RuntimeError(
            "Frozen fine-tuning requires an environment with unimol-tools"
        ) from exc
    train_frame = fit_frame[[smiles_col, y_col]].rename(
        columns={smiles_col: "SMILES", y_col: "TARGET"}
    )
    predict_frame = pd.DataFrame({"SMILES": predict_smiles, "TARGET": -1.0})
    metric = {
        "auroc": "auc",
        "auprc": "auprc",
        "mae": "mae",
        "spearman": "spearmanr",
    }[task.metric]
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer = MolTrain(
        task="classification" if task.task_type == "classification" else "regression",
        data_type="molecule",
        epochs=int(protocol["epochs"]),
        learning_rate=float(protocol["learning_rate"]),
        batch_size=int(protocol["batch_size"]),
        early_stopping=max(2, min(5, int(protocol["epochs"]) // 3)),
        metrics=metric,
        split="scaffold",
        kfold=int(protocol["inner_folds"]),
        save_path=str(model_dir),
        smiles_col="SMILES",
        target_cols=["TARGET"],
        target_normalize="auto",
        use_cuda=True,
        use_amp=True,
        use_ddp=False,
        use_gpu="0",
        model_name="unimolv2",
        model_size=str(protocol["model_size"]),
        conf_cache_level=1,
        seed=seed,
        split_seed=seed,
    )
    _set_unimol_training_seed(trainer, seed)
    trainer.fit(train_frame)
    oof = np.asarray(trainer.cv_pred, dtype=float).reshape(-1)
    predictor = MolPredict(load_model=str(model_dir))
    predicted = np.asarray(predictor.predict(predict_frame), dtype=float).reshape(-1)
    return oof, predicted


class FrozenRecipeExecutor:
    """Fit a frozen expert graph on all development rows and predict unlabeled rows."""

    def __init__(
        self,
        cache: FeatureCache,
        artifact_dir: Path,
        device: str,
        n_jobs: int,
        finetune_backend: FineTuneBackend = _default_finetune_backend,
        focal_head_backend: FocalHeadBackend = _default_focal_head_backend,
        maplight_gnn_backend: MapLightGNNBackend = _default_maplight_gnn_backend,
    ):
        self.cache = cache
        self.artifact_dir = artifact_dir
        self.device = device
        self.n_jobs = n_jobs
        self.finetune_backend = finetune_backend
        self.focal_head_backend = focal_head_backend
        self.maplight_gnn_backend = maplight_gnn_backend

    def _tabular(
        self,
        task: TaskSpec,
        name: str,
        seed: int,
        fit_smiles: list[str],
        predict_smiles: list[str],
        y_fit: np.ndarray,
        need_oof: bool,
        inner_folds: int,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if name == "rank_target_xgboost":
            model_name = "xgboost"
            representations = RICH_2D
            hyperparameters = {
                "n_estimators": 1000,
                "max_depth": 3,
                "learning_rate": 0.025,
                "min_child_weight": 4.0,
            }
            fit_target = rankdata(y_fit, method="average") / len(y_fit)
        else:
            model_name, representations, hyperparameters = BASE_TABULAR[name]
            fit_target = y_fit
            if model_name == "svm_rbf" and task.task_type == "regression":
                hyperparameters = {**hyperparameters, "epsilon": 0.1}
        all_features = self.cache.get_or_compute(
            [*fit_smiles, *predict_smiles], representations
        )
        x_fit = all_features[: len(fit_smiles)]
        x_predict = all_features[len(fit_smiles) :]
        imbalance = "class_weight" if task.task_type == "classification" else "none"

        def experiment(fold_seed: int) -> ExperimentSpec:
            return ExperimentSpec(
                task=task.name,
                seed=fold_seed,
                representations=representations,
                model=model_name,
                imbalance=imbalance,
                device=self.device,
                n_jobs=self.n_jobs,
                hyperparameters=hyperparameters,
                hypothesis="frozen full-development expert",
            )

        oof = None
        if need_oof:
            groups = _scaffolds(fit_smiles)
            unique_groups = len(np.unique(groups))
            folds = min(inner_folds, unique_groups)
            if folds < 2:
                raise ValueError("Frozen OOF execution requires at least two scaffolds")
            oof = np.zeros(len(fit_smiles), dtype=float)
            splitter = GroupKFold(n_splits=folds)
            for fold_index, (fit_indices, holdout_indices) in enumerate(
                splitter.split(x_fit, fit_target, groups)
            ):
                fold_model = build_model(
                    experiment(seed * 100 + fold_index), task, fit_target[fit_indices]
                )
                fold_model.fit(x_fit[fit_indices], fit_target[fit_indices])
                oof[holdout_indices] = predict(
                    fold_model, x_fit[holdout_indices], task
                )
        full_model = build_model(experiment(seed), task, fit_target)
        full_model.fit(x_fit, fit_target)
        return oof, predict(full_model, x_predict, task)

    def _minimol_morgan_kernel(
        self,
        fit_smiles: list[str],
        predict_smiles: list[str],
        y_fit: np.ndarray,
    ) -> np.ndarray:
        """Locked Bioavailability kernel confirmed on fresh scaffold seeds 6-10."""
        fit_r2 = self.cache.get_or_compute(fit_smiles, ("morgan",))
        predict_r2 = self.cache.get_or_compute(predict_smiles, ("morgan",))
        fit_r3 = self.cache.get_or_compute(fit_smiles, ("morgan_r3",))
        predict_r3 = self.cache.get_or_compute(predict_smiles, ("morgan_r3",))
        fit_minimol = _normalize_rows(
            self.cache.get_or_compute(fit_smiles, ("minimol",))
        )
        predict_minimol = _normalize_rows(
            self.cache.get_or_compute(predict_smiles, ("minimol",))
        )
        chemical_fit = (
            _cross_tanimoto(fit_r2, fit_r2)
            + _cross_tanimoto(fit_r3, fit_r3)
        ) / 2
        chemical_predict = (
            _cross_tanimoto(predict_r2, fit_r2)
            + _cross_tanimoto(predict_r3, fit_r3)
        ) / 2
        fit_kernel = 0.5 * chemical_fit + 0.5 * (fit_minimol @ fit_minimol.T)
        predict_kernel = (
            0.5 * chemical_predict + 0.5 * (predict_minimol @ fit_minimol.T)
        )
        model = SVC(
            C=10.0,
            kernel="precomputed",
            class_weight="balanced",
            cache_size=1024,
            random_state=0,
        )
        model.fit(fit_kernel, y_fit)
        return np.asarray(model.decision_function(predict_kernel), dtype=float)

    def predict(
        self,
        task: TaskSpec,
        recipe_value: str,
        seed: int,
        fit_frame: pd.DataFrame,
        predict_smiles: list[str],
        smiles_col: str,
        y_col: str,
        protocol: dict,
    ) -> np.ndarray:
        recipe = parse_recipe(recipe_value)
        unknown = sorted(set(recipe.expert_names) - EXPERTS.keys())
        if unknown:
            raise ValueError(f"Frozen recipe contains unknown experts: {unknown}")
        fit_smiles = fit_frame[smiles_col].astype(str).tolist()
        y_fit = fit_frame[y_col].to_numpy()
        if len(predict_smiles) == 0:
            raise ValueError("Frozen execution requires prediction SMILES")
        need_fixed = "fixed_rank_top3" in recipe.expert_names
        required = set(recipe.expert_names) - {"fixed_rank_top3"}
        if need_fixed:
            required.update(FIXED_TOP3_POOL)
        predictions: dict[str, np.ndarray] = {}
        oof: dict[str, np.ndarray] = {}
        if "finetuned_unimol2" in required:
            if self.device != "cuda" and self.finetune_backend is _default_finetune_backend:
                raise RuntimeError("Frozen Uni-Mol2 fine-tuning requires device='cuda'")
            fine_oof, fine_prediction = self.finetune_backend(
                task,
                fit_frame,
                predict_smiles,
                smiles_col,
                y_col,
                self.artifact_dir / "unimol2_finetune",
                protocol,
                seed,
            )
            if len(fine_oof) != len(fit_frame):
                raise ValueError("Uni-Mol2 OOF predictions do not match fit rows")
            oof["finetuned_unimol2"] = fine_oof
            predictions["finetuned_unimol2"] = fine_prediction
        if "minimol_morgan_kernel" in required:
            predictions["minimol_morgan_kernel"] = self._minimol_morgan_kernel(
                fit_smiles,
                predict_smiles,
                y_fit,
            )
        if "focal_minimol_16" in required:
            all_minimol = self.cache.get_or_compute(
                [*fit_smiles, *predict_smiles], ("minimol",)
            )
            predictions["focal_minimol_16"] = self.focal_head_backend(
                task,
                all_minimol[: len(fit_smiles)],
                y_fit,
                all_minimol[len(fit_smiles) :],
                seed,
                self.artifact_dir / "focal_minimol_16",
            )
        if "maplight_gnn" in required:
            predictions["maplight_gnn"] = self.maplight_gnn_backend(
                task,
                fit_smiles,
                y_fit,
                predict_smiles,
                seed,
                self.n_jobs,
                self.artifact_dir / "maplight_gnn",
            )
        for name in sorted(
            required
            - {
                "finetuned_unimol2",
                "minimol_morgan_kernel",
                "focal_minimol_16",
                "maplight_gnn",
            }
        ):
            expert_oof, expert_prediction = self._tabular(
                task,
                name,
                seed,
                fit_smiles,
                predict_smiles,
                y_fit,
                need_fixed and name in FIXED_TOP3_POOL,
                int(protocol["inner_folds"]),
            )
            predictions[name] = expert_prediction
            if expert_oof is not None:
                oof[name] = expert_oof
        audit = {
            "task": task.name,
            "seed": seed,
            "recipe": recipe_value,
            "fit_rows": len(fit_frame),
            "predict_rows": len(predict_smiles),
            "test_labels_passed_to_executor": False,
        }
        if "minimol_morgan_kernel" in required:
            audit["minimol_morgan_kernel"] = {
                "c": 10.0,
                "class_weight": "balanced",
                "chemical_weight": 0.5,
                "chemical_kernel": "mean Morgan radius-2/radius-3 Tanimoto",
                "embedding_kernel": "MiniMol cosine",
            }
        if "focal_minimol_16" in required:
            audit["focal_minimol_16"] = {
                "members": 16,
                "epochs": 100,
                "batch_size": 32,
                "loss": "positive-weighted focal BCE",
                "focal_gamma": 2.0,
                "head_seed_formula": "outer_seed*10000+500+member",
                "predict_labels_passed": False,
            }
        if "maplight_gnn" in required:
            audit["maplight_gnn"] = {
                "upstream_commit": "c249378c63232354d17083c83fe94fe728960a27",
                "upstream_sha256": "74fbd1c98d9afa7fa4bda1add21efd429e20dee0a4b0fb8fa7e9b3825c21fe13",
                "checkpoint_sha256": "6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9",
                "classifier": "CatBoost random_strength=2; all other model defaults",
                "predict_labels_passed": False,
                "offline_content_addressed_checkpoint": True,
            }
        if need_fixed:
            scores = {name: score(y_fit, oof[name], task.metric) for name in FIXED_TOP3_POOL}
            direction = 1 if task.maximize else -1
            selected = sorted(
                scores, key=lambda name: direction * scores[name], reverse=True
            )[:3]
            predictions["fixed_rank_top3"] = np.mean(
                [
                    rankdata(predictions[name], method="average")
                    / len(predictions[name])
                    for name in selected
                ],
                axis=0,
            )
            audit["fixed_rank_top3"] = {
                "oof_scores": scores,
                "selected_experts": selected,
            }
        result = aggregate_predictions(recipe, predictions)
        if len(result) != len(predict_smiles):
            raise ValueError("Frozen recipe output does not match prediction rows")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "execution_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
