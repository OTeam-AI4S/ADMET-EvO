"""Protocol firewall: validate experiments and discovery inputs before compute."""
from pathlib import Path
from .registry import MODELS
from .schemas import ALLOWED_DEVICES, ALLOWED_IMBALANCE, ALLOWED_MODELS, ALLOWED_REPRESENTATIONS, ExperimentSpec
from .tasks import resolve_task

MODEL_PARAMETERS = {
    "extra_trees": {"n_estimators", "min_samples_leaf", "max_features", "criterion", "bootstrap", "max_samples"},
    "random_forest": {"n_estimators", "min_samples_leaf", "max_features", "criterion", "bootstrap", "max_samples"},
    "balanced_extra_trees": {"n_estimators", "n_members", "majority_ratio", "min_samples_leaf", "max_features", "criterion"},
    "catboost": {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength"},
    "xgboost": {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "reg_alpha", "reg_lambda"},
    "lightgbm": {"n_estimators", "num_leaves", "max_depth", "learning_rate", "subsample",
                 "colsample_bytree", "min_child_samples", "reg_alpha", "reg_lambda"},
    "svm_rbf": {"C", "gamma", "epsilon"},
}

class ProtocolViolation(ValueError):
    pass


def validate_tdc_discovery_path(path: str | Path) -> Path:
    """Reject official TDC test files from any discovery/data-research loader.

    External source-heldout sets use a different namespace and are not covered
    by this guard.  This function specifically protects the canonical TDC
    benchmark tree, whose test labels must never enter discovery.
    """

    resolved = Path(path).resolve()
    lowered_parts = tuple(part.lower() for part in resolved.parts)
    in_tdc_benchmark = "admet_group" in lowered_parts
    test_named = resolved.name.lower() in {"test.csv", "test.tsv", "test.parquet"}
    if in_tdc_benchmark and test_named:
        raise ProtocolViolation(f"official TDC test path is forbidden during discovery: {resolved}")
    return resolved

def validate_experiment(spec: ExperimentSpec) -> None:
    task = resolve_task(spec.task)
    errors = []
    unknown = set(spec.representations) - ALLOWED_REPRESENTATIONS
    if unknown:
        errors.append(f"unknown representations: {sorted(unknown)}")
    if not spec.representations:
        errors.append("at least one representation is required")
    if len(spec.representations) > 4:
        errors.append("at most four representations are allowed")
    if len(set(spec.representations)) != len(spec.representations):
        errors.append("representations must not contain duplicates")
    if spec.model not in ALLOWED_MODELS:
        errors.append(f"model must be one of {sorted(ALLOWED_MODELS)}")
    elif task.task_type == "classification" and not MODELS[spec.model].supports_classification:
        errors.append(f"{spec.model} does not support classification")
    elif task.task_type == "regression" and not MODELS[spec.model].supports_regression:
        errors.append(f"{spec.model} does not support regression")
    if spec.imbalance not in ALLOWED_IMBALANCE:
        errors.append(f"imbalance must be one of {sorted(ALLOWED_IMBALANCE)}")
    if spec.device not in ALLOWED_DEVICES:
        errors.append(f"device must be one of {sorted(ALLOWED_DEVICES)}")
    if task.task_type == "regression" and spec.imbalance != "none":
        errors.append("imbalance strategies are classification-only")
    if spec.seed < 0:
        errors.append("seed must be non-negative")
    if isinstance(spec.n_jobs, bool) or not isinstance(spec.n_jobs, int) or not 1 <= spec.n_jobs <= 256:
        errors.append("n_jobs must be an integer between 1 and 256")
    if not 1 <= spec.budget.max_candidates <= 20:
        errors.append("max_candidates must be between 1 and 20")
    if not 1 <= spec.budget.max_minutes <= 1440:
        errors.append("max_minutes must be between 1 and 1440")
    forbidden = {"test", "test_labels", "y_test", "evaluation_override"}
    leaked = forbidden.intersection(spec.hyperparameters)
    if leaked:
        errors.append(f"forbidden test/evaluator fields: {sorted(leaked)}")
    if spec.model in MODEL_PARAMETERS:
        invalid = set(spec.hyperparameters) - MODEL_PARAMETERS[spec.model]
        if invalid:
            errors.append(f"invalid {spec.model} parameters: {sorted(invalid)}")
    ranges = {"n_estimators": (100, 1200), "n_members": (3, 25), "majority_ratio": (0.5, 10.0),
              "min_samples_leaf": (1, 20), "max_samples": (0.3, 1.0),
              "iterations": (100, 2000), "depth": (3, 12), "learning_rate": (0.005, 0.3),
              "l2_leaf_reg": (0.0, 30.0), "random_strength": (0.0, 10.0),
              "max_depth": (2, 16), "subsample": (0.3, 1.0), "colsample_bytree": (0.1, 1.0),
              "min_child_weight": (0.0, 30.0), "reg_alpha": (0.0, 50.0),
              "reg_lambda": (0.0, 50.0), "num_leaves": (7, 255),
              "min_child_samples": (2, 100), "C": (0.01, 1000.0),
              "epsilon": (0.001, 2.0)}
    integer_parameters = {"n_estimators", "n_members", "min_samples_leaf", "iterations", "depth",
                          "max_depth", "num_leaves", "min_child_samples"}
    for key, (low, high) in ranges.items():
        if key not in spec.hyperparameters:
            continue
        parameter = spec.hyperparameters[key]
        if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
            errors.append(f"{key} must be numeric")
            continue
        if key in integer_parameters and not isinstance(parameter, int):
            errors.append(f"{key} must be an integer")
        if not low <= parameter <= high:
            errors.append(f"{key} must be between {low} and {high}")
    value = spec.hyperparameters.get("max_features")
    if value is not None:
        if isinstance(value, bool) or not (
            value in {"sqrt", "log2"} if isinstance(value, str)
            else isinstance(value, float) and 0.1 <= value <= 1.0
        ):
            errors.append("max_features must be sqrt, log2, or a float between 0.1 and 1.0")
    gamma = spec.hyperparameters.get("gamma")
    if gamma is not None and not (
        gamma in {"scale", "auto"} if isinstance(gamma, str)
        else isinstance(gamma, (int, float)) and not isinstance(gamma, bool) and 1e-6 <= gamma <= 10.0
    ):
        errors.append("gamma must be scale, auto, or a number between 1e-6 and 10")
    if spec.model == "svm_rbf" and task.task_type == "classification" and "epsilon" in spec.hyperparameters:
        errors.append("epsilon is regression-only")
    bootstrap = spec.hyperparameters.get("bootstrap")
    if bootstrap is not None and not isinstance(bootstrap, bool):
        errors.append("bootstrap must be boolean")
    if spec.hyperparameters.get("max_samples") is not None and spec.hyperparameters.get("bootstrap") is False:
        errors.append("max_samples requires bootstrap=true")
    criterion = spec.hyperparameters.get("criterion")
    classification_criteria = {"gini", "entropy", "log_loss"}
    regression_criteria = {"squared_error", "absolute_error"}
    if criterion and task.task_type == "classification" and criterion not in classification_criteria:
        errors.append("classification criterion is invalid")
    if criterion and task.task_type == "regression" and criterion not in regression_criteria:
        errors.append("regression criterion is invalid")
    if errors:
        raise ProtocolViolation("; ".join(errors))
