"""Central registry for capabilities exposed to planners and validators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class RepresentationTool:
    name: str
    family: str
    width: int
    description: str


@dataclass(frozen=True)
class ModelTool:
    name: str
    family: str
    supports_classification: bool = True
    supports_regression: bool = True
    max_recommended_samples: int | None = None


REPRESENTATIONS = {
    item.name: item
    for item in (
        RepresentationTool("morgan", "circular", 2048, "ECFP radius 2 bit fingerprint"),
        RepresentationTool("morgan_r3", "circular", 2048, "ECFP radius 3 bit fingerprint"),
        RepresentationTool("morgan_count", "circular", 2048, "ECFP radius 2 count fingerprint"),
        RepresentationTool("avalon", "substructure", 1024, "Avalon substructure fingerprint"),
        RepresentationTool("erg", "pharmacophore", 315, "ErG reduced-graph fingerprint"),
        RepresentationTool("maccs", "keys", 167, "MACCS structural keys"),
        RepresentationTool("rdkit_fp", "path", 2048, "RDKit path fingerprint"),
        RepresentationTool("atom_pair", "topological", 2048, "Hashed atom-pair fingerprint"),
        RepresentationTool("torsion", "topological", 2048, "Hashed topological-torsion fingerprint"),
        RepresentationTool("rdkit2d", "descriptor", -1, "All finite RDKit 2D descriptors"),
        RepresentationTool(
            "unimol2",
            "pretrained_3d",
            -1,
            "Frozen Uni-Mol2 84M molecular CLS representation",
        ),
    )
}


def is_representation_available(name: str) -> bool:
    if name == "unimol2":
        return find_spec("unimol_tools") is not None
    return True

MODELS = {
    item.name: item
    for item in (
        ModelTool("extra_trees", "bagged_trees"),
        ModelTool("random_forest", "bagged_trees"),
        ModelTool(
            "balanced_extra_trees",
            "undersampling_ensemble",
            supports_regression=False,
        ),
        ModelTool("catboost", "gradient_boosting"),
        ModelTool("xgboost", "gradient_boosting"),
        ModelTool("lightgbm", "gradient_boosting"),
        ModelTool("svm_rbf", "kernel", max_recommended_samples=5000),
    )
}


def is_model_available(name: str) -> bool:
    modules = {"catboost": "catboost", "xgboost": "xgboost", "lightgbm": "lightgbm"}
    module = modules.get(name)
    return module is None or find_spec(module) is not None


def planner_catalog() -> dict:
    from .data_skill import skill_descriptor

    return {
        "representations": [
            {**asdict(value), "available": is_representation_available(value.name)}
            for value in REPRESENTATIONS.values()
        ],
        "models": [
            {**asdict(value), "available": is_model_available(value.name)}
            for value in MODELS.values()
        ],
        "skills": [skill_descriptor()],
    }
