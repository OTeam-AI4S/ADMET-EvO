from admet_agent.source_snapshot import production_source_manifest
from scripts.run_post_memory_confirmation import _specs


def test_confirmation_specs_are_a_single_model_axis_pair():
    protocol = {
        "representations": ["morgan", "rdkit2d"],
        "hyperparameters": {
            "extra_trees": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "random_forest": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
        },
    }
    specs = _specs(protocol, "DILI", 6, 4)
    assert specs["control"].model == "extra_trees"
    assert specs["treatment"].model == "random_forest"
    assert specs["control"].representations == specs["treatment"].representations
    assert specs["control"].data_policy == specs["treatment"].data_policy
    assert "unimol2" not in specs["treatment"].representations
    assert len(production_source_manifest()["source_sha256"]) == 64
