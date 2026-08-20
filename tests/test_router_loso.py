from admet_agent.schemas import RunResult, TaskProfile
from scripts.evaluate_router_loso import replay_loso


def test_loso_router_uses_only_training_seed_statistics():
    profile = TaskProfile(
        task="Example",
        task_type="regression",
        metric="mae",
        n_train=100,
        n_valid=20,
        positive_rate_train=None,
        target_mean=0.0,
        target_std=1.0,
        invalid_smiles_train=0,
        unique_scaffolds_train=80,
        unseen_valid_scaffold_rate=1.0,
    )
    templates = {
        "stable": RunResult("stable", "Example", 1, "lightgbm", "mae", 0, 0),
        "risky": RunResult(
            "risky", "Example", 1, "lightgbm", "mae", 0, 0,
            representations=("unimol2",),
        ),
    }
    scores = {
        "stable": {1: 0.40, 2: 0.41, 3: 0.39},
        "risky": {1: 0.30, 2: 0.60, 3: 0.50},
    }
    costs = {
        "stable": {1: 10, 2: 10, 3: 10},
        "risky": {1: 100, 2: 100, 3: 100},
    }

    report = replay_loso(profile, templates, scores, costs)

    assert report["policies"]["cost_aware_lcb"]["action_counts"] == {"stop_2d": 2}
    assert report["policies"]["cost_aware_lcb"]["mean"] == 0.40
    assert report["heldout_router_seeds"] == [2, 3]
