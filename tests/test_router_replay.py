from scripts.replay_cost_aware_router import replay_task


def _result(run_id, seed, score, std, seconds, representations):
    return {
        "run_id": run_id,
        "task": "Example",
        "seed": seed,
        "model": "lightgbm",
        "metric": "mae",
        "validation_score": score,
        "validation_std": std,
        "elapsed_seconds": seconds,
        "representations": representations,
        "hyperparameters": {},
        "hypothesis": "test",
        "status": "completed",
        "error": None,
    }


def test_replay_reconstructs_seed_cost_and_routes_by_lcb():
    profile = {
        "task": "Example",
        "task_type": "regression",
        "metric": "mae",
        "n_train": 100,
        "n_valid": 20,
        "positive_rate_train": None,
        "target_mean": 0.0,
        "target_std": 1.0,
        "invalid_smiles_train": 0,
        "unique_scaffolds_train": 80,
        "unseen_valid_scaffold_rate": 1.0,
    }
    raw = []
    for seed in (1, 2, 3):
        raw.append(_result(f"cheap-{seed}", seed, 0.41, None, 10, ["morgan"]))
        raw.append(_result(f"3d-{seed}", seed, 0.40, None, 100, ["unimol2"]))
    payload = {
        "profile": profile,
        "validation_seeds": [1, 2, 3],
        "results": raw,
        "robust_results": [
            _result("cheap-1", 1, 0.41, 0.005, 10, ["morgan"]),
            _result("3d-1", 1, 0.40, 0.08, 100, ["unimol2"]),
        ],
        "ensemble_results": [],
    }

    report = replay_task(payload)

    assert report["selections"]["score"]["run_id"] == "3d-1"
    assert report["selections"]["lcb"]["run_id"] == "cheap-1"
    assert report["selections"]["cost_aware_lcb"]["recipe_seconds"] == 30
    assert report["selections"]["cost_aware_lcb"]["action"] == "stop_2d"
