"""Test whether transformed-target experts add stable Half-Life rank evidence."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tdc.benchmark_group import admet_group

from admet_agent.features import FeatureCache
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

RICH = ("morgan", "avalon", "erg", "rdkit2d")
EXTENDED = (
    "morgan",
    "morgan_r3",
    "morgan_count",
    "avalon",
    "erg",
    "maccs",
    "rdkit2d",
)
PILOT_ROOT = Path("runs/unimol2-pilot-v22/Half_Life_Obach")
PILOT_SHARD_ROOT = Path("runs/unimol2-pilot-v22-shards")
OUTPUT_ROOT = Path("runs/half-life-fusion-ablation")


def _rank_blend(
    predictions: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    result = np.zeros_like(next(iter(predictions.values())), dtype=float)
    for name, weight in weights.items():
        result += weight * rankdata(predictions[name], method="average")
    return result


def _candidate_weights(names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    candidates = {name: {name: 1.0} for name in names}
    for left, right in itertools.combinations(names, 2):
        for left_weight in (0.25, 0.5, 0.75):
            label = f"rank({left}@{left_weight:.2f}+{right}@{1 - left_weight:.2f})"
            candidates[label] = {left: left_weight, right: 1 - left_weight}
    for group in itertools.combinations(names, 3):
        label = "rank(" + "+".join(f"{name}@0.33" for name in group) + ")"
        candidates[label] = {name: 1 / 3 for name in group}
    return candidates


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(Path("runs/hard-task-ablation/feature_cache"))
    group = admet_group(path="data")
    task = resolve_task("Half_Life_Obach")
    seed_scores: dict[int, dict[str, float]] = {}
    base_names = (
        "2d_svm",
        "finetuned_unimol2",
        "fixed_rank_top3",
        "rank_target_lightgbm",
        "rank_target_xgboost",
    )
    candidates = _candidate_weights(base_names)
    for seed in range(1, 6):
        train, valid = group.get_train_valid_split(
            benchmark=task.name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        all_smiles = train[smiles_col].tolist() + valid[smiles_col].tolist()
        target = rankdata(train[y_col].to_numpy(), method="average") / len(train)
        new_predictions = {}
        for label, model_name, representations, params in (
            ("rank_target_lightgbm", "lightgbm", EXTENDED, {}),
            (
                "rank_target_xgboost",
                "xgboost",
                RICH,
                {
                    "n_estimators": 1000,
                    "max_depth": 3,
                    "learning_rate": 0.025,
                    "min_child_weight": 4.0,
                },
            ),
        ):
            x_all = cache.get_or_compute(all_smiles, representations)
            spec = ExperimentSpec(
                task=task.name,
                seed=seed,
                representations=representations,
                model=model_name,
                imbalance="none",
                device="cpu",
                n_jobs=8,
                hyperparameters=params,
                hypothesis="transformed-target complementary rank evidence",
            )
            model = build_model(spec, task, target)
            model.fit(x_all[: len(train)], target)
            new_predictions[label] = predict(model, x_all[len(train) :], task)

        pilot_path = PILOT_ROOT / f"seed_{seed}" / "validation_predictions.npz"
        if not pilot_path.exists():
            matches = sorted(
                PILOT_SHARD_ROOT.glob(
                    f"*/Half_Life_Obach/seed_{seed}/validation_predictions.npz"
                )
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one pilot prediction shard for seed {seed}, got {matches}"
                )
            pilot_path = matches[0]
        with np.load(pilot_path) as pilot:
            predictions = {name: np.asarray(pilot[name]) for name in base_names[:3]}
        predictions.update(new_predictions)
        y_valid = valid[y_col].to_numpy()
        scores = {
            label: score(y_valid, _rank_blend(predictions, weights), task.metric)
            for label, weights in candidates.items()
        }
        seed_scores[seed] = scores
        seed_dir = OUTPUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(seed_dir / "validation_predictions.npz", **predictions)
        print(seed, max(scores, key=scores.get), max(scores.values()), flush=True)

    summaries = {
        name: {
            "mean": float(np.mean([seed_scores[seed][name] for seed in seed_scores])),
            "std": float(np.std([seed_scores[seed][name] for seed in seed_scores])),
            "scores": [seed_scores[seed][name] for seed in seed_scores],
        }
        for name in candidates
    }
    loso = {}
    for penalty in (0.0, 0.25, 0.5, 1.0):
        heldout_scores = []
        selected = {}
        for heldout in seed_scores:
            training_seeds = [seed for seed in seed_scores if seed != heldout]
            utilities = {}
            for name in candidates:
                values = [seed_scores[seed][name] for seed in training_seeds]
                utilities[name] = float(np.mean(values) - penalty * np.std(values))
            winner = max(utilities, key=utilities.get)
            selected[str(heldout)] = winner
            heldout_scores.append(seed_scores[heldout][winner])
        loso[f"lcb_{penalty}"] = {
            "mean": float(np.mean(heldout_scores)),
            "std": float(np.std(heldout_scores)),
            "scores": heldout_scores,
            "selected": selected,
        }
    report = {
        "test_labels_used": False,
        "selection_protocol": "leave-one-outer-scaffold-seed-out",
        "best_fixed_candidate": max(
            summaries, key=lambda name: summaries[name]["mean"]
        ),
        "summaries": summaries,
        "loso": loso,
    }
    (OUTPUT_ROOT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
