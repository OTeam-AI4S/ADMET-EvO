from __future__ import annotations

from types import MethodType

import pandas as pd

from admet_agent.runner import ADMETAgent
from admet_agent.schemas import ExperimentSpec, RunResult
from admet_agent.tasks import resolve_task


class _SplitGroup:
    @staticmethod
    def get_train_valid_split(*, benchmark, split_type, seed):
        del benchmark, split_type
        train = pd.DataFrame(
            {
                "Drug": ["CCO", "CCN", "CCC", "CCCl"] * 12,
                "Y": [0, 1, 0, 1] * 12,
            }
        )
        valid = pd.DataFrame(
            {
                "Drug": ["CCBr", "CCF", "CNC", "COC"] * 6,
                "Y": [0, 1, 0, 1] * 6,
            }
        )
        return train, valid


def _primary(run_id, model, score, seconds):
    return RunResult(
        run_id=run_id,
        task="HIA_Hou",
        seed=1,
        model=model,
        metric="auroc",
        validation_score=score,
        elapsed_seconds=seconds,
        representations=("morgan",),
        hyperparameters={},
        hypothesis="racing candidate",
    )


def test_runner_applies_seed2_barrier_and_records_fits_avoided():
    runner = ADMETAgent.__new__(ADMETAgent)
    runner.group = _SplitGroup()
    runner.device = "cpu"
    runner.n_jobs = 1
    runner.benchmark_protocol = {
        "robust_futility_policy": "seed2_gap0.005_time3x_worse_both_protect_action_pair",
        "robust_futility_applies_to_all_arms": True,
    }
    scores = {
        "lightgbm": (0.80, 1.0),
        "extra_trees": (0.70, 10.0),
        "random_forest": (0.69, 12.0),
    }

    def fake_execute(
        self,
        experiment,
        train,
        valid,
        smiles_col,
        y_col,
        profile,
        *,
        timeout_seconds,
    ):
        del self, train, valid, smiles_col, y_col, profile, timeout_seconds
        value, seconds = scores[experiment.model]
        return RunResult(
            run_id=f"{experiment.model}-seed-{experiment.seed}",
            task=experiment.task,
            seed=experiment.seed,
            model=experiment.model,
            metric="auroc",
            validation_score=value + 0.001 * experiment.seed,
            elapsed_seconds=seconds,
            representations=experiment.representations,
            hyperparameters=experiment.hyperparameters,
            hypothesis=experiment.hypothesis,
        )

    runner._execute = MethodType(fake_execute, runner)
    completed = [
        _primary("fast", "lightgbm", 0.801, 1.0),
        _primary("slow-a", "extra_trees", 0.701, 10.0),
        _primary("slow-b", "random_forest", 0.691, 12.0),
    ]
    experiments = {
        row.run_id: ExperimentSpec(
            task="HIA_Hou",
            seed=1,
            model=row.model,
            representations=("morgan",),
            hypothesis="racing candidate",
        )
        for row in completed
    }
    history = list(completed)
    plan = {
        "selected_action_id": "stop-low-value",
        "actions": [{"action_id": "stop-low-value", "axis": "stop"}],
    }

    robust, seed_runs, racing = runner._robust_results(
        resolve_task("HIA_Hou"),
        "HIA_Hou",
        1,
        [1, 2, 3, 4, 5],
        completed,
        experiments,
        history,
        3,
        plan,
        None,
    )

    assert racing["enabled"] is True
    assert set(racing["stopped_candidate_ids"]) == {"slow-a", "slow-b"}
    assert racing["planned_fits_avoided"] == 6
    assert racing["test_loaded"] is False
    assert set(seed_runs["fast"]) == {1, 2, 3, 4, 5}
    assert set(seed_runs["slow-a"]) == {1, 2}
    assert [row.run_id for row in robust] == ["fast"]
    assert len(history) == 3 + 3 + 3
