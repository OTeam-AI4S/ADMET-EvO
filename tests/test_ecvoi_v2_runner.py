from __future__ import annotations

import hashlib
import json
import types

import numpy as np
import pandas as pd

from admet_agent.endpoint_contracts import endpoint_contract
from admet_agent.policy_brains import benchmark_brain
from admet_agent.runner import ADMETAgent
from admet_agent.schemas import RunResult, TaskProfile
from admet_agent.task_context import CONTEXT_VERSION, HARDNESS_DIMENSIONS
from admet_agent.tasks import resolve_task


class _Group:
    def __init__(self):
        self.train = pd.DataFrame(
            {
                "Drug": [
                    "CC",
                    "CCC",
                    "CCCC",
                    "CCO",
                    "CCN",
                    "CCCl",
                    "CCBr",
                    "CC(=O)O",
                ],
                "Y": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        self.valid = pd.DataFrame(
            {"Drug": ["CO", "CN", "CCl", "CBr"], "Y": [0, 1, 0, 1]}
        )

    def get_train_valid_split(self, **_):
        return self.train.copy(), self.valid.copy()


class _Ledger:
    def __init__(self):
        self.events = []

    def append(self, event, payload):
        self.events.append((event, payload))


class _Database:
    @staticmethod
    def summary():
        return {"runs": 9}


class _Packet:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


def _context(task):
    return {
        "context_version": CONTEXT_VERSION,
        "task": task,
        "task_type": "classification",
        "metric": "auroc",
        "category": "distribution",
        "hardness": {name: 0.2 for name in HARDNESS_DIMENSIONS},
        "hardness_evidence_sha256": "a" * 64,
        "discovery_only": True,
    }


def _write_positive_memory(path):
    row = {
        "endpoint_id": "BBB_Martins",
        "pair_id": "frozen-positive-pair",
        "action_id": "data-quality-filter",
        "native_oriented_delta": 0.02,
        "native_standard_error": 0.001,
        "effect_standardization_scale": 0.005,
        "single_axis_verified": True,
        "complete_pair": True,
        "transport_eligible": True,
        "test_derived": False,
        "leaderboard_derived": False,
    }
    rendered = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    row["observation_sha256"] = hashlib.sha256(rendered).hexdigest()
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_shared_anchor_is_hash_bound_identity_safe_and_multiseed(tmp_path):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "BBB_Martins": {
                    "model": "extra_trees",
                    "representations": ["morgan", "rdkit2d"],
                    "data_policy": "official",
                }
            }
        ),
        encoding="utf-8",
    )
    agent = ADMETAgent.__new__(ADMETAgent)
    agent.group = _Group()
    agent.run_dir = tmp_path / "run"
    agent.run_dir.mkdir()
    agent.device = "cpu"
    agent.n_jobs = 1
    agent.evidence_ledger = _Ledger()
    agent.benchmark_protocol = {
        "frozen_invariants": {
            "shared_anchor_mapping_path": str(mapping),
            "shared_anchor_mapping_sha256": hashlib.sha256(
                mapping.read_bytes()
            ).hexdigest(),
            "shared_anchor_data_policy": "identity_safe",
        }
    }
    counter = {"value": 0}

    def fake_execute(self, experiment, *_args, **_kwargs):
        counter["value"] += 1
        run_id = f"anchor-member-{counter['value']}"
        artifact_dir = self.run_dir / "artifacts" / run_id
        artifact_dir.mkdir(parents=True)
        np.save(
            artifact_dir / "validation_predictions.npy",
            np.asarray([0.1, 0.9, 0.2, 0.8]),
            allow_pickle=False,
        )
        return RunResult(
            run_id=run_id,
            task=experiment.task,
            seed=experiment.seed,
            model=experiment.model,
            metric="auroc",
            validation_score=1.0,
            elapsed_seconds=1.0,
            representations=experiment.representations,
            hypothesis=experiment.hypothesis,
            data_policy=experiment.data_policy,
        )

    agent._execute = types.MethodType(fake_execute, agent)
    result = agent._execute_shared_anchor(
        task=resolve_task("BBB_Martins"),
        benchmark_name="BBB_Martins",
        validation_seeds=[1, 2, 3],
        task_dir=tmp_path / "task",
    )
    assert result is not None
    assert result["summary"]["complete"] is True
    assert result["summary"]["data_policy"] == "identity_safe"
    assert result["aggregate"].validation_score == 1.0
    hash(tuple(sorted(result["aggregate"].hyperparameters.items())))
    assert all(
        row.data_policy == "identity_safe" for row in result["member_results"]
    )
    profile = TaskProfile(
        task="BBB_Martins",
        task_type="classification",
        metric="auroc",
        n_train=8,
        n_valid=4,
        positive_rate_train=0.5,
        target_mean=0.5,
        target_std=0.5,
        invalid_smiles_train=0,
        unique_scaffolds_train=2,
        unseen_valid_scaffold_rate=0.5,
    )
    for arm in ("agent", "rules", "random", "automl", "expanded_automl"):
        assert benchmark_brain(arm).propose(profile, [result["aggregate"]])


def test_v2_runner_keeps_safe_anchor_and_executes_atomic_same_seed_pair(tmp_path):
    memory = tmp_path / "memory.jsonl"
    _write_positive_memory(memory)
    agent = ADMETAgent.__new__(ADMETAgent)
    agent.group = _Group()
    agent.research_routing_policy = "ecvoi-v2-no-cost"
    agent.action_context_registry = {"BBB_Martins": _context("BBB_Martins")}
    agent.action_context_registry_sha256 = "b" * 64
    agent.ecvoi_v2_memory_snapshot_path = memory
    agent.ecvoi_v2_memory_snapshot_sha256 = hashlib.sha256(
        memory.read_bytes()
    ).hexdigest()
    agent.action_memory_snapshot_sha256 = None
    agent.benchmark_protocol = None
    agent.device = "cpu"
    agent.n_jobs = 1
    agent.selection_policy = "score"
    agent.source_manifest = {"source_sha256": "c" * 64}
    agent.evidence_ledger = _Ledger()
    agent.database = _Database()
    counter = {"value": 0}

    def fake_execute(self, experiment, *_args, **_kwargs):
        counter["value"] += 1
        score = 0.8 if experiment.data_policy == "leakage_safe" else 0.7
        return RunResult(
            run_id=f"run-{counter['value']}",
            task=experiment.task,
            seed=experiment.seed,
            model=experiment.model,
            metric="auroc",
            validation_score=score,
            elapsed_seconds=1.0,
            representations=experiment.representations,
            hypothesis=experiment.hypothesis,
            data_policy=experiment.data_policy,
        )

    agent._execute = types.MethodType(fake_execute, agent)
    profile = TaskProfile(
        task="BBB_Martins",
        task_type="classification",
        metric="auroc",
        n_train=8,
        n_valid=4,
        positive_rate_train=0.5,
        target_mean=0.5,
        target_std=0.5,
        invalid_smiles_train=0,
        unique_scaffolds_train=2,
        unseen_valid_scaffold_rate=0.5,
    )
    summary = agent._run_ecvoi_v2_search(
        task=resolve_task("BBB_Martins"),
        benchmark_name="BBB_Martins",
        profile=profile,
        hardness=_Packet({"task": "BBB_Martins"}),
        contract=endpoint_contract("BBB_Martins"),
        research_plan={
            "actions": [
                {
                    "action_id": "data-quality-filter",
                    "axis": "data",
                    "scientific_risk": 0.05,
                }
            ]
        },
        task_dir=tmp_path / "task",
        validation_seeds=[1, 2, 3],
        task_compute_minutes=90,
        benchmark_arm="ecvoi_v2_no_cost",
    )
    assert summary["test_loaded"] is False
    assert summary["baseline_anchor"]["complete"] is True
    assert len(summary["baseline_anchor"]["run_ids"]) == 3
    assert len(summary["results"]) == 9
    assert summary["finalist_spec"]["data_policy"] == "leakage_safe"
    ledger = summary["ecvoi_v2_episode"]["ledger"]
    assert [event["event_type"] for event in ledger[:3]] == [
        "decision_emitted",
        "pair_budget_reserved",
        "pair_execution_complete",
    ]
    execution = ledger[2]
    assert execution["ordered_seeds"] == [1, 2, 3]
    assert len(execution["treatment_run_ids"]) == 3
    assert len(execution["control_run_ids"]) == 3
    assert execution["posterior_updated"] is True
