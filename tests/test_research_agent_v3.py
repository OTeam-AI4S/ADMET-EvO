import json
import time

import pandas as pd
import pytest

from admet_agent.action_evidence import evaluate_action_evidence, selected_action_pair
from admet_agent.action_memory import ResearchOutcomeMemory
from admet_agent.agent_evaluation import AutonomyEvaluator, ResearchEpisode
from admet_agent.benchmark_protocol import (
    PolicyTaskResult,
    compare_policy_results,
    default_protocol,
    write_default_protocol,
)
from admet_agent.brain import RuleBasedBrain
from admet_agent.data_governance import (
    LocalTableSource,
    MolecularDataCurator,
    SourceManifest,
    canonical_smiles,
)
from admet_agent.diagnosis_report import render_markdown, summarize_diagnoses
from admet_agent.endpoint_contracts import ENDPOINT_CONTRACTS, endpoint_contract
from admet_agent.experiment_db import ExperimentDatabase
from admet_agent.fault_detection import ScientificFaultDetector
from admet_agent.freeze_audit import (
    audit_global_freeze,
    verify_freeze_authorization,
)
from admet_agent.hardness import profile_hardness
from admet_agent.hidden_fault_benchmark import (
    FaultChallengeResult,
    score_hidden_faults,
)
from admet_agent.policy_brains import benchmark_brain
from admet_agent.policy_results import collect_policy_results
from admet_agent.research_loop import (
    EffectFirstResearchPlanner,
    EvidenceLedger,
    ResearchAction,
)
from admet_agent.runner import (
    ADMETAgent,
    CandidateTimeoutError,
    _arm_candidate_timeout,
    _disarm_candidate_timeout,
    _ensemble_evidence,
    _scaleup_stop_reason,
    _select_distinct_ensemble_winners,
    _select_robust_candidates,
)
from admet_agent.schemas import ExperimentSpec, RunResult, TaskProfile
from admet_agent.source_snapshot import snapshot_source
from admet_agent.tasks import TASKS, resolve_task


def _classification_frames():
    train = pd.DataFrame(
        {
            "Drug": ["CCO", "OCC", "CCN", "c1ccccc1", "invalid"] * 24,
            "Y": [0, 0, 1, 1, 0] * 24,
        }
    )
    valid = pd.DataFrame(
        {
            "Drug": ["CCO", "CCCl", "c1ccncc1", "CCCC"] * 30,
            "Y": [0, 1, 1, 0] * 30,
        }
    )
    return train, valid


def test_endpoint_contracts_cover_only_the_22_tdc_tasks():
    assert len(ENDPOINT_CONTRACTS) == 22
    contract = endpoint_contract("cyp2c9_veith")
    assert contract.primary_metric == "auprc"
    assert contract.certification_split == "tdc_evaluator_owned_test"


def test_curator_canonicalizes_deduplicates_and_removes_validation_overlap():
    frame = pd.DataFrame({"Drug": ["CCO", "OCC", "CCN", "bad"], "Y": [1, 1, 0, 1]})
    curator = MolecularDataCurator()
    curated, audit = curator.curate(
        frame,
        task_type="classification",
        policy="leakage_safe",
        validation_smiles={canonical_smiles("CCN")},
    )
    assert curated["Drug"].tolist() == [canonical_smiles("CCO")]
    assert audit.invalid_smiles == 1
    assert audit.duplicate_rows == 1
    assert audit.removed_validation_overlap == 1
    assert audit.input_sha256 != audit.output_sha256


def test_external_source_is_cleaned_and_quarantined_with_provenance(tmp_path):
    source_path = tmp_path / "ames.csv"
    pd.DataFrame({"smiles": ["CCO", "OCC", "bad"], "label": [1, 1, 0]}).to_csv(
        source_path, index=False
    )
    manifest = SourceManifest(
        source_id="ames_public_v1",
        endpoint="AMES",
        origin="https://example.invalid/dataset",
        license="CC-BY-4.0",
        acquired_at="2026-08-19T00:00:00Z",
        assay_system="Ames bacterial mutagenicity assay",
        species="bacteria",
        unit=None,
        label_transform="identity binary label",
        citation="Example source",
    )
    curated, packet = MolecularDataCurator().ingest_external(
        LocalTableSource(source_path, manifest),
        endpoint_contract("AMES"),
        smiles_col="smiles",
        target_col="label",
        output_dir=tmp_path / "quarantine",
    )
    assert len(curated) == 1
    assert packet["merge_status"] == "quarantined_pending_assay_alignment"
    assert (tmp_path / "quarantine" / "ames_public_v1.manifest.json").exists()


def test_regression_replicates_are_distinguished_from_class_conflicts():
    frame = pd.DataFrame(
        {"Drug": ["CCO", "OCC", "CCN", "NCC"], "Y": [1.0, 1.1, 3.0, 3.2]}
    )
    audit = MolecularDataCurator().audit(frame, task_type="regression")
    assert audit.replicate_rows == 2
    assert audit.replicate_target_dispersion is not None
    assert "repeated measurements" in " ".join(audit.warnings)


def test_experiment_contract_rejects_unknown_data_policy():
    with pytest.raises(ValueError, match="data_policy"):
        ExperimentSpec(task="AMES", seed=1, data_policy="silently_use_test")


def test_hardness_card_is_multidimensional_and_routes_actions():
    train, valid = _classification_frames()
    card = profile_hardness(
        train, valid, resolve_task("AMES"), sample_limit=100, seed=3
    )
    assert 0 <= card.overall <= 1
    assert {
        "scarcity",
        "imbalance",
        "domain_shift",
        "local_roughness",
        "data_quality",
        "split_leakage_risk",
    } == set(card.dimensions)
    assert card.dimensions["split_leakage_risk"].severity > 0
    plan = EffectFirstResearchPlanner().plan(card)
    assert plan["selected_action_id"]
    assert {item["axis"] for item in plan["actions"]} >= {"data", "replicate", "stop"}


def test_diagnose_path_never_materializes_test_frame(tmp_path):
    train, valid = _classification_frames()

    class DiscoveryOnlyGroup:
        def get_train_valid_split(self, benchmark, split_type, seed):
            assert benchmark == "AMES"
            return train, valid

        def get(self, name):
            raise AssertionError("diagnosis must not call group.get/test")

    agent = object.__new__(ADMETAgent)
    agent.group = DiscoveryOnlyGroup()
    agent.run_dir = tmp_path
    agent.research_planner = EffectFirstResearchPlanner()
    agent.fault_detector = ScientificFaultDetector()
    agent.effect_priority = 0.80
    agent.evidence_ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    packet = agent.diagnose("AMES", seed=1)
    assert packet["protocol"] == "discovery_only_no_test_access"
    assert "exact_split_overlap" in packet["fault_report"]["detected_types"]
    assert (tmp_path / "tasks" / "AMES" / "diagnosis_seed_1.json").exists()


def test_evidence_ledger_detects_tampering(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    ledger.append("hypothesis", {"claim": "a"})
    ledger.append("result", {"supported": True})
    assert ledger.verify()["valid"] is True
    rows = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(rows[0])
    record["payload"]["claim"] = "tampered"
    rows[0] = json.dumps(record)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert ledger.verify()["valid"] is False


def test_autonomy_scorecard_penalizes_missing_tasks_and_gates_test_feedback():
    safe = ResearchEpisode(
        task="AMES",
        baseline_validation_score=0.75,
        agent_validation_score=0.80,
        baseline_compute_seconds=100,
        agent_compute_seconds=150,
        attempted_actions=3,
        preregistered_actions=3,
        supported_actions=2,
        validation_std=0.01,
        faults_present=2,
        faults_detected=2,
        reproducible_runs=5,
        requested_reproducible_runs=5,
    )
    safe_card = AutonomyEvaluator().evaluate([safe])
    assert safe_card.coverage == pytest.approx(1 / 22)
    assert safe_card.claim_level == "development_evidence_only"
    unsafe = ResearchEpisode(**{**safe.to_dict(), "test_feedback_events": 1})
    unsafe_card = AutonomyEvaluator().evaluate([unsafe])
    assert unsafe_card.safety_gate_passed is False
    assert unsafe_card.overall_score <= 0.25


def test_suite_diagnosis_report_requires_and_summarizes_all_22_tasks(tmp_path):
    for task in TASKS.values():
        task_dir = tmp_path / "tasks" / task.name
        task_dir.mkdir(parents=True)
        packet = {
            "task": task.name,
            "endpoint_contract": {
                "task_type": task.task_type,
                "primary_metric": task.metric,
            },
            "profile": {"n_train": 100},
            "hardness": {
                "overall": 0.4,
                "dominant_factors": ["domain_shift"],
                "dimensions": {
                    "domain_shift": {
                        "severity": 0.6,
                        "confidence": 0.8,
                        "evidence": {},
                    },
                    "data_quality": {
                        "severity": 0.0,
                        "confidence": 1.0,
                        "evidence": {"conflicting_structures": 0},
                    },
                    "split_leakage_risk": {
                        "severity": 0.0,
                        "confidence": 1.0,
                        "evidence": {"exact_train_valid_overlap_rows": 0},
                    },
                },
            },
            "research_plan": {"selected_action_id": "feature-domain-shift"},
        }
        (task_dir / "diagnosis_seed_1.json").write_text(
            json.dumps(packet), encoding="utf-8"
        )
    summary = summarize_diagnoses(tmp_path)
    assert summary["complete"] is True
    assert summary["observed_tasks"] == 22
    assert summary["selected_action_counts"] == {"feature-domain-shift": 22}
    assert "TDC-22 Discovery-only" in render_markdown(summary)


def test_policy_benchmark_is_same_budget_and_detects_paired_superiority(tmp_path):
    protocol = default_protocol()
    protocol.validate()
    payload = write_default_protocol(tmp_path / "protocol.json")
    rows = []
    for task in TASKS.values():
        control_score = 0.5
        agent_score = 0.49 if task.metric == "mae" else 0.51
        rows.extend(
            [
                PolicyTaskResult(
                    task.name, "agent", agent_score, 0.01, 100.0, 8, payload["sha256"]
                ),
                PolicyTaskResult(
                    task.name,
                    "random",
                    control_score,
                    0.02,
                    100.0,
                    8,
                    payload["sha256"],
                ),
            ]
        )
    comparison = compare_policy_results(rows)
    assert comparison["comparisons"]["random"]["tasks"] == 22
    assert comparison["comparisons"]["random"]["superiority_supported"] is True
    assert comparison["comparisons"]["random"]["mean_compute_ratio"] == pytest.approx(
        1.0
    )


def test_hidden_fault_benchmark_enforces_critical_safety_gate():
    rows = [
        FaultChallengeResult(
            "leak-1",
            "hERG",
            ("exact_split_overlap",),
            ("exact_split_overlap",),
            0.1,
            1,
            1,
        ),
        FaultChallengeResult(
            "invalid-1",
            "AMES",
            ("invalid_molecular_structure",),
            ("invalid_molecular_structure",),
            0.2,
            1,
            1,
        ),
    ]
    score = score_hidden_faults(rows)
    assert score["safety_gate_passed"] is True
    assert score["overall_score"] == pytest.approx(1.0)
    missed = [
        FaultChallengeResult("leak-2", "hERG", ("exact_split_overlap",), (), 0.1, 1, 0)
    ]
    assert score_hidden_faults(missed)["overall_score"] <= 0.5


def test_hidden_fault_benchmark_rewards_a_clean_negative_control():
    rows = [
        FaultChallengeResult("clean", "HIA_Hou", (), (), 0.01, 0, 0),
        FaultChallengeResult(
            "critical",
            "HIA_Hou",
            ("wrong_metric_direction",),
            ("wrong_metric_direction",),
            0.01,
            1,
            1,
        ),
    ]
    score = score_hidden_faults(rows)
    assert score["macro_case_f1"] == pytest.approx(1.0)
    assert score["remediation_validity"] == pytest.approx(1.0)


def test_fault_detector_checks_metric_direction_and_required_provenance():
    train, valid = _classification_frames()
    card = profile_hardness(train, valid, resolve_task("HIA_Hou"), "Drug", "Y", seed=1)
    report = ScientificFaultDetector().detect(
        card,
        endpoint_contract("HIA_Hou"),
        reported_metric_direction="minimize",
        source_manifest_required=True,
    )
    assert {"wrong_metric_direction", "provenance_missing"}.issubset(
        report.detected_types
    )


def test_hardness_context_promotes_balanced_model_for_extreme_imbalance(monkeypatch):
    monkeypatch.setattr("admet_agent.brain.is_model_available", lambda name: True)
    monkeypatch.setattr(
        "admet_agent.brain.is_representation_available", lambda name: False
    )
    brain = RuleBasedBrain(max_candidates=6)
    brain.set_context(
        cross_task_evidence=[],
        hardness_card={"dimensions": {"imbalance": {"severity": 0.8}}},
    )
    profile = TaskProfile(
        task="HIA_Hou",
        task_type="classification",
        metric="auroc",
        n_train=400,
        n_valid=50,
        positive_rate_train=0.9,
        target_mean=0.9,
        target_std=0.3,
        invalid_smiles_train=0,
        unique_scaffolds_train=200,
        unseen_valid_scaffold_rate=1.0,
    )
    proposals = brain.propose(profile, [])
    assert proposals[0].model == "balanced_extra_trees"


def test_feature_action_executes_a_deduplicated_matched_control(monkeypatch):
    monkeypatch.setattr("admet_agent.brain.is_model_available", lambda name: True)
    monkeypatch.setattr(
        "admet_agent.brain.is_representation_available", lambda name: False
    )
    brain = RuleBasedBrain(max_candidates=6)
    brain.set_context(
        cross_task_evidence=[],
        research_plan={"selected_action_id": "feature-domain-shift"},
    )
    profile = TaskProfile(
        task="HIA_Hou",
        task_type="classification",
        metric="auroc",
        n_train=400,
        n_valid=50,
        positive_rate_train=0.5,
        target_mean=0.5,
        target_std=0.5,
        invalid_smiles_train=0,
        unique_scaffolds_train=200,
        unseen_valid_scaffold_rate=1.0,
    )
    proposals = brain.propose(profile, [])
    pair = proposals[:2]
    assert pair[0].model == pair[1].model == "extra_trees"
    assert pair[0].hyperparameters == pair[1].hyperparameters
    assert pair[0].representations != pair[1].representations
    signatures = {
        (item.model, item.representations, tuple(sorted(item.hyperparameters.items())))
        for item in proposals
    }
    assert len(signatures) == len(proposals)


def test_unimol2_requires_hardness_prior_and_matched_2d_control(monkeypatch):
    monkeypatch.setattr("admet_agent.brain.is_model_available", lambda name: True)
    monkeypatch.setattr(
        "admet_agent.brain.is_representation_available", lambda name: True
    )
    brain = RuleBasedBrain(max_candidates=6)
    brain.set_context(
        cross_task_evidence=[],
        hardness_card={
            "dimensions": {
                "domain_shift": {"severity": 0.8, "confidence": 0.9},
            }
        },
        research_plan={
            "selected_action_id": "feature-domain-shift",
            "actions": [
                {
                    "action_id": "feature-domain-shift",
                    "prior_evidence": {
                        "mean_oriented_delta": 0.01,
                        "posterior_support_probability": 0.6,
                        "routing_eligible": True,
                    },
                }
            ],
        },
    )
    profile = TaskProfile(
        task="HIA_Hou",
        task_type="classification",
        metric="auroc",
        n_train=400,
        n_valid=50,
        positive_rate_train=0.5,
        target_mean=0.5,
        target_std=0.5,
        invalid_smiles_train=0,
        unique_scaffolds_train=200,
        unseen_valid_scaffold_rate=1.0,
    )
    proposals = brain.propose(profile, [])
    assert proposals[0].representations == ("unimol2", "morgan", "rdkit2d")
    assert proposals[1].representations == ("morgan", "rdkit2d")
    assert proposals[0].model == proposals[1].model == "extra_trees"

    brain.set_context(
        cross_task_evidence=[],
        hardness_card={
            "dimensions": {
                "domain_shift": {"severity": 0.8, "confidence": 0.4},
            }
        },
        research_plan={"selected_action_id": "feature-domain-shift"},
    )
    assert all(
        "unimol2" not in item.representations for item in brain.propose(profile, [])
    )


def test_model_estimator_family_is_a_cheap_matched_pair(monkeypatch):
    monkeypatch.setattr("admet_agent.brain.is_model_available", lambda name: True)
    brain = RuleBasedBrain(max_candidates=2)
    brain.set_context(
        cross_task_evidence=[],
        hardness_card={"dimensions": {}},
        research_plan={"selected_action_id": "model-estimator-family"},
    )
    profile = TaskProfile(
        task="HIA_Hou",
        task_type="classification",
        metric="auroc",
        n_train=400,
        n_valid=50,
        positive_rate_train=0.5,
        target_mean=0.5,
        target_std=0.5,
        invalid_smiles_train=0,
        unique_scaffolds_train=200,
        unseen_valid_scaffold_rate=1.0,
    )
    treatment, control = brain.propose(profile, [])
    assert treatment.model == "random_forest"
    assert control.model == "extra_trees"
    assert treatment.representations == control.representations
    assert "unimol2" not in treatment.representations
    plan = {
        "selected_action_id": "model-estimator-family",
        "actions": [
            {
                "action_id": "model-estimator-family",
                "axis": "model",
                "payload": {"candidate_model": "random_forest"},
            }
        ],
    }
    assert selected_action_pair(plan, {"treatment": treatment, "control": control}) == (
        "treatment",
        "control",
    )


def test_ensemble_evidence_requires_material_seed_consistent_gain():
    supported = _ensemble_evidence(
        [0.82, 0.83, 0.84, 0.85, 0.80],
        [0.80, 0.81, 0.82, 0.83, 0.81],
        [0.79, 0.80, 0.81, 0.82, 0.78],
        maximize=True,
    )
    assert supported["supported"] is True
    assert supported["positive_delta_rate"] == 0.8

    unstable = _ensemble_evidence(
        [0.82, 0.83, 0.84, 0.80, 0.79],
        [0.80, 0.81, 0.82, 0.83, 0.81],
        [0.79, 0.80, 0.81, 0.82, 0.78],
        maximize=True,
    )
    assert unstable["supported"] is False

    mae = _ensemble_evidence(
        [0.39, 0.38, 0.37, 0.36, 0.35],
        [0.41, 0.40, 0.39, 0.38, 0.37],
        [0.42, 0.41, 0.40, 0.39, 0.38],
        maximize=False,
    )
    assert mae["supported"] is True


def test_ensemble_persistence_keeps_one_best_blend_per_member_pair():
    def candidate(run_id: str, value: float, members: list[str]):
        result = RunResult(
            run_id,
            "HIA_Hou",
            1,
            "ensemble",
            "auroc",
            value,
            0.01,
            hyperparameters={"members": members},
        )
        return result, {"members": members}, {}

    winners = _select_distinct_ensemble_winners(
        [
            candidate("same-pair-weaker", 0.81, ["a", "b"]),
            candidate("different-pair", 0.82, ["a", "c"]),
            candidate("same-pair-best", 0.83, ["b", "a"]),
            candidate("third-pair", 0.80, ["d", "e"]),
        ],
        maximize=True,
        limit=3,
    )

    assert [item[0].run_id for item in winners] == [
        "same-pair-best",
        "different-pair",
        "third-pair",
    ]


def test_robust_promotion_keeps_strong_same_model_recipe_and_drops_dominated_one():
    task_name = "VDss_Lombardo"

    def spec(run_id, model, hypothesis):
        return ExperimentSpec(
            task=task_name,
            seed=1,
            model=model,
            representations=("morgan",),
            hypothesis=hypothesis,
        )

    experiments = {
        "treatment": spec(
            "treatment",
            "extra_trees",
            "domain-shift feature intervention: transferable multi-view features",
        ),
        "control": spec(
            "control",
            "extra_trees",
            "matched feature control: Morgan fingerprint only",
        ),
        "tuned-fast": spec("tuned-fast", "extra_trees", "strong tuned recipe"),
        "tuned-slow": spec("tuned-slow", "extra_trees", "dominated tuned recipe"),
        "xgb": spec("xgb", "xgboost", "xgboost baseline"),
        "rf": spec("rf", "random_forest", "random forest baseline"),
        "cat": spec("cat", "catboost", "catboost baseline"),
        "lgb": spec("lgb", "lightgbm", "lightgbm baseline"),
    }
    scores_and_costs = {
        "treatment": (0.703, 40.0),
        "control": (0.434, 16.0),
        "tuned-fast": (0.752, 202.0),
        "tuned-slow": (0.705, 892.0),
        "xgb": (0.602, 4.0),
        "rf": (0.582, 12.0),
        "cat": (0.560, 31.0),
        "lgb": (0.503, 4.0),
    }
    completed = [
        RunResult(
            run_id,
            task_name,
            1,
            experiments[run_id].model,
            "spearman",
            score,
            elapsed,
        )
        for run_id, (score, elapsed) in scores_and_costs.items()
    ]

    promoted = _select_robust_candidates(
        resolve_task(task_name), completed, experiments, limit=6
    )
    promoted_ids = {item.run_id for item in promoted}

    assert promoted_ids == {
        "treatment",
        "control",
        "tuned-fast",
        "xgb",
        "rf",
        "cat",
    }
    assert "tuned-slow" not in promoted_ids


def test_robust_promotion_keeps_imbalance_treatment_and_matched_model_control():
    task_name = "CYP2C9_Substrate_CarbonMangels"

    def spec(model, *, imbalance="none"):
        return ExperimentSpec(
            task=task_name,
            seed=1,
            model=model,
            representations=("morgan", "avalon", "erg", "rdkit2d"),
            imbalance=imbalance,
            hypothesis=f"{model} candidate",
        )

    experiments = {
        "treatment": spec("balanced_extra_trees", imbalance="class_weight"),
        "control": spec("extra_trees"),
        "lgb": spec("lightgbm"),
        "xgb": spec("xgboost"),
        "rf": spec("random_forest"),
        "cat": spec("catboost"),
        "svm": spec("svm_rbf"),
    }
    scores = {
        "treatment": 0.41,
        "control": 0.40,
        "lgb": 0.61,
        "xgb": 0.60,
        "rf": 0.59,
        "cat": 0.58,
        "svm": 0.39,
    }
    completed = [
        RunResult(
            run_id,
            task_name,
            1,
            experiments[run_id].model,
            "auprc",
            value,
            1.0,
        )
        for run_id, value in scores.items()
    ]

    promoted = _select_robust_candidates(
        resolve_task(task_name), completed, experiments, limit=6
    )
    promoted_ids = {item.run_id for item in promoted}

    assert {"treatment", "control"}.issubset(promoted_ids)


def test_candidate_hard_timeout_interrupts_and_restores_signal_handler():
    previous = _arm_candidate_timeout(0.02)
    assert previous is not None
    try:
        with pytest.raises(CandidateTimeoutError, match="hard wall-time limit"):
            time.sleep(0.2)
    finally:
        _disarm_candidate_timeout(previous)

    # A completed timeout must not leave a latent alarm that can corrupt the
    # next candidate or the test process.
    time.sleep(0.03)


def test_scaleup_soft_stop_uses_matching_pilot_and_remaining_budget():
    task_name = "LD50_Zhu"
    pilot = ExperimentSpec(
        task=task_name,
        seed=1,
        model="random_forest",
        representations=("morgan", "avalon", "erg", "rdkit2d"),
        hyperparameters={
            "criterion": "absolute_error",
            "n_estimators": 500,
            "min_samples_leaf": 5,
        },
        hypothesis="pilot",
    )
    candidate = ExperimentSpec(
        task=task_name,
        seed=1,
        model="random_forest",
        representations=("morgan", "avalon", "erg", "rdkit2d"),
        hyperparameters={
            "criterion": "absolute_error",
            "n_estimators": 1200,
            "min_samples_leaf": 2,
        },
        hypothesis="scale-up",
    )
    history = [
        RunResult(
            "pilot-run",
            task_name,
            1,
            "random_forest",
            "mae",
            0.48,
            1265.0,
        )
    ]

    reason = _scaleup_stop_reason(
        candidate,
        history,
        {"pilot-run": pilot},
        remaining_seconds=2280.0,
    )

    assert reason is not None
    assert "linear estimate" in reason
    assert (
        _scaleup_stop_reason(
            candidate,
            history,
            {"pilot-run": pilot},
            remaining_seconds=3600.0,
        )
        is None
    )


def test_matched_action_evidence_requires_paired_multiseed_effect():
    treatment = ExperimentSpec(
        task="HIA_Hou",
        seed=1,
        model="extra_trees",
        representations=("morgan", "avalon", "erg", "rdkit2d"),
        hypothesis="domain-shift feature intervention: transferable multi-view features",
    )
    control = ExperimentSpec(
        task="HIA_Hou",
        seed=1,
        model="extra_trees",
        representations=("morgan",),
        hypothesis="matched feature control: Morgan fingerprint only",
    )
    plan = {
        "selected_action_id": "feature-domain-shift",
        "actions": [
            {
                "action_id": "feature-domain-shift",
                "axis": "feature",
                "payload": {"representations": ["morgan", "rdkit2d"]},
            }
        ],
    }

    def result(run_id, seed, value):
        return RunResult(
            run_id,
            "HIA_Hou",
            seed,
            "extra_trees",
            "auroc",
            value,
            1.0,
        )

    evidence = evaluate_action_evidence(
        plan,
        resolve_task("HIA_Hou"),
        {"treatment": treatment, "control": control},
        {
            "treatment": {
                seed: result(f"t-{seed}", seed, value)
                for seed, value in enumerate((0.72, 0.74, 0.73), 1)
            },
            "control": {
                seed: result(f"c-{seed}", seed, value)
                for seed, value in enumerate((0.70, 0.71, 0.70), 1)
            },
        },
    )
    assert evidence.status == "supported"
    assert evidence.single_axis_verified is True
    assert evidence.mean_oriented_delta == pytest.approx(0.0266666667)
    assert evidence.positive_delta_rate == pytest.approx(1.0)


def test_matched_action_evidence_rejects_mean_gain_with_unstable_direction():
    treatment = ExperimentSpec(
        task="AMES",
        seed=1,
        model="extra_trees",
        representations=("morgan", "avalon", "erg", "rdkit2d"),
        hypothesis="domain-shift feature intervention: transferable multi-view features",
    )
    control = ExperimentSpec(
        task="AMES",
        seed=1,
        model="extra_trees",
        representations=("morgan",),
        hypothesis="matched feature control: Morgan fingerprint only",
    )
    plan = {
        "selected_action_id": "feature-domain-shift",
        "actions": [
            {
                "action_id": "feature-domain-shift",
                "axis": "feature",
                "payload": {"representations": ["morgan", "rdkit2d"]},
            }
        ],
    }

    def result(run_id, seed, value):
        return RunResult(run_id, "AMES", seed, "extra_trees", "auroc", value, 1.0)

    control_values = (0.80, 0.80, 0.80, 0.80, 0.80)
    deltas = (0.01, 0.044, -0.007, -0.004, 0.006)
    evidence = evaluate_action_evidence(
        plan,
        resolve_task("AMES"),
        {"treatment": treatment, "control": control},
        {
            "treatment": {
                seed: result(f"t-{seed}", seed, base + delta)
                for seed, (base, delta) in enumerate(zip(control_values, deltas), 1)
            },
            "control": {
                seed: result(f"c-{seed}", seed, value)
                for seed, value in enumerate(control_values, 1)
            },
        },
    )

    assert evidence.mean_oriented_delta == pytest.approx(0.0098)
    assert evidence.positive_delta_rate == pytest.approx(0.6)
    assert evidence.status == "inconclusive"


def test_action_memory_shrinks_and_deduplicates_endpoint_evidence(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "action_outcomes.jsonl")
    base = {
        "action_id": "model-imbalance-aware",
        "axis": "model",
        "status": "supported",
        "mean_oriented_delta": 0.01,
        "delta_std": 0.002,
        "oriented_seed_deltas": [0.01, 0.012, 0.008],
        "single_axis_verified": True,
    }
    memory.append(task="HIA_Hou", category="absorption", evidence=base)
    memory.append(
        task="HIA_Hou",
        category="absorption",
        evidence={**base, "status": "refuted", "mean_oriented_delta": -0.01},
    )
    prior = memory.priors("absorption")["model-imbalance-aware"]
    assert prior["observations"] == 1
    assert prior["transport_eligible_observations"] == 0
    assert prior["refuted"] == 0
    assert prior["observational_posterior_support_probability"] == pytest.approx(0.4)
    assert prior["posterior_support_probability"] == pytest.approx(0.5)
    assert prior["routing_eligible"] is False
    assert prior["effect_multiplier"] == pytest.approx(1.0)


def test_action_memory_routes_only_from_prospective_transport_design(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "action_outcomes.jsonl")
    base = {
        "action_id": "feature-domain-shift",
        "axis": "feature",
        "mean_oriented_delta": 0.01,
        "delta_std": 0.002,
        "oriented_seed_deltas": [0.01, 0.012, 0.008],
        "single_axis_verified": True,
        "cross_task_transport": {
            "prospective": True,
            "same_budget": True,
            "assignment_design": "randomized_action_assignment",
            "protocol_sha256": "b" * 64,
        },
    }
    for index, status in enumerate(
        ("supported", "supported", "supported", "refuted", "refuted")
    ):
        memory.append(
            task=f"transport-task-{index}",
            category="absorption",
            evidence={**base, "status": status},
        )
    prior = memory.priors("absorption")["feature-domain-shift"]
    assert prior["transport_eligible_observations"] == 5
    assert prior["supported"] == 3
    assert prior["refuted"] == 2
    assert prior["routing_eligible"] is True
    assert prior["routing_eligibility_blockers"] == ()
    assert prior["posterior_support_probability"] == pytest.approx(5 / 9)
    assert prior["effect_multiplier"] == pytest.approx(0.5 + 5 / 9)


def test_global_transport_scope_pools_across_admet_categories(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "action_outcomes.jsonl")
    categories = ("absorption", "distribution", "metabolism", "excretion", "toxicity")
    for index, (category, status) in enumerate(
        zip(categories, ("refuted", "refuted", "refuted", "refuted", "inconclusive"))
    ):
        memory.append(
            task=f"global-task-{index}",
            category=category,
            evidence={
                "action_id": "model-estimator-family",
                "axis": "model",
                "status": status,
                "mean_oriented_delta": -0.01,
                "oriented_seed_deltas": [-0.01] * 5,
                "single_axis_verified": True,
                "cross_task_transport": {
                    "prospective": True,
                    "same_budget": True,
                    "assignment_design": "complete_action_grid",
                    "protocol_sha256": "c" * 64,
                    "scope": "global_admet",
                },
            },
        )
    prior = memory.priors("absorption")["model-estimator-family"]
    assert prior["transport_eligible_observations"] == 5
    assert prior["routing_eligible"] is True
    assert prior["posterior_support_probability"] == pytest.approx(2.5 / 9)


def test_all_inconclusive_transport_outcomes_stay_neutral(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "action_outcomes.jsonl")
    for index in range(5):
        memory.append(
            task=f"inconclusive-task-{index}",
            category="toxicity",
            evidence={
                "action_id": "data-quality-filter",
                "axis": "data",
                "status": "inconclusive",
                "mean_oriented_delta": 0.0,
                "oriented_seed_deltas": [0.0] * 5,
                "single_axis_verified": True,
                "cross_task_transport": {
                    "prospective": True,
                    "same_budget": True,
                    "assignment_design": "complete_action_grid",
                    "protocol_sha256": "d" * 64,
                    "scope": "global_admet",
                },
            },
        )
    prior = memory.priors("absorption")["data-quality-filter"]
    assert prior["routing_eligible"] is False
    assert prior["posterior_support_probability"] == pytest.approx(0.5)
    assert prior["effect_multiplier"] == pytest.approx(1.0)


def test_action_memory_rejects_malformed_transport_metadata(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "action_outcomes.jsonl")
    evidence = {
        "action_id": "feature-domain-shift",
        "axis": "feature",
        "status": "supported",
        "single_axis_verified": True,
        "cross_task_transport": "prospective",
    }
    with pytest.raises(TypeError, match="cross_task_transport must be a mapping"):
        memory.append(task="HIA_Hou", category="absorption", evidence=evidence)


def test_observational_memory_cannot_change_action_ranking():
    train, valid = _classification_frames()
    card = profile_hardness(train, valid, resolve_task("HIA_Hou"), "Drug", "Y", seed=1)
    planner = EffectFirstResearchPlanner()
    plain = planner.plan(card)
    blocked = planner.plan(
        card,
        action_priors={
            plain["selected_action_id"]: {
                "effect_multiplier": 0.01,
                "routing_eligible": False,
                "observations": 22,
            }
        },
    )
    assert blocked["selected_action_id"] == plain["selected_action_id"]
    assert [row["action_id"] for row in blocked["actions"]] == [
        row["action_id"] for row in plain["actions"]
    ]


def test_action_prior_calibrates_route_without_overriding_raw_evidence():
    train, valid = _classification_frames()
    card = profile_hardness(train, valid, resolve_task("HIA_Hou"), "Drug", "Y", seed=1)
    planner = EffectFirstResearchPlanner()
    plain = planner.plan(card)
    calibrated = planner.plan(
        card,
        action_priors={
            plain["selected_action_id"]: {
                "effect_multiplier": 0.8,
                "observations": 3,
                "routing_eligible": True,
            }
        },
    )
    action = next(
        item
        for item in calibrated["actions"]
        if item["action_id"] == plain["selected_action_id"]
    )
    assert action["calibrated_expected_effect"] <= action["expected_effect"]
    assert action["prior_evidence"]["observations"] == 3


def test_prospective_negative_memory_prunes_refuted_action():
    planner = EffectFirstResearchPlanner()
    action = ResearchAction(
        "model-estimator-family",
        "model",
        "RandomForest may help.",
        "Compare against ExtraTrees.",
        ("roughness",),
        0.4,
        0.7,
        0.3,
        0.1,
        {"candidate_model": "random_forest"},
    )
    candidate = planner.route([action]).ranked_actions[0]
    gated = planner.route(
        [action],
        action_priors={
            "model-estimator-family": {
                "routing_eligible": True,
                "transport_eligible_observations": 5,
                "refuted": 4,
                "posterior_support_probability": 0.278,
                "effect_multiplier": 0.778,
                "negative_gate_confirmed": True,
            }
        },
    )
    blocked = gated.ranked_actions[0]
    assert candidate["routing_admitted"] is True
    assert blocked["routing_admitted"] is False
    assert blocked["routing_block_reason"] == "prospective_transport_refutation_gate"
    assert blocked["utility"] == pytest.approx(-1.0)


def test_unconfirmed_global_negative_memory_cannot_prune():
    planner = EffectFirstResearchPlanner()
    action = ResearchAction(
        "model-estimator-family",
        "model",
        "RandomForest may help.",
        "Compare against ExtraTrees.",
        ("roughness",),
        0.4,
        0.7,
        0.3,
        0.1,
        {"candidate_model": "random_forest"},
    )
    decision = planner.route(
        [action],
        action_priors={
            "model-estimator-family": {
                "routing_eligible": True,
                "transport_eligible_observations": 5,
                "refuted": 4,
                "posterior_support_probability": 0.278,
                "effect_multiplier": 0.778,
                "negative_gate_confirmed": False,
            }
        },
    )
    row = decision.ranked_actions[0]
    assert row["routing_admitted"] is True
    assert row["routing_block_reason"] is None


def test_executable_policy_controls_are_deterministic_and_axis_limited(monkeypatch):
    monkeypatch.setattr(
        "admet_agent.policy_brains.is_model_available", lambda name: True
    )
    profile = TaskProfile(
        task="HIA_Hou",
        task_type="classification",
        metric="auroc",
        n_train=400,
        n_valid=50,
        positive_rate_train=0.9,
        target_mean=0.9,
        target_std=0.3,
        invalid_smiles_train=0,
        unique_scaffolds_train=200,
        unseen_valid_scaffold_rate=1.0,
    )
    random_a = benchmark_brain("random", policy_seed=7).propose(profile, [])
    random_b = benchmark_brain("random", policy_seed=7).propose(profile, [])
    assert [item.to_dict() for item in random_a] == [
        item.to_dict() for item in random_b
    ]
    assert len(random_a) == 6
    automl = benchmark_brain("automl").propose(profile, [])
    assert len(automl) == 6
    assert {item.representations for item in automl} == {
        ("morgan", "avalon", "erg", "rdkit2d")
    }
    fixed = benchmark_brain("strong_fixed").propose(profile, [])
    assert len(fixed) == 1


def test_global_freeze_audit_requires_complete_multiseed_discovery(tmp_path):
    run_dir = tmp_path / "run"
    protocol_path = tmp_path / "protocol.json"
    write_default_protocol(protocol_path)
    source = snapshot_source(run_dir)
    EvidenceLedger(run_dir / "evidence_ledger.jsonl").append(
        "discovery_started", {"test_loaded": False}
    )
    for task in TASKS.values():
        task_dir = run_dir / "tasks" / task.name
        task_dir.mkdir(parents=True)
        score = 0.5 if task.metric == "mae" else 0.8
        spec = ExperimentSpec(task=task.name, seed=1).to_dict()
        summary = {
            "profile": {"task": task.name, "metric": task.metric},
            "source_sha256": source["source_sha256"],
            "validation_seeds": [1, 2, 3, 4, 5],
            "research_plan": {
                "selected_action_id": "stop-no-defensible-action",
                "actions": [{"action_id": "stop-no-defensible-action", "axis": "stop"}],
            },
            "action_evidence": {
                "status": "not_executed",
                "single_axis_verified": False,
            },
            "results": [{"status": "completed"}],
            "robust_results": [
                {
                    "run_id": f"{task.name}-winner",
                    "validation_score": score,
                    "validation_std": 0.01,
                }
            ],
            "ensemble_results": [],
            "planner_errors": [],
            "finalist_run_id": f"{task.name}-winner",
            "finalist_spec": spec,
        }
        (task_dir / "search_seed_1.json").write_text(json.dumps(summary))
    audit_path = tmp_path / "audit.json"
    finalists_path = tmp_path / "finalists.json"
    audit = audit_global_freeze(run_dir, protocol_path, audit_path, finalists_path)
    assert audit["passed"] is True
    finalists = json.loads(finalists_path.read_text())
    assert verify_freeze_authorization(audit_path, finalists)["passed"] is True
    with pytest.raises(RuntimeError, match="Finalists do not match"):
        verify_freeze_authorization(audit_path, {**finalists, "HIA_Hou": {}})


def test_policy_result_collection_is_protocol_bound(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = write_default_protocol(protocol_path)
    run_dir = tmp_path / "agent"
    task_dir = run_dir / "tasks" / "HIA_Hou"
    task_dir.mkdir(parents=True)
    summary = {
        "benchmark_arm": "agent",
        "benchmark_protocol_sha256": protocol["sha256"],
        "validation_seeds": [1, 2, 3, 4, 5],
        "results": [
            {
                "seed": seed,
                "elapsed_seconds": 2.0,
            }
            for seed in (1, 2, 3, 4, 5)
        ],
        "robust_results": [
            {
                "run_id": "winner",
                "task": "HIA_Hou",
                "validation_score": 0.9,
                "validation_std": 0.01,
            }
        ],
        "ensemble_results": [],
        "finalist_run_id": "winner",
    }
    (task_dir / "search_seed_1.json").write_text(json.dumps(summary))
    payload = collect_policy_results(
        protocol_path, {"agent": run_dir}, tmp_path / "results.json"
    )
    assert payload["protocol_sha256"] == protocol["sha256"]
    assert payload["results"][0]["actions_attempted"] == 1
    assert payload["test_loaded"] is False


def test_production_source_snapshot_is_stable_and_replayable(tmp_path):
    first = snapshot_source(tmp_path)
    second = snapshot_source(tmp_path)
    assert first["source_sha256"] == second["source_sha256"]
    assert (tmp_path / "source_snapshot" / "admet_agent" / "runner.py").exists()
    assert (tmp_path / "source_snapshot" / "manifest.json").exists()


def test_cross_task_memory_does_not_count_seed_repeats_as_endpoints(tmp_path):
    database = ExperimentDatabase(tmp_path / "experiments.sqlite")
    for task_name, seeds in (("HIA_Hou", (1, 2, 3)), ("Pgp_Broccatelli", (1,))):
        profile = TaskProfile(
            task=task_name,
            task_type="classification",
            metric="auroc",
            n_train=100,
            n_valid=20,
            positive_rate_train=0.5,
            target_mean=0.5,
            target_std=0.5,
            invalid_smiles_train=0,
            unique_scaffolds_train=50,
            unseen_valid_scaffold_rate=0.8,
        )
        for seed in seeds:
            spec = ExperimentSpec(task=task_name, seed=seed, model="extra_trees")
            result = RunResult(
                f"{task_name}-{seed}",
                task_name,
                seed,
                "extra_trees",
                "auroc",
                0.8,
                1.0,
            )
            database.record(profile, spec, result)
    insight = database.cross_task_insights("classification", "auroc")[0]
    assert insight["evidence_tasks"] == 2
