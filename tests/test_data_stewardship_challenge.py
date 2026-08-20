from admet_agent.data_stewardship_challenge import (
    build_sealed_data_key,
    compare_data_stewardship_controls,
    default_data_dossiers,
    public_dossier_manifest,
)


def test_public_dossiers_are_committed_and_gold_answers_remain_hidden():
    dossiers = default_data_dossiers()
    key = build_sealed_data_key(dossiers, reveal_nonce="secret")
    manifest = public_dossier_manifest(dossiers, key)
    assert len(dossiers) == 11
    assert manifest["answers_revealed"] is False
    assert "answers" not in manifest
    assert len(manifest["answer_commitment_sha256"]) == 64
    assert len(manifest["dossiers_sha256"]) == 64
    assert manifest["test_loaded"] is False


def test_data_controls_reveal_automl_safety_gap_and_complete_rules_tie():
    report = compare_data_stewardship_controls()
    agent = report["arms"]["agent_contract_steward"]
    rules = report["arms"]["complete_rules_steward"]
    expanded = report["arms"]["expanded_automl_schema_steward"]
    automl = report["arms"]["conventional_automl_direct_fit"]
    assert agent["decision_accuracy"] == 1.0
    assert agent["issue_f1"] == 1.0
    assert agent["unsafe_merge_count"] == 0
    assert rules["decision_accuracy"] == agent["decision_accuracy"]
    assert rules["issue_f1"] == agent["issue_f1"]
    assert expanded["unsafe_merge_count"] > 0
    assert expanded["safety_gate_passed"] is False
    assert automl["unsafe_merge_count"] == 10
    assert automl["safety_gate_passed"] is False
    assert report["test_loaded"] is False


def test_data_control_report_is_deterministic():
    assert compare_data_stewardship_controls(policy_seed=3) == (
        compare_data_stewardship_controls(policy_seed=3)
    )
