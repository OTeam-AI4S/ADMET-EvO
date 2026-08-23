import json

import pytest

from admet_agent.cli import parser
from admet_agent.data_skill import ADMETDataCollectorSkill, skill_descriptor
from admet_agent.registry import planner_catalog


class _StaticScout:
    def discover(self, task_name):
        return {
            "task": task_name,
            "candidate_count": 1,
            "candidates": [{"candidate": {"source_id": "source-1"}}],
            "connector_failures": [],
            "automatic_download_allowed": False,
            "test_labels_loaded": False,
        }


def test_packaged_data_skill_is_visible_to_planners_with_hashed_resources():
    descriptor = skill_descriptor()
    assert descriptor["name"] == "admet-data-collector"
    assert descriptor["modes"] == ["discover", "collect", "audit", "advise", "update"]
    assert descriptor["automatic_download_allowed"] is False
    assert len(descriptor["resource_bundle_sha256"]) == 64
    assert {item["name"] for item in planner_catalog()["skills"]} == {
        "admet-data-collector"
    }


def test_external_data_action_activates_discovery_without_admission():
    skill = ADMETDataCollectorSkill(_StaticScout())
    route = skill.route(
        "hERG",
        {
            "selected_action_id": "data-external-quarantine",
            "actions": [
                {
                    "action_id": "data-external-quarantine",
                    "axis": "data",
                    "payload": {"skill": "admet-data-collector"},
                }
            ],
        },
    )
    assert route["status"] == "candidate_register_ready"
    assert route["mode"] == "discover"
    assert route["automatic_download_allowed"] is False
    assert route["test_labels_loaded"] is False


def test_non_data_action_does_not_activate_collector():
    route = ADMETDataCollectorSkill(_StaticScout()).route(
        "AMES",
        {
            "selected_action_id": "replicate-multiseed",
            "actions": [
                {"action_id": "replicate-multiseed", "axis": "replicate"}
            ],
        },
    )
    assert route["status"] == "not_activated"


def test_bundle_init_uses_registered_endpoint_contract_and_audits_empty_draft(tmp_path):
    bundle = tmp_path / "herg-bundle"
    skill = ADMETDataCollectorSkill(_StaticScout())
    initialized = skill.initialize_bundle(
        "hERG", bundle, si_choice="include"
    )
    assert initialized["endpoint_id"] == "herg_blockade"
    endpoint_text = (bundle / "contracts/endpoint_contract.yaml").read_text()
    assert 'endpoint_id: "herg_blockade"' in endpoint_text
    assert "TODO" not in endpoint_text
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["skill"] == "admet-data-collector"
    assert manifest["si_choice"] == "include"
    assert manifest["test_labels_loaded"] is False

    audit = skill.audit_bundle(bundle, strict=True)
    assert audit["summary"]["suggested_release_state"] == "pending_scientific_review"
    assert audit["strict_gate_passed"] is True
    assert audit["test_labels_loaded"] is False
    assert audit["modeling_admission"]["admitted_for_direct_training"] is False
    assert "no_direct_records" in audit["modeling_admission"]["reasons"]
    updated_manifest = json.loads((bundle / "manifest.json").read_text())
    assert updated_manifest["audit"]["status"] == "completed"
    assert updated_manifest["suggested_release_state"] == "pending_scientific_review"


def test_bundle_init_requires_predeclared_si_choice(tmp_path):
    with pytest.raises(ValueError, match="si_choice"):
        ADMETDataCollectorSkill().initialize_bundle(
            "hERG", tmp_path / "bundle", si_choice="later"
        )


def test_data_skill_cli_contract_requires_subcommand_and_si_choice():
    parsed = parser().parse_args(
        [
            "data-skill",
            "init",
            "--task",
            "hERG",
            "--out",
            "bundle",
            "--si-choice",
            "exclude",
        ]
    )
    assert parsed.command == "data-skill"
    assert parsed.data_command == "init"
    assert parsed.si_choice == "exclude"
