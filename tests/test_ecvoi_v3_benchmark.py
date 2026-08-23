from __future__ import annotations

from admet_agent.ecvoi_v3_benchmark import protocol_payload, validate_protocol


def test_v3_development_protocol_has_shared_anchor_and_strict_boundary():
    payload = protocol_payload("development")
    assert payload["tasks"] == [
        "BBB_Martins",
        "CYP2C9_Veith",
        "CYP2D6_Veith",
    ]
    invariants = payload["frozen_invariants"]
    assert invariants["shared_anchor_mapping_sha256"] == invariants[
        "tdc22_mapping_sha256"
    ]
    assert invariants["mandatory_discovery_data_policy"] == "identity_safe"
    assert payload["pair_contract"]["one_safe_local_exploration_max"] == 1
    validate_protocol(payload)


def test_v3_confirmation_keeps_development_tasks_excluded():
    payload = protocol_payload("confirmation")
    assert len(payload["tasks"]) == 19
    assert not set(payload["tasks"]) & {
        "BBB_Martins",
        "CYP2C9_Veith",
        "CYP2D6_Veith",
    }
