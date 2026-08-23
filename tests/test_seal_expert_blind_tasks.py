import json

import pytest

from scripts.seal_expert_blind_tasks import PROTOCOL, seal, verify_reveal


def _packet():
    return {
        "protocol": PROTOCOL,
        "case_id": "blind-001",
        "category": "identity_leakage",
        "visible_packet": {
            "endpoint_contract": {"endpoint": "CYP2D6 inhibition"},
            "rows": [{"smiles": "CC", "split": "train"}],
            "permitted_resources": ["visible rows only"],
        },
        "gold_packet": {
            "hidden_fault": "same identity crosses split",
            "acceptable_actions": ["group by standardized identity"],
            "unsafe_actions": ["read hidden labels"],
        },
        "author_attestation": {
            "author_pseudonym": "expert-a",
            "independent_of_agent_development": True,
            "did_not_view_agent_outputs": True,
            "gold_authored_before_release": True,
        },
    }


def test_seal_excludes_gold_and_reveal_verifies(tmp_path):
    hidden = tmp_path / "hidden.json"
    public = tmp_path / "public.json"
    commitment = tmp_path / "commitment.json"
    hidden.write_text(json.dumps(_packet()))
    result = seal(hidden, public, commitment)
    visible = json.loads(public.read_text())
    assert result["contains_gold"] is False
    assert "gold_packet" not in visible
    assert "hidden_fault" not in public.read_text()
    assert verify_reveal(hidden, public, commitment)["verified"] is True


def test_visible_answer_key_is_rejected(tmp_path):
    packet = _packet()
    packet["visible_packet"]["answer"] = "leaked"
    hidden = tmp_path / "hidden.json"
    hidden.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="hidden-answer"):
        seal(hidden, tmp_path / "public.json", tmp_path / "commitment.json")


def test_tampered_hidden_packet_fails_reveal(tmp_path):
    hidden = tmp_path / "hidden.json"
    public = tmp_path / "public.json"
    commitment = tmp_path / "commitment.json"
    hidden.write_text(json.dumps(_packet()))
    seal(hidden, public, commitment)
    packet = _packet()
    packet["gold_packet"]["hidden_fault"] = "tampered"
    hidden.write_text(json.dumps(packet))
    result = verify_reveal(hidden, public, commitment)
    assert result["verified"] is False
    assert result["checks"]["hidden_hash_matches"] is False
