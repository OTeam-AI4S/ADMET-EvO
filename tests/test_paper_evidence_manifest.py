import pytest

from scripts.build_paper_evidence_manifest import (
    manifest_digest,
    validate_firewall,
)


def test_validation_firewall_accepts_explicit_no_test_declaration():
    assert validate_firewall(
        "report",
        {"official_test_loaded": False, "leaderboard_used": False},
    ) == {"official_test_loaded": False, "leaderboard_used": False}


@pytest.mark.parametrize(
    "payload",
    [
        {"official_test_loaded": True},
        {"test_labels_used": True},
        {"leaderboard_used": False},
        {},
    ],
)
def test_validation_firewall_fail_closes(payload):
    with pytest.raises(ValueError):
        validate_firewall("report", payload)


def test_manifest_digest_is_order_invariant_and_content_sensitive():
    assert manifest_digest({"a": 1, "b": 2}) == manifest_digest({"b": 2, "a": 1})
    assert manifest_digest({"a": 1}) != manifest_digest({"a": 2})
