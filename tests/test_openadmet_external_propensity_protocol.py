import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_propensity_protocol_is_self_hashed_auxiliary_only_and_blind():
    path = ROOT / "reports/openadmet_external_propensity_protocol_v1.json"
    protocol = json.loads(path.read_text())
    claimed = protocol.pop("sha256")
    assert claimed == hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    assert protocol["exact_primary_endpoint_merge"] is False
    assert protocol["test_labels_loaded"] is False
    assert protocol["leaderboard_feedback_used"] is False
    assert protocol["blind_test_can_override"] is False
    assert protocol["sealed_certification_can_override"] is False
    assert protocol["outer_folds"] == [0, 1, 2, 3, 4]
    assert protocol["seeds"] == [1, 2, 3]
    assert protocol["implementation_source_sha256"] == _sha256(
        ROOT / "scripts/run_openadmet_external_propensity_bridge.py"
    )
