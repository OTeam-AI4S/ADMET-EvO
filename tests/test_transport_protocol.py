import json

import pytest

from admet_agent.transport_protocol import (
    build_transport_protocol,
    load_transport_protocol,
    validate_transport_protocol,
)
from scripts.run_transport_grid import _specs


def test_transport_protocol_is_complete_grid_and_test_blind(tmp_path):
    payload = build_transport_protocol("a" * 64)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    loaded = load_transport_protocol(path)
    assert len(loaded["sentinel_endpoints"]) == 5
    assert {row["axis"] for row in loaded["actions"]} == {
        "feature",
        "model",
        "data",
    }
    assert loaded["matched_episodes"] == 15
    assert loaded["maximum_total_fits"] == 100
    assert loaded["shared_control_deduplication"] is True
    assert loaded["test_feedback_to_discovery"] is False


def test_transport_protocol_rejects_partial_grid():
    payload = build_transport_protocol("a" * 64)
    payload.pop("sha256")
    payload["actions"] = payload["actions"][:2]
    with pytest.raises(ValueError, match="Feature, model and data"):
        validate_transport_protocol(payload)


def test_transport_protocol_rejects_confounded_pair():
    payload = build_transport_protocol("a" * 64)
    payload.pop("sha256")
    payload["actions"][0]["treatment"]["model"] = "random_forest"
    with pytest.raises(ValueError, match="not a single-axis matched contrast"):
        validate_transport_protocol(payload)


def test_transport_runner_deduplicates_shared_control():
    protocol = build_transport_protocol("a" * 64, "b" * 64)
    specs = _specs(protocol, "Caco2_Wang", 1, 4)
    assert set(specs) == {
        "shared-control",
        "treatment:data-quality-filter",
        "treatment:feature-domain-shift",
        "treatment:model-estimator-family",
    }
    control = specs["shared-control"]
    assert control.model == "extra_trees"
    assert control.data_policy == "official"
    assert specs["treatment:feature-domain-shift"].representations != (
        control.representations
    )
    assert specs["treatment:model-estimator-family"].model == "random_forest"
    assert specs["treatment:data-quality-filter"].data_policy == "quality_filtered"
