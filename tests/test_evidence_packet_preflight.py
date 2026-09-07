import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preflight_evidence_packet_v4",
    ROOT / "scripts/preflight_evidence_packet_v4.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _registry(active_packets: list[dict]) -> dict:
    return {
        "schema_version": "admet-evidence-packet-registry-v4.1",
        "policy": {
            "active_packet_root": "data/evidence_packets_v4/packets",
            "legacy_bulk_pool_allowed": False,
        },
        "active_packets": active_packets,
        "quarantined_legacy_sources": [],
        "test_labels_loaded": False,
    }


def test_empty_registry_blocks_experiment_without_fabricating_data(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(_registry([])), encoding="utf-8")
    report = MODULE.preflight(path)
    assert report["status"] == "blocked_no_modeling_ready_packets"
    assert report["formal_experiment_allowed"] is False
    assert report["active_record_count"] == 0
    assert report["legacy_bulk_pool_used"] is False


def test_registry_rejects_legacy_file_outside_packet_root(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            _registry(
                [
                    {
                        "packet_id": "legacy-bulk",
                        "task": "hERG",
                        "manifest_path": "data/ADMET_v3_clean_external_candidate_pool.csv",
                        "manifest_sha256": "0" * 64,
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    report = MODULE.preflight(path)
    assert report["status"] == "failed_closed"
    assert report["formal_experiment_allowed"] is False
    assert "outside the active packet root" in report["failures"][0]["message"]
