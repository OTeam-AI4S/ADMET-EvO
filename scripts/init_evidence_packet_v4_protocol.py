#!/usr/bin/env python3
"""Freeze the new Evidence Packet data-axis protocol and historical registry."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "reports/evidence_packet_data_axis_v4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discovery_data_summary(version: str) -> dict[str, Any] | None:
    source = ROOT / f"reports/all22_three_axis_ablation_{version}/discovery"
    paths = sorted(source.glob("*.json"))
    if len(paths) != 22:
        return None
    values = []
    promoted = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload["selected"]["data"]["D3"]
        value = float(selected["development"]["mean_normalized_improvement"])
        values.append(value)
        if selected.get("promotion", {}).get("passed"):
            promoted.append(
                {
                    "task": payload["task"],
                    "mean_normalized_improvement": value,
                    "config": selected["config"],
                }
            )
    return {
        "source_directory": str(source.resolve()),
        "task_count": len(values),
        "mean_normalized_improvement": sum(values) / len(values),
        "median_normalized_improvement": statistics.median(values),
        "promoted_tasks": promoted,
    }


def _final_axis_summary(version: str) -> dict[str, Any] | None:
    path = (
        ROOT
        / f"reports/all22_three_axis_ablation_{version}"
        / f"all22_three_axis_ablation_{version}_results.audit.json"
    )
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path.resolve()),
        "D3": payload.get("aggregate_axis_ladders", {}).get("D3"),
        "data_policy": payload.get("data_policy"),
        "all_certifications_complete": payload.get("all_certifications_complete"),
    }


def main() -> None:
    registry = ROOT / "data/evidence_packets_v4/registry.json"
    implementation = [
        ROOT / "admet_agent/evidence_packets.py",
        ROOT / "admet_agent/evidence_packet_splits.py",
        ROOT / "admet_agent/evidence_packet_loop.py",
        ROOT / "admet_agent/evidence_packet_closed_loop.py",
        ROOT / "admet_agent/external_structure_firewall.py",
        ROOT / "admet_agent/features.py",
        ROOT / "admet_agent/selection.py",
        ROOT / "admet_agent/schemas.py",
        ROOT / "admet_agent/tasks.py",
        ROOT / "scripts/preflight_evidence_packet_v4.py",
        ROOT / "scripts/plan_evidence_packet_acquisition_v4.py",
        ROOT / "scripts/prepare_evidence_packet_training_v4.py",
        ROOT / "scripts/run_evidence_packet_closed_loop_v4.py",
        ROOT / "scripts/init_evidence_packet_v4_protocol.py",
        ROOT / "tests/test_evidence_packets.py",
        ROOT / "tests/test_evidence_packet_loop.py",
        ROOT / "tests/test_evidence_packet_closed_loop.py",
        ROOT / "tests/test_evidence_packet_splits.py",
        ROOT / "tests/test_evidence_packet_preflight.py",
        ROOT / "tests/test_evidence_packet_acquisition.py",
    ]
    method_registry = {
        "protocol": "admet-data-axis-method-registry-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": {
            "v2_bulk_coreset": {
                "role": "historical_exploratory_ablation_only",
                "frozen_read_only": True,
                "discovery": _discovery_data_summary("v2"),
                "final_axis_summary": _final_axis_summary("v2"),
                "warning": "The approximately 10% all-task discovery mean is dominated by Half_Life_Obach; median is zero and this source policy is not reusable in v4.",
            },
            "v3_strict_bulk_pool": {
                "role": "historical_safety_ablation_only",
                "frozen_read_only": True,
                "discovery": _discovery_data_summary("v3"),
                "final_axis_summary": _final_axis_summary("v3"),
                "warning": "Static endpoint-level pools remain quarantined despite stricter admission.",
            },
            "v4_evidence_packet": {
                "role": "active_mainline",
                "starts_from_empty_registry": True,
                "legacy_data_allowed": False,
                "formal_experiment_allowed_before_real_packets": False,
            },
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    method_path = OUTPUT_ROOT / "method_registry.json"
    method_path.write_text(json.dumps(method_registry, indent=2, sort_keys=True) + "\n")

    protocol = {
        "protocol": "admet-evidence-packet-data-axis-v4.5",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "implementation_ready_data_acquisition_pending",
        "reference_implementation": {
            "paper": "Closed-loop Auto Research for Molecular Property Prediction: Discovering and Certifying Generalizable Improvements",
            "repository": "https://github.com/cxcscmu/Auto-Research-Molprop",
            "audited_commit": "91bf92a6f0f8f8bab5823a7018349a2e0db2a92a",
        },
        "acquisition": {
            "target": "2-3 tasks frozen in a hash-bound acquisition plan from prior all-task discovery scores",
            "mutation_unit": "one FDA/literature Evidence Packet add/remove/replace per trial",
            "source_priority": [
                "FDA primary document",
                "independent peer-reviewed assay",
            ],
            "maximum_packet_rows": 5000,
            "minimum_packet_rows": 1,
            "existing_external_data": "quarantined_and_forbidden",
        },
        "packet_admission": {
            "required": [
                "stable citation",
                "verified independence",
                "direct endpoint match",
                "explicit assay semantics",
                "explicit label mapping and unit",
                "unique source record IDs",
                "bound records SHA-256",
                "modeling-ready human review",
            ],
            "hard_firewall": {
                "exact_identity": "remove against train, validation, and label-blind test structures",
                "test_skeleton_overlap": "reject whole packet when >0.05",
                "test_tanimoto": "remove row when >=0.90",
                "test_labels_loaded": False,
            },
            "not_hard_admission_gates": [
                "minimum 30/100 identities",
                "target-distribution similarity",
                "train chemical-space coverage",
                "coreset relevance",
            ],
        },
        "training": {
            "primary": "paper_equal direct concatenation, no coreset or ratio sampling",
            "control": "native_priority: native weight 1.0, reviewed packet weight <1.0",
            "external_data_used_for": "training only",
        },
        "closed_loop": {
            "development_seed": 42,
            "development_split": "official Auto-Research-Molprop random scaffold greedy fill",
            "frozen_model": "LightGBM with Morgan ECFP4; only packet registry and sample weights vary",
            "reward": "strict increase in mean normalized improvement over all 22 tasks",
            "routing": "candidate task must be in the immutable acquisition plan; reward scores are never reused to rank the zero-valued baseline",
            "reward_verification": "recompute each metric from private validation labels and hash-bound predictions",
            "snapshot": "full registry, manifests, records, evaluation and predictions",
            "rollback": "restore incumbent registry, manifests and records on every rejected trial",
            "post_discovery_confirmation": "paired independent seeds before ablation claim",
        },
        "registry_path": str(registry.resolve()),
        "registry_sha256": _sha256(registry),
        "method_registry_path": str(method_path.resolve()),
        "method_registry_sha256": _sha256(method_path),
        "implementation_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in implementation
        },
        "formal_experiment_allowed": False,
        "blocking_condition": "no genuine modeling-ready FDA/literature packets are registered",
        "test_labels_loaded": False,
    }
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    output = OUTPUT_ROOT / "protocol.json"
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": protocol["sha256"]}))


if __name__ == "__main__":
    main()
