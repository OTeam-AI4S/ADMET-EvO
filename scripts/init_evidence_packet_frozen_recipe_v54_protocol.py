#!/usr/bin/env python3
"""Freeze Evidence Packet v5.4 candidates with the 22 task-specific recipes."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from admet_agent.evidence_packet_loop import CANONICAL_TASKS
from admet_agent.evidence_packets import (
    SAME_SOURCE_TEST_SKELETON_LIMIT,
    TEST_TANIMOTO_LIMIT,
    sha256_file,
    validate_evidence_packet,
)

ROOT = Path(__file__).resolve().parents[1]
PACKET_ROOT = ROOT / "data/evidence_packets_v4/packets"
MAPPING = ROOT / "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"

CANDIDATES = {
    "AMES": ["chembl37-kim2021-ames-panel", "chembl37-urea2013-ames-panel"],
    "Bioavailability_Ma": [
        "chembl37-arginase2021-mouse-oral-f",
        "chembl37-nr2b2019-rat-oral-f",
        "chembl37-prmt5-2021-rat-oral-f",
        "chembl37-schisto2021-mouse-oral-f",
    ],
    "CYP2C9_Substrate_CarbonMangels": ["fda-cyp-ddi-current-2c9-moderate-sensitive"],
    "CYP2D6_Substrate_CarbonMangels": [
        "fda-cyp-ddi-current-2d6-moderate-sensitive",
        "fda-cyp-ddi-current-2d6-sensitive",
    ],
    "CYP3A4_Substrate_CarbonMangels": [
        "fda-cyp-ddi-current-3a-moderate-sensitive",
        "fda-cyp-ddi-current-3a-sensitive",
    ],
    "CYP3A4_Veith": ["gonzalez2021-cyp3a4-aid1645841"],
    "Caco2_Wang": [
        "avdeef2005-caco2-peerj1405",
        "bergstrom2003-caco2-peerj1405",
        "corti2006-caco2-peerj1405",
        "hayeshi2008-caco2-peerj1405",
        "lin2011-caco2-peerj1405",
        "marino2005-caco2-peerj1405",
        "peng2014-caco2-peerj1405",
        "skolnik2010-caco2-peerj1405",
        "uchida2009-caco2-peerj1405",
    ],
    "Clearance_Hepatocyte_AZ": ["jrc-ecvam2020-clearance-hepatocyte"],
    "Clearance_Microsome_AZ": ["chembl37-granberg2024-hlm-clint"],
    "DILI": ["fda-dilirank2-2025-new-strict"],
    "Half_Life_Obach": [
        "chembl37-maribavir-2008-human-terminal-half-life",
        "chembl37-nemonoxacin-2010-human-terminal-half-life",
        "chembl37-s013420-2009-human-terminal-half-life",
    ],
    "Lipophilicity_AstraZeneca": ["chembl37-lombardo2002-elogd74"],
    "PPBR_AZ": [
        "jrc-ecvam2020-ppbr",
        "sutherland2023-ppbr-drugcentral",
        "sutherland2023-ppbr-pharmapendium",
        "sutherland2023-ppbr-smit",
    ],
    "Pgp_Broccatelli": ["fda-current-clinical-pgp-inhibitors"],
    "Solubility_AqSolDB": ["chembl37-granberg2024-kinetic-logs"],
    "hERG": ["dorn2005-herg-aid376"],
}

REJECTED = {
    "BBB_Martins": "secondary compilation identifiers lack molecule-level original assay locators",
    "CYP2C9_Veith": "candidate source exceeds the preregistered 5% hidden-test skeleton coverage ceiling",
    "CYP2D6_Veith": "candidate source exceeds the preregistered 5% hidden-test skeleton coverage ceiling",
    "HIA_Hou": "candidate values are computational/secondary estimates rather than direct human assay evidence",
    "LD50_Zhu": "same-source lineage and excessive benchmark/test structural overlap",
    "VDss_Lombardo": "all candidate molecules overlap the native development/test identity union",
}

DEEP_TASKS = {"DILI", "Half_Life_Obach", "Lipophilicity_AstraZeneca", "hERG"}


def _source_sets(task: str, packet_ids: list[str]) -> list[dict]:
    sets = [{"source_set_id": "all", "packet_ids": packet_ids}]
    if len(packet_ids) > 1:
        sets.extend(
            {"source_set_id": f"only__{packet_id}", "packet_ids": [packet_id]}
            for packet_id in packet_ids
        )
    if task == "PPBR_AZ":
        sets.extend(
            [
                {
                    "source_set_id": "jrc_plus_smit",
                    "packet_ids": ["jrc-ecvam2020-ppbr", "sutherland2023-ppbr-smit"],
                },
                {
                    "source_set_id": "drugcentral_plus_smit",
                    "packet_ids": [
                        "sutherland2023-ppbr-drugcentral",
                        "sutherland2023-ppbr-smit",
                    ],
                },
            ]
        )
    return sets


def _pilot_rows(task: str) -> list[dict]:
    rows = []
    pattern = ROOT / "reports/evidence_packet_data_axis_v5/*development.json"
    for value in glob.glob(str(pattern)):
        try:
            payload = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("task") != task:
            continue
        for summary in payload.get("configuration_summaries", []):
            rows.append(
                {
                    "packet_ids": sorted(payload.get("packet_ids", [])),
                    "weight": float(summary["weight"]),
                    "quantity_fraction": float(summary["quantity_fraction"]),
                    "pilot_mean": float(summary["mean_normalized_improvement"]),
                    "pilot_positive_fraction": float(summary["positive_seed_fraction"]),
                }
            )
    return rows


def _configurations(task: str, source_sets: list[dict]) -> list[dict]:
    pilot = _pilot_rows(task)
    configurations = []
    for source_set in source_sets:
        exact = [
            row
            for row in pilot
            if row["packet_ids"] == sorted(source_set["packet_ids"])
        ]
        exact.sort(
            key=lambda row: (row["pilot_positive_fraction"], row["pilot_mean"]),
            reverse=True,
        )
        if exact:
            candidates = exact[:3]
        else:
            candidates = [
                {"weight": 0.25, "quantity_fraction": 0.50},
                {"weight": 0.50, "quantity_fraction": 0.50},
            ]
        if task in DEEP_TASKS and source_set["source_set_id"] != "all":
            candidates = candidates[:1]
        for candidate in candidates:
            raw_id = (
                f"{source_set['source_set_id']}|w={candidate['weight']:.3g}|"
                f"q={candidate['quantity_fraction']:.3g}"
            )
            configurations.append(
                {
                    "config_id": "cfg_" + hashlib.sha256(raw_id.encode()).hexdigest()[:12],
                    "source_set_id": source_set["source_set_id"],
                    "packet_ids": source_set["packet_ids"],
                    "weight": float(candidate["weight"]),
                    "quantity_fraction": float(candidate["quantity_fraction"]),
                    "shortlisted_from_v5_controller": bool(exact),
                }
            )
    unique = {item["config_id"]: item for item in configurations}
    return sorted(unique.values(), key=lambda item: item["config_id"])


def build_protocol() -> dict:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    if set(mapping) != set(CANONICAL_TASKS):
        raise RuntimeError("frozen model mapping does not cover all 22 tasks")
    tasks = {}
    for task in CANONICAL_TASKS:
        if task in REJECTED:
            tasks[task] = {
                "status": "source_rejected_before_training",
                "reason": REJECTED[task],
                "source_sets": [],
                "candidate_configurations": [],
            }
            continue
        packet_ids = CANDIDATES[task]
        packets = []
        for packet_id in packet_ids:
            manifest = PACKET_ROOT / packet_id / "manifest.json"
            packet = validate_evidence_packet(manifest)
            if packet.task != task:
                raise RuntimeError(f"{packet_id} is bound to {packet.task}, not {task}")
            review = packet.manifest["review"]
            audit_value = review.get("promotion_audit_path")
            if audit_value:
                audit = ROOT / audit_value
            else:
                audit = manifest.parent / "candidate_firewall_audit.json"
            if not audit.is_file():
                raise RuntimeError(f"{packet_id} lacks a standalone promotion audit")
            audit_payload = json.loads(audit.read_text(encoding="utf-8"))
            inner = audit_payload.get("audit", audit_payload)
            firewall = inner.get("structure_firewall", {})
            if firewall.get("tanimoto_rejection_threshold") != TEST_TANIMOTO_LIMIT:
                raise RuntimeError(f"{packet_id} was not audited at Tanimoto 0.90")
            if not (
                inner.get("test_labels_loaded") is False
                or audit_payload.get("test_labels_loaded") is False
            ):
                raise RuntimeError(f"{packet_id} audit does not prove label blindness")
            packets.append(
                {
                    "packet_id": packet_id,
                    "manifest_path": str(manifest.relative_to(ROOT)),
                    "manifest_sha256": sha256_file(manifest),
                    "records_sha256": sha256_file(packet.records_path),
                    "audit_path": str(audit.relative_to(ROOT)),
                    "audit_sha256": sha256_file(audit),
                    "source_kind": packet.source_kind,
                    "native_priority_external_weight": packet.native_priority_weight,
                }
            )
        source_sets = _source_sets(task, packet_ids)
        tasks[task] = {
            "status": "candidate_ready",
            "packets": packets,
            "source_sets": source_sets,
            "candidate_configurations": _configurations(task, source_sets),
            "frozen_recipe": mapping[task],
        }
    return {
        "protocol": "admet-evidence-packet-v5.4-frozen-recipe-data-axis",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model_axis": {
            "mapping_path": str(MAPPING.relative_to(ROOT)),
            "mapping_sha256": sha256_file(MAPPING),
            "task_specific_frozen_recipe": True,
            "same_recipe_in_native_and_augmented_arms": True,
            "v5_controller_used_only_to_shorten_candidate_grid": True,
        },
        "data_axis": {
            "development_seeds": [271, 307, 347, 389, 431],
            "confirmation_seeds": [467, 503, 547, 593],
            "certification_seeds": [631, 677, 719, 761],
            "maximum_external_to_native_mass_ratio": 0.10,
            "tanimoto_rejection_threshold": TEST_TANIMOTO_LIMIT,
            "same_source_test_skeleton_coverage_limit": SAME_SOURCE_TEST_SKELETON_LIMIT,
            "development_minimum_positive_seed_fraction": 0.80,
            "confirmation_minimum_mean_normalized_improvement": 0.005,
            "confirmation_minimum_positive_seed_fraction": 0.75,
            "certification_minimum_mean_normalized_improvement": 0.0,
            "certification_minimum_positive_seed_fraction": 0.75,
        },
        "safety": {
            "test_structures_used_only_for_label_blind_decontamination": True,
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
            "old_tanimoto_0_80_flow_used": False,
            "top_seed_selection_allowed": False,
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_packet_data_axis_v54/protocol.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_protocol()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "tasks": len(payload["tasks"]),
                "candidate_tasks": sum(
                    item["status"] == "candidate_ready" for item in payload["tasks"].values()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
