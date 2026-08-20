"""Fail-closed release audit for third-party weights used by ADMET-EvO v2.7.

Network observations are pinned as dated evidence rather than repeated during
tests.  This keeps the reviewer bundle deterministic while separating public
downloadability from the distinct question of checkpoint redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MAPPING = Path("runs/v27-maplight-gnn-freeze/full-22-task.mapping.json")
PREDICTOR = Path("scripts/predict_maplight_gnn.py")
GIN_EQUIVALENCE = Path("reports/gin_checkpoint_equivalence_v27.json")
OBSERVED_AT = "2026-08-17"
GIN_SHA256 = "6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9"
GIN_METADATA_SHA256 = "75ea305d643d800b8b272819a78b842d5aca1c4ad55d47207321ab9b81d44d02"
UNIMOL2_SHA256 = "5b9241630f1cf0b173fb06d1e76096e5daf5f91c3e51b066ba69524dafe60e35"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _recipe_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True).lower()


def build_audit(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    mapping_path = root / MAPPING
    predictor_path = root / PREDICTOR
    equivalence_path = root / GIN_EQUIVALENCE
    mapping = _load(mapping_path)
    equivalence = _load(equivalence_path)
    if len(mapping) != 22:
        raise ValueError("Frozen mapping must contain exactly 22 tasks")

    maplight_tasks = sorted(
        task for task, recipe in mapping.items() if recipe.get("recipe") == "maplight_gnn"
    )
    unimol2_tasks = sorted(
        task for task, recipe in mapping.items() if "unimol2" in _recipe_text(recipe)
    )
    minimol_tasks = sorted(
        task for task, recipe in mapping.items() if "minimol" in _recipe_text(recipe)
    )
    if maplight_tasks != ["Bioavailability_Ma", "CYP2C9_Substrate_CarbonMangels"]:
        raise ValueError(f"Unexpected frozen MapLight task set: {maplight_tasks}")
    if unimol2_tasks != ["DILI", "Half_Life_Obach", "Lipophilicity_AstraZeneca", "hERG"]:
        raise ValueError(f"Unexpected frozen Uni-Mol2 task set: {unimol2_tasks}")
    if minimol_tasks:
        raise ValueError(f"MiniMol unexpectedly entered the frozen mapping: {minimol_tasks}")

    predictor = predictor_path.read_text(encoding="utf-8")
    pinned = dict(
        re.findall(r'^(CHECKPOINT_SHA256|METADATA_SHA256) = "([0-9a-f]{64})"$', predictor, re.MULTILINE)
    )
    if pinned != {"CHECKPOINT_SHA256": GIN_SHA256, "METADATA_SHA256": GIN_METADATA_SHA256}:
        raise ValueError("MapLight predictor no longer pins the audited GIN artifacts")
    if (
        equivalence["verified"] is not True
        or equivalence["comparison"]["all_parameter_tensors_bitwise_equal"] is not True
        or equivalence["comparison"]["maximum_absolute_parameter_difference"] != 0.0
        or equivalence["comparison"]["parameter_tensor_count"] != 57
        or equivalence["official_dgllife"]["sha256"]
        != "a43274c29be5a8cf38f990794d0dc3fcc0572dfa5524240e77ed1c6aa1b54fb8"
        or equivalence["molfeat"]["sha256"] != GIN_SHA256
    ):
        raise ValueError("GIN checkpoint equivalence evidence is incomplete or changed")

    dependencies = {
        "unimol2_84m": {
            "essential": True,
            "used_by_tasks": unimol2_tasks,
            "runtime": {"package": "unimol-tools", "version": "0.1.5"},
            "source_repository": "https://github.com/deepmodeling/unimol_tools",
            "source_license": {
                "spdx": "MIT",
                "verified": True,
                "url": "https://github.com/deepmodeling/unimol_tools/blob/main/LICENSE",
            },
            "checkpoint": {
                "repository": "https://huggingface.co/dptech/Uni-Mol2",
                "path": "modelzoo/84M/checkpoint.pt",
                "direct_url": "https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt?download=true",
                "repository_commit": "bf37d253c0f5ac61f468137eb00059978a104ff0",
                "sha256": UNIMOL2_SHA256,
                "bytes": 336_632_021,
                "private": False,
                "gated": False,
                "registration_required": False,
                "anonymous_access_verified": True,
                "http_observation": "anonymous 302 to public CDN followed by HTTP 200",
                "license": "MIT",
                "checkpoint_specific_license_explicit": True,
            },
            "redistribution_policy": "external_link_only_in_project_bundle",
            "included_in_reviewer_bundle": False,
        },
        "gin_supervised_masking": {
            "essential": True,
            "used_by_tasks": maplight_tasks,
            "runtime": {
                "packages": {
                    "molfeat": "0.11.0",
                    "dgllife": "0.3.2",
                    "dgl": "2.1.0",
                }
            },
            "source_repositories": [
                {
                    "name": "MapLight audited source",
                    "url": "https://github.com/receptor-ai/tdc-admet-bench",
                    "license": "MIT",
                    "license_url": "https://github.com/receptor-ai/tdc-admet-bench/blob/main/LICENSE",
                    "verified": True,
                },
                {
                    "name": "Molfeat",
                    "url": "https://github.com/datamol-io/molfeat",
                    "license": "Apache-2.0",
                    "license_url": "https://github.com/datamol-io/molfeat/blob/main/LICENSE",
                    "verified": True,
                },
                {
                    "name": "DGL-LifeSci",
                    "url": "https://github.com/awslabs/dgl-lifesci",
                    "license": "Apache-2.0",
                    "license_url": "https://github.com/awslabs/dgl-lifesci/blob/master/LICENSE",
                    "verified": True,
                },
            ],
            "checkpoint": {
                "metadata_url": "https://fs.molfeat.datamol.io/artifacts/dgllife/gin_supervised_masking/0/metadata.json",
                "direct_url": "https://fs.molfeat.datamol.io/artifacts/dgllife/gin_supervised_masking/0/model.save",
                "official_dgllife_url": equivalence["official_dgllife"]["url"],
                "official_dgllife_sha256": equivalence["official_dgllife"]["sha256"],
                "sha256": GIN_SHA256,
                "metadata_sha256": GIN_METADATA_SHA256,
                "bytes": 7_467_310,
                "registration_required": False,
                "anonymous_access_verified": True,
                "http_observation": "anonymous HTTP GET returned 200 for metadata and model",
                "metadata_license_field": None,
                "checkpoint_specific_license_explicit": False,
                "license_boundary": (
                    "The Molfeat tensors exactly match the DGL-LifeSci official artifact, but "
                    "source SPDX metadata still does not prove a checkpoint redistribution license."
                ),
                "official_parameter_equivalence": {
                    "verified": equivalence["verified"],
                    "parameter_tensor_count": equivalence["comparison"]["parameter_tensor_count"],
                    "parameter_value_count": equivalence["comparison"]["parameter_value_count"],
                    "all_parameter_tensors_bitwise_equal": equivalence["comparison"][
                        "all_parameter_tensors_bitwise_equal"
                    ],
                    "maximum_absolute_parameter_difference": equivalence["comparison"][
                        "maximum_absolute_parameter_difference"
                    ],
                },
            },
            "redistribution_policy": "external_link_only_verified_against_official_dgllife_parameters",
            "included_in_reviewer_bundle": False,
        },
    }

    access_verified = all(
        item["checkpoint"]["anonymous_access_verified"]
        and item["checkpoint"]["registration_required"] is False
        for item in dependencies.values()
    )
    code_licenses_verified = all(
        item.get("source_license", {}).get("verified", True)
        and all(source["verified"] for source in item.get("source_repositories", []))
        for item in dependencies.values()
    )
    checkpoint_licenses_verified = all(
        item["checkpoint"]["checkpoint_specific_license_explicit"]
        for item in dependencies.values()
    )
    return {
        "schema_version": "third-party-release-audit-v27-1",
        "observed_at": OBSERVED_AT,
        "scope": {
            "frozen_task_count": len(mapping),
            "essential_dependency_count": len(dependencies),
            "final_mapping_minimol_used": False,
            "rejected_experiments_are_not_release_dependencies": True,
        },
        "dependencies": dependencies,
        "gates": {
            "registration_free_checkpoint_access_verified": access_verified,
            "source_code_licenses_verified": code_licenses_verified,
            "checkpoint_specific_licenses_verified": checkpoint_licenses_verified,
            "third_party_checkpoints_in_bundle": False,
            "safe_to_redistribute_third_party_checkpoints": checkpoint_licenses_verified,
            "bundle_safe_for_external_deposit": True,
            "ready_for_external_deposit": True,
            "remaining_external_gate": (
                "deposit the prepared link-only bundle and mint a DOI; do not add third-party "
                "checkpoint files unless their artifact licenses are made explicit"
            ),
        },
        "policy": {
            "bundle_action": "exclude all third-party checkpoints and publish immutable URLs plus hashes",
            "link_only_release_justification": (
                "The bundle contains no third-party checkpoint bytes; the public DGL and Molfeat "
                "URLs plus exact parameter-equivalence evidence are sufficient to reacquire the model."
            ),
            "fail_closed_rule": (
                "A package or source-repository license is not treated as a checkpoint license "
                "unless the model repository or artifact metadata explicitly says so."
            ),
            "official_test_loaded": False,
            "selection_or_freeze_changed": False,
        },
        "sources": {
            "frozen_mapping": {"path": MAPPING.as_posix(), "sha256": _sha256(mapping_path)},
            "maplight_predictor": {"path": PREDICTOR.as_posix(), "sha256": _sha256(predictor_path)},
            "gin_checkpoint_equivalence": {
                "path": GIN_EQUIVALENCE.as_posix(),
                "sha256": _sha256(equivalence_path),
            },
        },
    }


def markdown(report: dict[str, Any]) -> str:
    rows = []
    for name, item in report["dependencies"].items():
        checkpoint = item["checkpoint"]
        license_value = checkpoint.get("license") or "not explicit"
        rows.append(
            f"| {name} | {', '.join(item['used_by_tasks'])} | yes | {license_value} | "
            f"{checkpoint['sha256']} | {item['redistribution_policy']} |"
        )
    gates = report["gates"]
    return "\n".join(
        [
            "# ADMET-EvO v2.7 third-party release audit",
            "",
            f"Observation date: `{report['observed_at']}`.",
            "",
            "| Dependency | Frozen tasks | Anonymous access | Checkpoint license | SHA-256 | Bundle policy |",
            "|---|---|---:|---|---|---|",
            *rows,
            "",
            "## Release gates",
            "",
            f"- Registration-free checkpoint access verified: **{str(gates['registration_free_checkpoint_access_verified']).lower()}**",
            f"- Source-code licenses verified: **{str(gates['source_code_licenses_verified']).lower()}**",
            f"- Checkpoint-specific licenses verified: **{str(gates['checkpoint_specific_licenses_verified']).lower()}**",
            f"- Ready for external deposit: **{str(gates['ready_for_external_deposit']).lower()}**",
            "",
            "The Uni-Mol2 model repository explicitly applies MIT metadata to the public, ungated repository. "
            "The Molfeat GIN artifact is publicly downloadable and content-addressed. Its 57 parameter tensors "
            "are bitwise identical to DGL-LifeSci's official checkpoint, but its metadata has no license field. "
            "Apache-2.0 on source code is therefore not promoted into an artifact redistribution claim. The "
            "reviewer bundle links to both checkpoints and redistributes neither, so the bundle itself is ready "
            "for external deposit while checkpoint redistribution remains disallowed.",
            "",
            f"Remaining gate: {gates['remaining_external_gate']}.",
            "",
            "This audit is release-only: it did not load official test data and did not change model selection or the frozen scorecard.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/third_party_release_audit_v27"))
    args = parser.parse_args()
    report = build_audit(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
