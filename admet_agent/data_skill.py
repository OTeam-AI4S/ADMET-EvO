"""Runtime bridge between ADMET-EvO and the packaged Data Collector skill.

The skill is intentionally a gated evidence workflow.  Discovery can create a
source register, but it cannot download, merge, or train on external records.
Only an audited bundle with an explicit release state may cross that boundary.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from .endpoint_contracts import EndpointContract, endpoint_contract
from .external_data_scout import ExternalDataScout

SKILL_ID = "admet-data-collector"
SKILL_VERSION = "1.0.0"
OPERATING_MODES = ("discover", "collect", "audit", "advise", "update")
SI_CHOICES = ("include", "exclude", "main_text_only")

_RESOURCE_PATHS = (
    "SKILL.md",
    "agents/openai.yaml",
    "references/endpoint-ontology.md",
    "references/source-routing.md",
    "references/evidence-validation.md",
    "references/hardness-modeling.md",
    "references/output-contracts.md",
    "references/design-provenance.md",
    "scripts/init_collection_bundle.py",
    "scripts/audit_admet_records.py",
)


def _resource_manifest() -> tuple[list[dict[str, str]], str]:
    root = resources.files("skills.dataskill")
    rows = []
    digest = hashlib.sha256()
    for relative in _RESOURCE_PATHS:
        content = root.joinpath(relative).read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        rows.append({"path": relative, "sha256": sha256})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return rows, digest.hexdigest()


def skill_descriptor() -> dict[str, Any]:
    files, bundle_sha256 = _resource_manifest()
    return {
        "name": SKILL_ID,
        "version": SKILL_VERSION,
        "family": "external_data_stewardship",
        "description": (
            "Endpoint-contract-first discovery, collection, evidence audit, "
            "contamination control, HardnessCard advice and versioned updates."
        ),
        "modes": list(OPERATING_MODES),
        "automated_modes": ["discover", "audit"],
        "scaffolded_human_or_connector_modes": ["collect", "advise", "update"],
        "available": True,
        "requires_endpoint_contract": True,
        "automatic_download_allowed": False,
        "direct_training_requires_release_state": "modeling_ready_discovery",
        "test_labels_allowed": False,
        "resource_bundle_sha256": bundle_sha256,
        "resources": files,
    }


def _yaml_scalar(value: Any) -> str:
    if value is None or value == "":
        value = "unknown"
    return json.dumps(value, ensure_ascii=False)


def _endpoint_contract_yaml(contract: EndpointContract) -> str:
    label_type = "binary" if contract.task_type == "classification" else "continuous"
    return f"""endpoint_id: {_yaml_scalar(contract.endpoint_id)}
property: {_yaml_scalar(contract.property_name)}
admet_domain: {_yaml_scalar(contract.category)}
task_type: {_yaml_scalar(contract.task_type)}
biological_entity: "unknown"
species: {_yaml_scalar(contract.species)}
matrix_or_tissue: "unknown"
assay_system: {_yaml_scalar(contract.assay_system)}
readout: "unknown"
measurement:
  original_unit: {_yaml_scalar(contract.tdc_native_unit)}
  canonical_unit: {_yaml_scalar(contract.tdc_native_unit)}
  scale: "TDC-native; reconstruct before external merging"
  direction: {_yaml_scalar(contract.metric_direction)}
label_definition:
  type: {_yaml_scalar(label_type)}
  description: {_yaml_scalar(contract.label_semantics)}
  threshold: "unknown"
censoring_allowed: "unknown"
intended_role: "external evidence for discovery; no direct merge before audit"
split_contract:
  discovery: {_yaml_scalar(contract.discovery_split)}
  certification: "evaluator_owned_holdout"
  future_audit: "new_independent_evidence"
explicit_exclusions:
  - "predicted-only values as experimental labels"
  - "endpoint-, assay-, species-, unit-, or direction-incompatible records"
  - "held-out exact identities and predeclared near analogues"
"""


class ADMETDataCollectorSkill:
    """Execute the bounded, label-blind parts of the repository data skill."""

    def __init__(self, scout: ExternalDataScout | None = None):
        self.scout = scout or ExternalDataScout()

    def describe(self) -> dict[str, Any]:
        return skill_descriptor()

    def discover(self, task_name: str) -> dict[str, Any]:
        result = self.scout.discover(task_name)
        return {
            "skill": SKILL_ID,
            "skill_version": SKILL_VERSION,
            "mode": "discover",
            "status": (
                "candidate_register_ready"
                if result["candidate_count"]
                else "no_registered_candidate_sources"
            ),
            "discovery": result,
            "required_next_gate": "initialize_and_audit_collection_bundle",
            "automatic_download_allowed": False,
            "test_labels_loaded": False,
        }

    def route(self, task_name: str, research_plan: dict[str, Any]) -> dict[str, Any]:
        selected_id = research_plan.get("selected_action_id")
        selected = next(
            (
                action
                for action in research_plan.get("actions", [])
                if action.get("action_id") == selected_id
            ),
            None,
        )
        if not selected or selected.get("axis") != "data":
            return {
                "skill": SKILL_ID,
                "skill_version": SKILL_VERSION,
                "task": endpoint_contract(task_name).tdc_name,
                "status": "not_activated",
                "selected_action_id": selected_id,
                "test_labels_loaded": False,
            }
        if selected_id == "data-external-quarantine":
            packet = self.discover(task_name)
            packet.update(
                {
                    "task": endpoint_contract(task_name).tdc_name,
                    "selected_action_id": selected_id,
                    "activation_reason": "selected scarcity-driven external-data action",
                }
            )
            return packet
        return {
            "skill": SKILL_ID,
            "skill_version": SKILL_VERSION,
            "task": endpoint_contract(task_name).tdc_name,
            "mode": "audit",
            "status": "delegated_to_internal_split_safe_curator",
            "selected_action_id": selected_id,
            "data_policy": selected.get("payload", {}).get("data_policy"),
            "external_records_admitted": False,
            "test_labels_loaded": False,
        }

    def initialize_bundle(
        self,
        task_name: str,
        output_dir: str | Path,
        *,
        si_choice: str,
        allow_existing: bool = False,
    ) -> dict[str, Any]:
        if si_choice not in SI_CHOICES:
            raise ValueError(f"si_choice must be one of {SI_CHOICES}")
        contract = endpoint_contract(task_name)
        output = Path(output_dir).resolve()
        contract_path = output / "contracts/endpoint_contract.yaml"
        contract_existed = contract_path.exists() and contract_path.stat().st_size > 0
        command = [
            sys.executable,
            "-m",
            "skills.dataskill.scripts.init_collection_bundle",
            "--out",
            str(output),
            "--endpoint-id",
            contract.endpoint_id,
        ]
        if allow_existing:
            command.append("--allow-existing")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Data-skill bundle initialization failed: {message}")
        if not contract_existed:
            contract_path.write_text(_endpoint_contract_yaml(contract), encoding="utf-8")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "skill": SKILL_ID,
                "skill_version": SKILL_VERSION,
                "tdc_task": contract.tdc_name,
                "si_choice": si_choice,
                "endpoint_contract_source": "admet_agent.endpoint_contracts",
                "test_labels_loaded": False,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "initialized",
            "skill": SKILL_ID,
            "task": contract.tdc_name,
            "endpoint_id": contract.endpoint_id,
            "bundle": str(output),
            "si_choice": si_choice,
            "release_state": manifest["release_state"],
            "test_labels_loaded": False,
        }

    def modeling_admission_status(self, bundle: str | Path) -> dict[str, Any]:
        bundle_path = Path(bundle).resolve()
        reasons = []
        manifest_path = bundle_path / "manifest.json"
        audit_path = bundle_path / "reports/automated_audit.json"
        if not manifest_path.exists():
            reasons.append("manifest_missing")
            manifest: dict[str, Any] = {}
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not audit_path.exists():
            reasons.append("automated_audit_missing")
            audit: dict[str, Any] = {}
        else:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        summary = audit.get("summary", {})
        if int(summary.get("critical_issues", 0)):
            reasons.append("critical_audit_issues_present")
        if int(summary.get("direct_records", 0)) <= 0:
            reasons.append("no_direct_records")
        if summary.get("suggested_release_state") != "modeling_ready_discovery":
            reasons.append("automated_audit_does_not_recommend_modeling_ready")
        if manifest.get("release_state") != "modeling_ready_discovery":
            reasons.append("manifest_not_promoted_by_scientific_review")
        review = manifest.get("review", {})
        if not review.get("scientific_reviewer") or not review.get("reviewed_at"):
            reasons.append("scientific_review_signoff_missing")
        contamination = audit.get("contamination", {})
        if contamination.get("status") != "completed":
            reasons.append("heldout_contamination_audit_incomplete")
        return {
            "skill": SKILL_ID,
            "skill_version": SKILL_VERSION,
            "bundle": str(bundle_path),
            "admitted_for_direct_training": not reasons,
            "reasons": sorted(set(reasons)),
            "release_state": manifest.get("release_state"),
            "suggested_release_state": summary.get("suggested_release_state"),
            "test_labels_loaded": False,
        }

    def audit_bundle(
        self,
        bundle: str | Path,
        *,
        report_dir: str | Path | None = None,
        heldout: str | Path | None = None,
        source_overlap_threshold: float = 0.05,
        near_analogue_threshold: float = 0.80,
        max_analog_comparisons: int = 2_000_000,
        strict: bool = False,
    ) -> dict[str, Any]:
        bundle_path = Path(bundle).resolve()
        output = Path(report_dir).resolve() if report_dir else bundle_path / "reports"
        command = [
            sys.executable,
            "-m",
            "skills.dataskill.scripts.audit_admet_records",
            "--bundle",
            str(bundle_path),
            "--report-dir",
            str(output),
            "--source-overlap-threshold",
            str(source_overlap_threshold),
            "--near-analogue-threshold",
            str(near_analogue_threshold),
            "--max-analog-comparisons",
            str(max_analog_comparisons),
        ]
        if heldout is not None:
            command.extend(("--heldout", str(Path(heldout).resolve())))
        if strict:
            command.append("--strict")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode not in {0, 2}:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Data-skill audit failed: {message}")
        report_path = output / "automated_audit.json"
        result = json.loads(report_path.read_text(encoding="utf-8"))
        result["skill"] = SKILL_ID
        result["skill_version"] = SKILL_VERSION
        result["strict_gate_passed"] = completed.returncode == 0
        result["test_labels_loaded"] = False
        manifest_path = bundle_path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["audit"] = {
                "status": "completed",
                "critical_issues": result["summary"]["critical_issues"],
                "warnings": result["summary"]["warnings"],
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            }
            manifest["suggested_release_state"] = result["summary"][
                "suggested_release_state"
            ]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        result["modeling_admission"] = self.modeling_admission_status(bundle_path)
        return result


__all__ = [
    "OPERATING_MODES",
    "SI_CHOICES",
    "SKILL_ID",
    "SKILL_VERSION",
    "ADMETDataCollectorSkill",
    "skill_descriptor",
]
