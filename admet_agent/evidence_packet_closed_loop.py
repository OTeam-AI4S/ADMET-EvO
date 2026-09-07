"""Tamper-evident Evidence Packet evaluation artifacts and transactional snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evidence_packet_loop import CANONICAL_TASKS, TrialDecision, evaluate_packet_trial
from .evidence_packet_splits import scaffold_train_validation_split
from .evidence_packets import sha256_file, validate_evidence_packet
from .selection import score
from .tasks import resolve_task

ARTIFACT_PROTOCOL = "admet-evidence-packet-all22-evaluation-v4.5"
SNAPSHOT_PROTOCOL = "admet-evidence-packet-snapshot-v4.5"
STATE_PROTOCOL = "admet-evidence-packet-closed-loop-state-v4.5"
_SHA256 = frozenset("0123456789abcdef")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_CONFIG = {
    "model": "lightgbm",
    "hyperparameters": {},
    "imbalance": "none",
}
EXPECTED_FEATURE_CONFIG = ["morgan"]
REQUIRED_FROZEN_IMPLEMENTATION = (
    "admet_agent/evidence_packet_closed_loop.py",
    "admet_agent/evidence_packet_splits.py",
    "admet_agent/evidence_packets.py",
    "admet_agent/external_structure_firewall.py",
    "admet_agent/features.py",
    "admet_agent/selection.py",
    "admet_agent/schemas.py",
    "admet_agent/tasks.py",
    "scripts/run_evidence_packet_closed_loop_v4.py",
)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256)


def _normalized_improvement(metric: str, baseline: float, candidate: float) -> float:
    denominator = max(abs(float(baseline)), 1e-12)
    return float(
        (baseline - candidate) / denominator
        if metric == "mae"
        else (candidate - baseline) / denominator
    )


def _evaluation_contract_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "seed": payload["seed"],
            "model_config": payload["model_config"],
            "feature_config": payload["feature_config"],
            "weight_mode": payload["weight_mode"],
            "acquisition_plan_sha256": payload["acquisition_plan_sha256"],
            "acquisition_target_tasks": payload["acquisition_target_tasks"],
            "frozen_implementation": payload["frozen_implementation_before"],
        }
    )


def finalize_evaluation_artifact(
    payload: Mapping[str, Any], output: str | Path
) -> dict[str, Any]:
    """Write an immutable self-hashed evaluator artifact."""

    value = dict(payload)
    value["protocol"] = ARTIFACT_PROTOCOL
    value.pop("artifact_sha256", None)
    value["artifact_sha256"] = canonical_sha256(value)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return value


def load_verified_evaluation_artifact(
    path: str | Path,
    *,
    expected_baseline_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify all-22 scores from bound raw metrics and prediction arrays."""

    artifact_path = Path(path).resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != ARTIFACT_PROTOCOL:
        raise ValueError("evaluation artifact protocol mismatch")
    if payload.get("status") != "completed":
        raise ValueError("only completed all-22 evaluations may enter the loop")
    stored_hash = payload.get("artifact_sha256")
    core = dict(payload)
    core.pop("artifact_sha256", None)
    if not _valid_sha256(stored_hash) or canonical_sha256(core) != stored_hash:
        raise ValueError("evaluation artifact self hash mismatch")
    if payload.get("test_labels_loaded") is not False:
        raise ValueError("closed-loop discovery must not load test labels")
    if payload.get("legacy_bulk_pool_used") is not False:
        raise ValueError("legacy bulk data are forbidden")

    kind = payload.get("kind")
    if kind not in {"baseline", "challenger"}:
        raise ValueError("evaluation kind must be baseline or challenger")
    baseline_sha = payload.get("baseline_artifact_sha256")
    if kind == "baseline":
        if baseline_sha is not None:
            raise ValueError("baseline evaluation may not reference another baseline")
        if payload.get("packet_ids"):
            raise ValueError("the frozen baseline must contain no external packets")
    else:
        if not _valid_sha256(baseline_sha):
            raise ValueError("challenger is missing a frozen baseline binding")
        if (
            expected_baseline_sha256 is not None
            and baseline_sha != expected_baseline_sha256
        ):
            raise ValueError("challenger baseline binding differs from loop baseline")

    frozen_before = payload.get("frozen_implementation_before")
    frozen_after = payload.get("frozen_implementation_after")
    if not isinstance(frozen_before, dict) or frozen_before != frozen_after:
        raise ValueError(
            "model/feature/evaluator implementation changed during evaluation"
        )
    if not frozen_before or not all(
        _valid_sha256(value) for value in frozen_before.values()
    ):
        raise ValueError("evaluation lacks complete frozen implementation hashes")

    if payload.get("seed") != 42:
        raise ValueError("evaluation seed must be frozen to 42")
    if payload.get("model_config") != EXPECTED_MODEL_CONFIG:
        raise ValueError("evaluation model config differs from frozen data-only model")
    if payload.get("feature_config") != EXPECTED_FEATURE_CONFIG:
        raise ValueError(
            "evaluation feature config differs from frozen data-only features"
        )
    if payload.get("weight_mode") not in {"paper_equal", "native_priority"}:
        raise ValueError("evaluation weight mode is not preregistered")
    if not _valid_sha256(payload.get("acquisition_plan_sha256")):
        raise ValueError("evaluation artifact acquisition-plan hash is invalid")
    target_tasks = payload.get("acquisition_target_tasks")
    if (
        not isinstance(target_tasks, list)
        or len(target_tasks) not in {2, 3}
        or len(set(target_tasks)) != len(target_tasks)
        or not set(target_tasks).issubset(CANONICAL_TASKS)
    ):
        raise ValueError("evaluation artifact acquisition targets are invalid")
    if set(frozen_before) != set(REQUIRED_FROZEN_IMPLEMENTATION):
        raise ValueError("evaluation frozen implementation file set is incomplete")
    for relative, expected_sha256 in frozen_before.items():
        implementation_path = (PROJECT_ROOT / relative).resolve()
        if (
            not implementation_path.is_relative_to(PROJECT_ROOT)
            or not implementation_path.is_file()
            or sha256_file(implementation_path) != expected_sha256
        ):
            raise ValueError(f"frozen implementation hash mismatch: {relative}")
    task_results = payload.get("tasks")
    if not isinstance(task_results, dict) or set(task_results) != set(CANONICAL_TASKS):
        raise ValueError("evaluation artifact must contain exactly all 22 tasks")
    normalized_scores: dict[str, float] = {}
    prediction_root = artifact_path.parent.resolve()
    for task in CANONICAL_TASKS:
        result = task_results[task]
        if result.get("status") != "completed":
            raise ValueError(f"task {task} did not complete")
        metric = str(result.get("metric"))
        if metric not in {"mae", "auroc", "auprc", "spearman"}:
            raise ValueError(f"task {task} has unsupported metric")
        raw_score = float(result["score"])
        baseline_score = float(result["baseline_score"])
        if not math.isfinite(raw_score) or not math.isfinite(baseline_score):
            raise ValueError(f"task {task} contains a non-finite score")
        expected_normalized = (
            0.0
            if kind == "baseline"
            else _normalized_improvement(metric, baseline_score, raw_score)
        )
        observed_normalized = float(result["normalized_improvement"])
        if not math.isclose(
            observed_normalized, expected_normalized, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"task {task} normalized score was not evaluator-derived")
        prediction_relative = Path(str(result["prediction_path"]))
        if prediction_relative.is_absolute():
            raise ValueError("prediction paths must be artifact-relative")
        prediction_path = (prediction_root / prediction_relative).resolve()
        if not prediction_path.is_relative_to(prediction_root):
            raise ValueError("prediction path escapes the artifact directory")
        if not prediction_path.is_file():
            raise ValueError(f"task {task} prediction file is missing")
        if sha256_file(prediction_path) != result.get("prediction_sha256"):
            raise ValueError(f"task {task} prediction hash mismatch")
        predictions = np.load(prediction_path, allow_pickle=False)
        if predictions.ndim != 1 or len(predictions) != int(result["validation_rows"]):
            raise ValueError(f"task {task} prediction shape mismatch")
        if not np.isfinite(predictions).all():
            raise ValueError(f"task {task} predictions are not finite")
        if not _valid_sha256(result.get("validation_target_sha256")):
            raise ValueError(f"task {task} lacks a validation-target binding")
        if not _valid_sha256(result.get("split_sha256")):
            raise ValueError(f"task {task} lacks an official split binding")
        normalized_scores[task] = observed_normalized

    declared_scores = payload.get("normalized_scores")
    if not isinstance(declared_scores, dict) or set(declared_scores) != set(
        CANONICAL_TASKS
    ):
        raise ValueError("declared normalized scores are incomplete")
    for task in CANONICAL_TASKS:
        if not math.isclose(
            float(declared_scores[task]),
            normalized_scores[task],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("declared normalized scores differ from task results")
    expected_mean = sum(normalized_scores.values()) / len(normalized_scores)
    if not math.isclose(
        float(payload["mean_normalized_improvement"]),
        expected_mean,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("aggregate score differs from the all-22 mean")
    if not _valid_sha256(payload.get("registry_sha256")):
        raise ValueError("evaluation is not bound to a registry")
    packet_ids = payload.get("packet_ids")
    if not isinstance(packet_ids, list) or packet_ids != sorted(set(packet_ids)):
        raise ValueError("packet_ids must be a sorted unique list")
    payload["normalized_scores"] = normalized_scores
    return payload


def load_verified_evaluation_artifact_against_data(
    path: str | Path,
    *,
    data_dir: str | Path,
    expected_baseline_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute every task metric from private validation labels and predictions."""

    artifact_path = Path(path).resolve()
    payload = load_verified_evaluation_artifact(
        artifact_path,
        expected_baseline_sha256=expected_baseline_sha256,
    )
    seed = int(payload.get("seed", -1))
    if seed != 42:
        raise ValueError("formal evaluation artifacts must use seed 42")
    root = Path(data_dir).resolve()
    for task_name in CANONICAL_TASKS:
        task = resolve_task(task_name)
        source = root / "admet_group" / task_name.lower() / "train_val.csv"
        raw = pd.read_csv(source, usecols=["Drug", "Y"])
        _, validation, split_audit = scaffold_train_validation_split(raw, seed=seed)
        result = payload["tasks"][task_name]
        if result["split_sha256"] != split_audit["split_sha256"]:
            raise ValueError(f"{task_name}: split hash differs from frozen data")
        y_validation = validation["Y"].to_numpy(
            dtype=int if task.task_type == "classification" else float
        )
        target_sha256 = hashlib.sha256(
            np.ascontiguousarray(y_validation).tobytes()
        ).hexdigest()
        if result["validation_target_sha256"] != target_sha256:
            raise ValueError(f"{task_name}: validation-target hash mismatch")
        prediction_path = artifact_path.parent / result["prediction_path"]
        predictions = np.load(prediction_path, allow_pickle=False)
        observed_score = score(y_validation, predictions, task.metric)
        if not math.isclose(
            float(result["score"]),
            observed_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{task_name}: metric differs from bound predictions")
    return payload


def _registry_packets(registry_path: Path) -> tuple[list[str], dict[str, str]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("active_packets")
    if not isinstance(entries, list):
        raise TypeError("registry active_packets must be a list")
    packet_ids: list[str] = []
    task_by_id: dict[str, str] = {}
    for entry in entries:
        packet_id = str(entry["packet_id"])
        manifest_value = Path(str(entry["manifest_path"]))
        manifest_path = (
            manifest_value
            if manifest_value.is_absolute()
            else PROJECT_ROOT / manifest_value
        ).resolve()
        packet = validate_evidence_packet(manifest_path)
        if packet.packet_id != packet_id or packet.task != entry.get("task"):
            raise ValueError("registry entry differs from its Evidence Packet")
        if entry.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("registry manifest binding mismatch")
        if packet_id in task_by_id:
            raise ValueError("registry contains duplicate packet IDs")
        packet_ids.append(packet_id)
        task_by_id[packet_id] = packet.task
    return sorted(packet_ids), task_by_id


class EvidencePacketClosedLoop:
    """Artifact-only closed loop with immutable, full-data snapshots."""

    def __init__(self, state_path: str | Path, *, data_dir: str | Path):
        self.state_path = Path(state_path).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.snapshot_root = self.state_path.parent / (
            self.state_path.stem + ".snapshots"
        )

    def initialize(
        self,
        *,
        evaluation_artifact: str | Path,
        registry_path: str | Path,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if self.state_path.exists() and not overwrite:
            raise FileExistsError(self.state_path)
        artifact = load_verified_evaluation_artifact_against_data(
            evaluation_artifact,
            data_dir=self.data_dir,
        )
        if artifact["kind"] != "baseline":
            raise ValueError("closed loop must initialize from an empty-data baseline")
        registry = Path(registry_path).resolve()
        packet_ids, _ = _registry_packets(registry)
        if packet_ids or artifact["packet_ids"]:
            raise ValueError("closed-loop baseline registry must be empty")
        self._require_registry_binding(artifact, registry, packet_ids)
        snapshot = self._create_snapshot(registry, Path(evaluation_artifact), artifact)
        state = {
            "protocol": STATE_PROTOCOL,
            "iteration": 0,
            "baseline_evaluation_sha256": artifact["artifact_sha256"],
            "evaluation_contract_sha256": _evaluation_contract_sha256(artifact),
            "weight_mode": artifact["weight_mode"],
            "acquisition_plan_sha256": artifact["acquisition_plan_sha256"],
            "acquisition_target_tasks": artifact["acquisition_target_tasks"],
            "data_dir": str(self.data_dir),
            "baseline": snapshot,
            "incumbent": snapshot,
            "history": [],
            "transactional_staging": True,
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
        }
        self._write_state(state)
        return state

    def load(self) -> dict[str, Any]:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("protocol") != STATE_PROTOCOL:
            raise ValueError("closed-loop state protocol mismatch")
        self._verify_snapshot(state["baseline"])
        if not _valid_sha256(state.get("acquisition_plan_sha256")):
            raise ValueError("closed-loop acquisition-plan hash is invalid")
        target_tasks = state.get("acquisition_target_tasks")
        if (
            not isinstance(target_tasks, list)
            or len(target_tasks) not in {2, 3}
            or len(set(target_tasks)) != len(target_tasks)
            or not set(target_tasks).issubset(CANONICAL_TASKS)
        ):
            raise ValueError("closed-loop acquisition targets are invalid")
        if Path(state.get("data_dir", "")).resolve() != self.data_dir:
            raise ValueError(
                "closed-loop data directory differs from initialized state"
            )
        self._verify_snapshot(state["incumbent"])
        return state

    def validate_candidate_registry(
        self,
        registry_path: str | Path,
        *,
        weak_task_count: int = 3,
    ) -> dict[str, Any]:
        """Fail before fitting unless the staged registry is one valid mutation."""

        state = self.load()
        registry = Path(registry_path).resolve()
        packet_ids, task_by_id = _registry_packets(registry)
        incumbent = state["incumbent"]
        old_bindings = self._registry_bindings(
            Path(str(incumbent["snapshot_path"])) / "registry.json"
        )
        task_by_id.update(
            {packet_id: task for packet_id, (task, _) in old_bindings.items()}
        )
        try:
            self._require_incumbent_packets_unchanged(incumbent, registry)
            decision = evaluate_packet_trial(
                incumbent_packet_ids=incumbent["packet_ids"],
                challenger_packet_ids=packet_ids,
                incumbent_scores=incumbent["scores"],
                challenger_scores=incumbent["scores"],
                packet_task_by_id=task_by_id,
                weak_task_count=weak_task_count,
                acquisition_target_tasks=state["acquisition_target_tasks"],
            )
        except Exception:
            self._restore_snapshot(incumbent, registry)
            raise
        return {
            "packet_ids": packet_ids,
            "mutation": decision.mutation,
            "target_tasks": list(decision.target_tasks),
            "targeted_weak_task": decision.targeted_weak_task,
        }

    def submit(
        self,
        *,
        evaluation_artifact: str | Path,
        registry_path: str | Path,
        weak_task_count: int = 3,
    ) -> TrialDecision:
        state = self.load()
        artifact = load_verified_evaluation_artifact_against_data(
            evaluation_artifact,
            data_dir=self.data_dir,
            expected_baseline_sha256=state["baseline_evaluation_sha256"],
        )
        if artifact["kind"] != "challenger":
            raise ValueError("submit requires a challenger evaluation")
        if _evaluation_contract_sha256(artifact) != state["evaluation_contract_sha256"]:
            self._restore_snapshot(state["incumbent"], Path(registry_path).resolve())
            raise ValueError(
                "challenger changed the frozen model/feature/weight implementation contract"
            )
        registry = Path(registry_path).resolve()
        packet_ids, task_by_id = _registry_packets(registry)
        self._require_registry_binding(artifact, registry, packet_ids)
        incumbent = state["incumbent"]
        try:
            self._require_incumbent_packets_unchanged(incumbent, registry)
        except Exception:
            self._restore_snapshot(incumbent, registry)
            raise
        decision = evaluate_packet_trial(
            incumbent_packet_ids=incumbent["packet_ids"],
            challenger_packet_ids=packet_ids,
            incumbent_scores=incumbent["scores"],
            challenger_scores=artifact["normalized_scores"],
            packet_task_by_id=task_by_id,
            weak_task_count=weak_task_count,
            acquisition_target_tasks=state["acquisition_target_tasks"],
        )
        state["iteration"] += 1
        event = {
            "iteration": state["iteration"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "challenger_artifact_sha256": artifact["artifact_sha256"],
            "challenger_registry_sha256": artifact["registry_sha256"],
            "challenger_packet_ids": packet_ids,
            "decision": decision.to_dict(),
            "test_labels_loaded": False,
        }
        if decision.keep:
            state["incumbent"] = self._create_snapshot(
                registry, Path(evaluation_artifact), artifact
            )
            event["registry_rolled_back"] = False
        else:
            self._restore_snapshot(incumbent, registry)
            event["registry_rolled_back"] = True
        event["kept_snapshot_sha256"] = state["incumbent"]["snapshot_sha256"]
        state["history"].append(event)
        self._write_state(state)
        return decision

    @staticmethod
    def _require_registry_binding(
        artifact: Mapping[str, Any], registry: Path, packet_ids: list[str]
    ) -> None:
        if sha256_file(registry) != artifact["registry_sha256"]:
            raise ValueError("evaluation registry hash differs from submitted registry")
        if packet_ids != artifact["packet_ids"]:
            raise ValueError("evaluation packet IDs differ from submitted registry")

    @staticmethod
    def _registry_bindings(registry: Path) -> dict[str, tuple[str, str]]:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        return {
            str(entry["packet_id"]): (
                str(entry["task"]),
                str(entry["manifest_sha256"]),
            )
            for entry in payload.get("active_packets", [])
        }

    def _require_incumbent_packets_unchanged(
        self, incumbent: Mapping[str, Any], challenger_registry: Path
    ) -> None:
        incumbent_registry = Path(str(incumbent["snapshot_path"])) / "registry.json"
        old = self._registry_bindings(incumbent_registry)
        new = self._registry_bindings(challenger_registry)
        changed = sorted(
            packet_id
            for packet_id in set(old) & set(new)
            if old[packet_id] != new[packet_id]
        )
        if changed:
            raise ValueError(
                "challenger mutated incumbent packet bindings: " + ", ".join(changed)
            )

    @staticmethod
    def _restore_snapshot(snapshot: Mapping[str, Any], registry: Path) -> None:
        root = Path(str(snapshot["snapshot_path"])).resolve()
        source_registry = root / "registry.json"
        payload = json.loads(source_registry.read_text(encoding="utf-8"))
        for entry in payload.get("active_packets", []):
            packet_id = str(entry["packet_id"])
            manifest_value = Path(str(entry["manifest_path"]))
            target_manifest = (
                manifest_value
                if manifest_value.is_absolute()
                else PROJECT_ROOT / manifest_value
            ).resolve()
            source_packet = root / "packets" / packet_id
            source_manifest = source_packet / "manifest.json"
            source_records = source_packet / "records.csv"
            manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            records_relative = Path(str(manifest_payload["records"]["path"]))
            target_records = (target_manifest.parent / records_relative).resolve()
            target_manifest.parent.mkdir(parents=True, exist_ok=True)
            target_records.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_records, target_records)
            shutil.copy2(source_manifest, target_manifest)
        temporary = registry.with_suffix(registry.suffix + ".rollback.tmp")
        shutil.copy2(source_registry, temporary)
        temporary.replace(registry)

    def _create_snapshot(
        self,
        registry: Path,
        artifact_path: Path,
        artifact: Mapping[str, Any],
    ) -> dict[str, Any]:
        packet_ids, _ = _registry_packets(registry)
        registry_payload = json.loads(registry.read_text(encoding="utf-8"))
        identity = canonical_sha256(
            {
                "registry_sha256": sha256_file(registry),
                "evaluation_artifact_sha256": artifact["artifact_sha256"],
                "packet_ids": packet_ids,
            }
        )
        destination = self.snapshot_root / identity
        if not destination.exists():
            self.snapshot_root.mkdir(parents=True, exist_ok=True)
            temporary = self.snapshot_root / (".tmp-" + uuid.uuid4().hex)
            temporary.mkdir()
            shutil.copy2(registry, temporary / "registry.json")
            packet_root = temporary / "packets"
            for entry in registry_payload.get("active_packets", []):
                manifest_value = Path(str(entry["manifest_path"]))
                manifest_path = (
                    manifest_value
                    if manifest_value.is_absolute()
                    else PROJECT_ROOT / manifest_value
                ).resolve()
                packet = validate_evidence_packet(manifest_path)
                target = packet_root / packet.packet_id
                target.mkdir(parents=True)
                shutil.copy2(packet.manifest_path, target / "manifest.json")
                shutil.copy2(packet.records_path, target / "records.csv")
            evaluation_target = temporary / "evaluation"
            shutil.copytree(artifact_path.resolve().parent, evaluation_target)
            files = {
                str(path.relative_to(temporary)): sha256_file(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            }
            manifest_core = {
                "protocol": SNAPSHOT_PROTOCOL,
                "packet_ids": packet_ids,
                "scores": artifact["normalized_scores"],
                "mean_normalized_improvement": artifact["mean_normalized_improvement"],
                "registry_sha256": sha256_file(registry),
                "evaluation_artifact_sha256": artifact["artifact_sha256"],
                "files": files,
            }
            snapshot_sha = canonical_sha256(manifest_core)
            manifest = {**manifest_core, "snapshot_sha256": snapshot_sha}
            (temporary / "snapshot.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        manifest = json.loads(
            (destination / "snapshot.json").read_text(encoding="utf-8")
        )
        return {
            "packet_ids": manifest["packet_ids"],
            "scores": manifest["scores"],
            "mean_normalized_improvement": manifest["mean_normalized_improvement"],
            "registry_sha256": manifest["registry_sha256"],
            "evaluation_artifact_sha256": manifest["evaluation_artifact_sha256"],
            "snapshot_path": str(destination),
            "snapshot_sha256": manifest["snapshot_sha256"],
        }

    @staticmethod
    def _verify_snapshot(snapshot: Mapping[str, Any]) -> None:
        root = Path(str(snapshot["snapshot_path"])).resolve()
        manifest_path = root / "snapshot.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        core = dict(manifest)
        stored = core.pop("snapshot_sha256", None)
        if canonical_sha256(core) != stored or stored != snapshot.get(
            "snapshot_sha256"
        ):
            raise ValueError("closed-loop snapshot manifest hash mismatch")
        bound_fields = (
            "packet_ids",
            "scores",
            "mean_normalized_improvement",
            "registry_sha256",
            "evaluation_artifact_sha256",
        )
        if any(snapshot.get(field) != manifest.get(field) for field in bound_fields):
            raise ValueError("closed-loop state differs from its snapshot manifest")
        for relative, expected in manifest["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha256_file(path) != expected:
                raise ValueError("closed-loop snapshot file hash mismatch")

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)


__all__ = [
    "ARTIFACT_PROTOCOL",
    "REQUIRED_FROZEN_IMPLEMENTATION",
    "SNAPSHOT_PROTOCOL",
    "STATE_PROTOCOL",
    "EvidencePacketClosedLoop",
    "canonical_sha256",
    "finalize_evaluation_artifact",
    "load_verified_evaluation_artifact",
    "load_verified_evaluation_artifact_against_data",
]
