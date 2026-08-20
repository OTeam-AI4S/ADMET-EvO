"""Deterministic hidden-fault injection suite over discovery-only TDC splits."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tdc.benchmark_group import admet_group

from .data_governance import SourceManifest
from .endpoint_contracts import endpoint_contract
from .fault_detection import ScientificFaultDetector
from .hardness import profile_hardness
from .hidden_fault_benchmark import FaultChallengeResult, score_hidden_faults
from .tasks import resolve_task


def _columns(frame: pd.DataFrame) -> tuple[str, str]:
    return ("Drug" if "Drug" in frame.columns else "SMILES", "Y")


def _detect(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    task_name: str,
    *,
    manifest: SourceManifest | None = None,
    manifest_required: bool = False,
    metric_direction: str | None = None,
) -> tuple[set[str], dict[str, str]]:
    task = resolve_task(task_name)
    smiles_col, target_col = _columns(train)
    card = profile_hardness(
        train,
        valid,
        task,
        smiles_col,
        target_col,
        seed=20260819,
    )
    report = ScientificFaultDetector().detect(
        card,
        endpoint_contract(task.name),
        source_manifest=manifest,
        source_manifest_required=manifest_required,
        reported_metric_direction=metric_direction,
    )
    return set(report.detected_types), {
        item.fault_type: item.remediation for item in report.findings
    }


def _manifest(task_name: str, *, species: str | None = None, assay: str | None = None) -> SourceManifest:
    contract = endpoint_contract(task_name)
    return SourceManifest(
        source_id="hidden-source",
        endpoint=contract.tdc_name,
        origin="hidden-fault-fixture",
        license="benchmark-only",
        acquired_at="2026-08-19",
        assay_system=assay or contract.assay_system,
        species=species or contract.species or "not_applicable",
        unit="hidden-unit" if contract.task_type == "regression" else None,
        label_transform="identity",
        citation="hidden benchmark fixture",
    )


def run_hidden_fault_suite(
    data_dir: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Run injected challenges without ever materialising the TDC test split."""
    group = admet_group(path=str(data_dir))
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, set[str]]] = {}

    def frames(task_name: str):
        if task_name not in cache:
            train, valid = group.get_train_valid_split(
                benchmark=task_name,
                split_type="default",
                seed=1,
            )
            baseline, _ = _detect(train, valid, task_name)
            cache[task_name] = (train, valid, baseline)
        return cache[task_name]

    challenges: list[FaultChallengeResult] = []

    def evaluate(
        challenge_id: str,
        task_name: str,
        expected: tuple[str, ...],
        train: pd.DataFrame,
        valid: pd.DataFrame,
        baseline: set[str],
        **detect_kwargs: Any,
    ) -> None:
        started = time.monotonic()
        detected, remediations = _detect(
            train, valid, task_name, **detect_kwargs
        )
        novel = detected - baseline
        proposed = [remediations[item] for item in novel if item in remediations]
        valid_count = sum(bool(value.strip()) for value in proposed)
        challenges.append(
            FaultChallengeResult(
                challenge_id,
                task_name,
                expected,
                tuple(sorted(novel)),
                time.monotonic() - started,
                len(proposed),
                valid_count,
            )
        )

    train, valid, baseline = frames("HIA_Hou")
    evaluate("clean-control", "HIA_Hou", (), train.copy(), valid, baseline)

    smiles_col, target_col = _columns(train)
    invalid = pd.concat(
        [
            train,
            pd.DataFrame([{smiles_col: "not-a-valid-smiles", target_col: 1}]),
        ],
        ignore_index=True,
    )
    evaluate(
        "invalid-structure",
        "HIA_Hou",
        ("invalid_molecular_structure",),
        invalid,
        valid,
        baseline,
    )

    missing = pd.concat(
        [train, pd.DataFrame([{smiles_col: "CCCO", target_col: float("nan")}])],
        ignore_index=True,
    )
    evaluate("invalid-target", "HIA_Hou", ("invalid_target",), missing, valid, baseline)

    conflict_row = train.iloc[[0]].copy()
    conflict_row[target_col] = 1 - int(conflict_row[target_col].iloc[0])
    conflict = pd.concat([train, conflict_row], ignore_index=True)
    evaluate(
        "label-conflict",
        "HIA_Hou",
        ("conflicting_class_labels",),
        conflict,
        valid,
        baseline,
    )

    overlap = pd.concat([train, valid.iloc[:20]], ignore_index=True)
    evaluate(
        "split-overlap",
        "HIA_Hou",
        ("exact_split_overlap",),
        overlap,
        valid,
        baseline,
    )

    evaluate(
        "wrong-metric-direction",
        "HIA_Hou",
        ("wrong_metric_direction",),
        train,
        valid,
        baseline,
        metric_direction="minimize",
    )
    evaluate(
        "missing-provenance",
        "HIA_Hou",
        ("provenance_missing",),
        train,
        valid,
        baseline,
        manifest_required=True,
    )
    evaluate(
        "species-mismatch",
        "HIA_Hou",
        ("species_mismatch",),
        train,
        valid,
        baseline,
        manifest=_manifest("HIA_Hou", species="rat"),
    )
    evaluate(
        "assay-mismatch",
        "HIA_Hou",
        ("assay_mismatch",),
        train,
        valid,
        baseline,
        manifest=_manifest("HIA_Hou", assay="unrelated assay"),
    )

    reg_train, reg_valid, reg_baseline = frames("Caco2_Wang")
    evaluate(
        "unit-transform-mismatch",
        "Caco2_Wang",
        ("unit_or_transform_mismatch",),
        reg_train,
        reg_valid,
        reg_baseline,
        manifest=_manifest("Caco2_Wang"),
    )

    score = score_hidden_faults(challenges)
    payload = {
        "protocol": "tdc-discovery-hidden-fault-suite-v1",
        "claim_scope": "deterministic_integration_fixture_not_independent_blind_evaluation",
        "test_loaded": False,
        "results": [item.__dict__ for item in challenges],
        "score": score,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
