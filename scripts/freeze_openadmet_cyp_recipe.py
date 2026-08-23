"""Freeze the selected OpenADMET CYP recipe before certification or blind inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_SEEDS = (1, 2, 3)
EXECUTION_SOURCES = (
    "scripts/openadmet_cyp_metrics.py",
    "scripts/openadmet_multifidelity_model.py",
    "scripts/openadmet_split_safety.py",
    "scripts/run_openadmet_cyp_multifidelity.py",
    "scripts/run_openadmet_cyp_tree_baselines.py",
    "scripts/fit_openadmet_cyp_oof_stack.py",
    "scripts/extract_minimol_embeddings.py",
    "scripts/run_openadmet_cyp_frozen_b1.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_training_artifact(payload: dict[str, Any], label: str) -> None:
    if payload.get("training_only") is not True:
        raise RuntimeError(f"{label} is not training-only")
    if payload.get("blind_test_opened") is not False:
        raise RuntimeError(f"{label} indicates blind-test access")
    if payload.get("sealed_certification_opened") is not False:
        raise RuntimeError(f"{label} indicates certification access")


def freeze_recipe(
    *,
    component_paths: dict[str, Path],
    stack_path: Path | None,
    split_path: Path,
    longform_path: Path,
    embedding_path: Path | None,
    endpoint_contract_path: Path,
    firewall_manifest_path: Path,
    data_manifest_path: Path,
    submission_schema_path: Path,
    output_path: Path,
    root: Path,
) -> dict[str, Any]:
    if not 1 <= len(component_paths) <= 4:
        raise ValueError("Frozen recipe requires one to four components")
    root = root.resolve()
    components = {}
    shared_longform_hashes = set()
    for label, path in sorted(component_paths.items()):
        path = path.resolve()
        payload = _load_json(path)
        _safe_training_artifact(payload, f"component {label}")
        if tuple(payload.get("outer_folds", [])) != REQUIRED_FOLDS:
            raise RuntimeError(f"component {label} lacks five-fold Stage-B coverage")
        if tuple(payload.get("seeds", [])) != REQUIRED_SEEDS:
            raise RuntimeError(f"component {label} lacks three-seed Stage-B coverage")
        shared_longform_hashes.add(payload["longform_sha256"])
        components[label] = {
            "path": str(path),
            "sha256": _sha256(path),
            "family": payload.get("variant", payload.get("baseline")),
            "longform_sha256": payload["longform_sha256"],
            "embedding_sha256": payload.get(
                "embedding_sha256",
                payload.get("feature_contract", {}).get("embedding_sha256"),
            ),
            "external_bridge": payload.get("external_bridge"),
        }
    if len(shared_longform_hashes) != 1:
        raise RuntimeError("Frozen components do not share one long-form data artifact")

    if len(components) == 1:
        if stack_path is not None:
            raise ValueError("Single-component fallback must not supply a synthetic stack")
        only_label = next(iter(components))
        weights = {
            "regression": {only_label: 1.0},
            "classification": {only_label: 1.0},
        }
        stack_contract = {
            "mode": "single_component_fallback",
            "path": None,
            "sha256": None,
            "weights": weights,
        }
    else:
        if stack_path is None:
            raise ValueError("Multi-component recipe requires a nested OOF stack")
        stack_path = stack_path.resolve()
        stack = _load_json(stack_path)
        _safe_training_artifact(stack, "E0 stack")
        expected_component_hashes = {
            label: values["sha256"] for label, values in components.items()
        }
        observed_component_hashes = {
            label: values["result_sha256"]
            for label, values in stack.get("components", {}).items()
        }
        if observed_component_hashes != expected_component_hashes:
            raise RuntimeError("E0 stack does not bind exactly the selected components")
        weights = stack.get("final_all_oof_weights_for_future_frozen_refit")
        if set(weights or {}) != {"regression", "classification"}:
            raise RuntimeError("E0 stack lacks final regression/classification weights")
        stack_contract = {
            "mode": "nested_oof_stack",
            "path": str(stack_path),
            "sha256": _sha256(stack_path),
            "weights": weights,
        }

    split_path = split_path.resolve()
    split_manifest_path = split_path.with_suffix(".manifest.json")
    split_manifest = _load_json(split_manifest_path)
    if split_manifest.get("training_only") is not True or split_manifest.get("blind_test_opened") is not False:
        raise RuntimeError("Split manifest is not training-only")
    if split_manifest.get("output", {}).get("sha256") != _sha256(split_path):
        raise RuntimeError("Split artifact hash mismatch")

    longform_path = longform_path.resolve()
    longform_manifest_path = longform_path.with_suffix(".manifest.json")
    longform_manifest = _load_json(longform_manifest_path)
    if longform_manifest.get("training_only") is not True or longform_manifest.get("blind_test_opened") is not False:
        raise RuntimeError("Long-form manifest is not training-only")
    if longform_manifest.get("output_sha256") != _sha256(longform_path):
        raise RuntimeError("Long-form artifact hash mismatch")
    if longform_manifest.get("split_sha256") != _sha256(split_path):
        raise RuntimeError("Long-form artifact does not bind the frozen split")
    if shared_longform_hashes != {_sha256(longform_path)}:
        raise RuntimeError("Components do not bind the supplied long-form artifact")

    if embedding_path is None:
        if any(value.get("embedding_sha256") for value in components.values()):
            raise RuntimeError("Selected component requires a frozen embedding")
        embedding_contract = {"mode": "not_used_by_selected_components", "path": None, "sha256": None, "manifest_path": None, "manifest_sha256": None, "weight_sha256": None, "license_sha256": None}
    else:
        embedding_path = embedding_path.resolve()
        embedding_manifest_path = embedding_path.with_suffix(".manifest.json")
        embedding_manifest = _load_json(embedding_manifest_path)
        if embedding_manifest.get("output_sha256") != _sha256(embedding_path):
            raise RuntimeError("Embedding artifact hash mismatch")
        if embedding_manifest.get("official_test_loaded") is not False:
            raise RuntimeError("Embedding artifact indicates blind-test access")
        embedding_contract = {"mode": "frozen_embedding", "path": str(embedding_path), "sha256": _sha256(embedding_path), "manifest_path": str(embedding_manifest_path), "manifest_sha256": _sha256(embedding_manifest_path), "weight_sha256": embedding_manifest.get("weight_sha256"), "license_sha256": embedding_manifest.get("license_sha256")}

    endpoint_contract_path = endpoint_contract_path.resolve()
    firewall_manifest_path = firewall_manifest_path.resolve()
    data_manifest_path = data_manifest_path.resolve()
    submission_schema_path = submission_schema_path.resolve()
    firewall = _load_json(firewall_manifest_path)
    if firewall.get("test_labels_loaded") is not False or firewall.get("leaderboard_feedback_used") is not False:
        raise RuntimeError("Blind firewall manifest is not label/leaderboard blind")
    implementation_hash = firewall.get("standardization", {}).get("implementation_hash")
    if not isinstance(implementation_hash, str) or len(implementation_hash) != 64:
        raise RuntimeError("Blind firewall implementation hash is not frozen")

    source_hashes = {}
    for relative in EXECUTION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen execution source: {path}")
        source_hashes[relative] = _sha256(path)
    payload: dict[str, Any] = {
        "version": "openadmet-cyp-frozen-recipe-v1",
        "frozen_before_certification": True,
        "frozen_before_blind_inference": True,
        "certification_opened": False,
        "blind_test_labels_loaded": False,
        "leaderboard_feedback_used": False,
        "selection_reopened": False,
        "recipe_mode": stack_contract["mode"],
        "required_outer_folds": list(REQUIRED_FOLDS),
        "required_seeds": list(REQUIRED_SEEDS),
        "components": components,
        "stack": stack_contract,
        "split": {
            "path": str(split_path),
            "sha256": _sha256(split_path),
            "manifest_path": str(split_manifest_path),
            "manifest_sha256": _sha256(split_manifest_path),
        },
        "longform": {
            "path": str(longform_path),
            "sha256": _sha256(longform_path),
            "manifest_path": str(longform_manifest_path),
            "manifest_sha256": _sha256(longform_manifest_path),
        },
        "embedding": embedding_contract,
        "endpoint_contract": {
            "path": str(endpoint_contract_path),
            "sha256": _sha256(endpoint_contract_path),
        },
        "blind_firewall": {
            "path": str(firewall_manifest_path),
            "sha256": _sha256(firewall_manifest_path),
            "implementation_hash": implementation_hash,
        },
        "data_manifest": {"path": str(data_manifest_path), "sha256": _sha256(data_manifest_path)},
        "submission_schema": {"path": str(submission_schema_path), "sha256": _sha256(submission_schema_path)},
        "execution_source_sha256": source_hashes,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    frozen = {**payload, "recipe_sha256": digest}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = _load_json(output_path)
        if existing.get("recipe_sha256") != digest:
            raise RuntimeError("Refusing to overwrite a different frozen CYP recipe")
    else:
        output_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


def _assign(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("Components must use LABEL=PATH")
        output[label] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", action="append", required=True)
    parser.add_argument("--stack", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--longform", type=Path, required=True)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--endpoint-contract", type=Path, required=True)
    parser.add_argument("--firewall-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--submission-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = freeze_recipe(
        component_paths=_assign(args.component),
        stack_path=args.stack,
        split_path=args.split,
        longform_path=args.longform,
        embedding_path=args.embedding,
        endpoint_contract_path=args.endpoint_contract,
        firewall_manifest_path=args.firewall_manifest,
        data_manifest_path=args.data_manifest,
        submission_schema_path=args.submission_schema,
        output_path=args.output,
        root=args.root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
