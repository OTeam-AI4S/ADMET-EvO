import hashlib
import json

import pytest

from scripts.freeze_openadmet_cyp_recipe import EXECUTION_SOURCES, freeze_recipe


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def _fixture(tmp_path, *, frozen_firewall=True):
    root = tmp_path / "root"
    for relative in EXECUTION_SOURCES:
        _write(root / relative, f"# {relative}\n")
    split = tmp_path / "split.csv"
    _write(split, "inchikey,partition\na,development\n")
    _write(
        split.with_suffix(".manifest.json"),
        {
            "training_only": True,
            "blind_test_opened": False,
            "output": {"sha256": _sha256(split)},
        },
    )
    longform = tmp_path / "longform.csv"
    _write(longform, "inchikey,target\na,1\n")
    _write(
        longform.with_suffix(".manifest.json"),
        {
            "training_only": True,
            "blind_test_opened": False,
            "output_sha256": _sha256(longform),
            "split_sha256": _sha256(split),
        },
    )
    embedding = tmp_path / "embedding.npy"
    embedding.write_bytes(b"embedding")
    _write(
        embedding.with_suffix(".manifest.json"),
        {
            "output_sha256": _sha256(embedding),
            "official_test_loaded": False,
            "weight_sha256": "b" * 64,
            "license_sha256": "c" * 64,
        },
    )
    components = {}
    for label in ("B1", "M2"):
        path = tmp_path / f"{label}.json"
        _write(
            path,
            {
                "training_only": True,
                "blind_test_opened": False,
                "sealed_certification_opened": False,
                "outer_folds": [0, 1, 2, 3, 4],
                "seeds": [1, 2, 3],
                "longform_sha256": _sha256(longform),
                "baseline" if label == "B1" else "variant": label,
            },
        )
        components[label] = path
    stack = tmp_path / "stack.json"
    _write(
        stack,
        {
            "training_only": True,
            "blind_test_opened": False,
            "sealed_certification_opened": False,
            "components": {
                label: {"result_sha256": _sha256(path)}
                for label, path in components.items()
            },
            "final_all_oof_weights_for_future_frozen_refit": {
                "regression": {"B1": 0.5, "M2": 0.5},
                "classification": {"B1": 0.5, "M2": 0.5},
            },
        },
    )
    endpoint = tmp_path / "endpoint.json"
    _write(endpoint, {"contract": "frozen"})
    firewall = tmp_path / "firewall.json"
    _write(
        firewall,
        {
            "test_labels_loaded": False,
            "leaderboard_feedback_used": False,
            "standardization": {
                "implementation_hash": "d" * 64 if frozen_firewall else "pending"
            },
        },
    )
    data_manifest = tmp_path / "data-manifest.json"
    _write(data_manifest, {"files": {}})
    submission_schema = tmp_path / "submission-schema.json"
    _write(submission_schema, {"activity_dataset_size": 750})
    return {
        "component_paths": components,
        "stack_path": stack,
        "split_path": split,
        "longform_path": longform,
        "embedding_path": embedding,
        "endpoint_contract_path": endpoint,
        "firewall_manifest_path": firewall,
        "data_manifest_path": data_manifest,
        "submission_schema_path": submission_schema,
        "output_path": tmp_path / "frozen.json",
        "root": root,
    }


def test_recipe_freeze_binds_every_artifact_before_certification(tmp_path):
    values = _fixture(tmp_path)
    result = freeze_recipe(**values)
    assert result["frozen_before_certification"] is True
    assert result["blind_test_labels_loaded"] is False
    assert len(result["recipe_sha256"]) == 64
    assert set(result["execution_source_sha256"]) == set(EXECUTION_SOURCES)
    assert result["stack"]["weights"]["regression"]["M2"] == 0.5


def test_pending_firewall_hash_blocks_recipe_freeze(tmp_path):
    values = _fixture(tmp_path, frozen_firewall=False)
    with pytest.raises(RuntimeError, match="implementation hash"):
        freeze_recipe(**values)


def test_single_component_fallback_has_identity_weights_without_fake_stack(tmp_path):
    values = _fixture(tmp_path)
    values["component_paths"] = {"B1": values["component_paths"]["B1"]}
    values["stack_path"] = None
    result = freeze_recipe(**values)
    assert result["recipe_mode"] == "single_component_fallback"
    assert result["stack"]["path"] is None
    assert result["stack"]["weights"] == {
        "regression": {"B1": 1.0},
        "classification": {"B1": 1.0},
    }


def test_single_component_fallback_rejects_synthetic_stack(tmp_path):
    values = _fixture(tmp_path)
    values["component_paths"] = {"B1": values["component_paths"]["B1"]}
    with pytest.raises(ValueError, match="must not supply"):
        freeze_recipe(**values)
