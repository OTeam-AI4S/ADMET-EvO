import hashlib
import json

import pytest

from scripts.build_v24_freeze_manifest import (
    _finalist_config,
    attach_smoke_evidence,
    require_seed_artifacts,
    sha256_file,
)


def test_sha256_file_is_content_addressed(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"validation-only")

    assert sha256_file(artifact) == hashlib.sha256(b"validation-only").hexdigest()


def test_require_seed_artifacts_enforces_all_five_seeds(tmp_path):
    for seed in range(1, 6):
        seed_dir = tmp_path / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "predictions.npz").write_bytes(str(seed).encode())

    paths = require_seed_artifacts(tmp_path, "predictions.npz")
    assert [path.parent.name for path in paths] == [
        "seed_1",
        "seed_2",
        "seed_3",
        "seed_4",
        "seed_5",
    ]

    paths[-1].unlink()
    with pytest.raises(FileNotFoundError, match="seed_5"):
        require_seed_artifacts(tmp_path, "predictions.npz")


def test_finalist_config_is_directly_consumable_by_finalizer():
    protocol = {
        "model_size": "84m",
        "learning_rate": 1e-4,
        "epochs": 20,
        "batch_size": 8,
        "inner_folds": 5,
    }

    assert _finalist_config("finetuned_unimol2", protocol) == {
        "kind": "recipe",
        "recipe": "finetuned_unimol2",
        "protocol": protocol,
    }


def test_smoke_evidence_requires_hash_config_and_leakage_checks(tmp_path):
    mapping = tmp_path / "runs" / "v24-freeze" / "finalists.mapping.json"
    mapping.parent.mkdir(parents=True)
    protocol = {
        "model_size": "84m",
        "learning_rate": 1e-4,
        "epochs": 20,
        "batch_size": 16,
        "inner_folds": 5,
    }
    finalist = _finalist_config("finetuned_unimol2", protocol)
    mapping.write_text(json.dumps({"DILI": finalist}), encoding="utf-8")
    result_path = (
        tmp_path / "runs" / "v24-recipe-smoke" / "DILI" / "seed_1" / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "task": "DILI",
                "score": 0.8,
                "elapsed_seconds": 1.0,
                "frozen_config": {"recipe": "finetuned_unimol2", "protocol": protocol},
                "mapping_sha256": sha256_file(mapping),
                "official_test_loaded": False,
                "test_labels_passed_to_executor": False,
                "reproducibility": {
                    "requested_seed": 1,
                    "training_seed": 1,
                    "split_seed": 1,
                },
                "environment": {
                    "cuda": "13.0",
                    "gpu": "A100",
                    "unimol_tools": "0.1.5",
                },
            }
        ),
        encoding="utf-8",
    )
    candidate = {
        "task": "DILI",
        "finalist_config": finalist,
        "execution_ready": False,
        "ready_for_official_test": False,
        "execution_blocker": "smoke",
    }

    attach_smoke_evidence(tmp_path, candidate, mapping)

    assert candidate["smoke_test"]["status"] == "valid"
    assert candidate["execution_ready"] is True
    assert candidate["execution_blocker"] is None

    result = json.loads(result_path.read_text())
    result["official_test_loaded"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    candidate["execution_ready"] = False
    attach_smoke_evidence(tmp_path, candidate, mapping)
    assert candidate["smoke_test"]["status"] == "invalid"
