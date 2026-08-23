"""Acquire and content-hash the preregistered v6 source candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tdc
from tdc.single_pred import HTS, QM, Tox
from tdc.utils import retrieve_label_name_list


LOADERS = {"HTS": HTS, "QM": QM, "Tox": Tox}
MULTILABEL = {"ToxCast", "hERG_Central", "QM7b", "QM8", "QM9"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_registry(path: Path) -> dict:
    payload = json.loads(path.read_text())
    unbound = {key: value for key, value in payload.items() if key != "sha256"}
    expected = hashlib.sha256(
        (json.dumps(unbound, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if payload.get("sha256") != expected:
        raise RuntimeError("Fresh-source registry self-hash mismatch")
    return payload


def main() -> None:
    registry_path = Path("reports/ecvoi_v6_fresh_source_registry_v1.json")
    cache = Path("external/ecvoi_v6_tdc_cache")
    cache.mkdir(parents=True, exist_ok=True)
    registry = _load_registry(registry_path)
    acquisitions = []
    failures = []
    for candidate in registry["candidates"]:
        dataset = candidate["dataset"]
        kwargs = {"name": dataset, "path": str(cache)}
        if dataset in MULTILABEL:
            labels = sorted(set(retrieve_label_name_list(dataset)))
            if not labels:
                failures.append({"dataset": dataset, "error": "no labels returned"})
                continue
            kwargs["label_name"] = labels[0]
        try:
            obj = LOADERS[candidate["loader"]](**kwargs)
            frame = obj.get_data(format="df")
            acquisitions.append({
                "dataset": dataset,
                "loader": candidate["loader"],
                "source_id": candidate["source_id"],
                "bootstrap_rows": len(frame),
                "bootstrap_columns": list(frame.columns),
                "bootstrap_label": kwargs.get("label_name"),
            })
        except Exception as exc:
            failures.append({"dataset": dataset, "error": repr(exc)})
    files = {
        str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(cache.rglob("*")) if path.is_file()
    }
    report = {
        "version": "ecvoi-v6-fresh-source-acquisition-v1",
        "source_registry_sha256": registry["sha256"],
        "tdc_version": getattr(tdc, "__version__", "unknown"),
        "acquisitions": acquisitions,
        "failures": failures,
        "raw_files": files,
        "all_registry_candidates_acquired": len(failures) == 0,
        "model_actions_run": False,
        "action_outcomes_computed": False,
        "protected_labels_loaded": False,
        "test_or_leaderboard_feedback_used": False,
        "tdc_19_unlocked": False,
    }
    output = Path("reports/ecvoi_v6_fresh_source_acquisition_v1.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "acquired": len(acquisitions), "failed": len(failures),
        "files": len(files), "bytes": sum(item["bytes"] for item in files.values()),
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
