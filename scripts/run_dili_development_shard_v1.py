#!/usr/bin/env python3
"""Run one content-addressed DILI development seed for parallel execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (ROOT, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_evidence_packet_frozen_recipe_v54 as v54  # noqa: E402
from run_all22_optimal_recipe_adaptive_data import _load_mapping, _recipe_label, _runtime_blocker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    for field in ("protocol", "mapping", "data_dir", "output", "feature_cache_root", "artifact_root"):
        setattr(args, field, getattr(args, field).resolve())
    protocol, protocol_sha = v54._load_protocol(args.protocol)
    allowed = [int(value) for value in protocol["data_axis"]["development_seeds"]]
    if args.seed not in allowed:
        raise RuntimeError(f"seed {args.seed} is not preregistered: {allowed}")
    entry = protocol["tasks"]["DILI"]
    mapping, mapping_sha = _load_mapping(args.mapping)
    if mapping_sha != protocol["model_axis"]["mapping_sha256"]:
        raise RuntimeError("frozen mapping changed")
    frozen = mapping["DILI"]
    blocker = _runtime_blocker(frozen, args.device, ROOT)
    if blocker:
        raise RuntimeError(blocker)
    prepared, packet_audit = v54._verify_task_packets("DILI", entry, args.data_dir)
    frames, source_audits = v54._source_frames("DILI", entry, prepared)
    rows, split_audits = v54._evaluate(
        task_name="DILI", frozen=frozen, seeds=(args.seed,),
        configurations=entry["candidate_configurations"], source_frames=frames,
        data_dir=args.data_dir, cache_root=args.feature_cache_root,
        artifact_root=args.artifact_root, device=args.device, n_jobs=args.n_jobs,
        maximum_mass_ratio=float(protocol["data_axis"]["maximum_external_to_native_mass_ratio"]),
    )
    payload = {
        "protocol": v54.PROTOCOL_NAME, "protocol_sha256": protocol_sha,
        "phase": "development", "task": "DILI", "status": "completed_shard",
        "frozen_recipe": frozen, "frozen_recipe_label": _recipe_label(frozen),
        "seeds": [args.seed], "rows": rows, "packet_audit": packet_audit,
        "source_set_audits": source_audits, "split_audits": split_audits,
        "safety": {"test_structures_used_only_for_decontamination": True,
                   "test_labels_loaded": False, "test_results_used_for_selection": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"seed": args.seed, "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
