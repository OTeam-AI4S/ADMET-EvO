"""Execute the frozen EC-VOI v4 development matched-state matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from admet_agent.matched_state import construct_matched_state, matched_action_spec
from admet_agent.profiling import profile_task
from admet_agent.runner import ADMETAgent
from admet_agent.tasks import resolve_task


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    claimed = protocol.pop("sha256", None)
    if claimed != _digest(protocol):
        raise ValueError("Matched-state protocol SHA-256 is invalid")
    if protocol.get("test_loaded") is not False:
        raise ValueError("Matched-state protocol must forbid test access")
    if protocol.get("leaderboard_loaded") is not False:
        raise ValueError("Matched-state protocol must forbid leaderboard access")
    for name, expected in protocol["implementation_sha256"].items():
        source = Path(name)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen implementation hash mismatch: {name}")
    return {**protocol, "sha256": claimed}


def _integrity_decision(registry: dict, protocol: dict) -> dict:
    records = registry["records"]
    expected = {
        f"{task}|{seed}|{state}|{action}"
        for task in protocol["tasks"]
        for seed in protocol["screen_seeds"]
        for state in protocol["states"]
        for action in protocol["actions"]
    }
    present = expected.intersection(records)
    complete = [key for key in present if records[key]["result"]["status"] == "completed"]
    no_label_state = all(
        records[key]["state_audit"]["validation_labels_read_for_state"] is False
        for key in present
    )
    identity_safe = all(
        records[key]["spec"]["data_policy"] == "identity_safe" for key in present
    )
    body = {
        "version": "ecvoi-v4-matched-state-integrity-gate-v1",
        "protocol_sha256": protocol["sha256"],
        "expected_records": len(expected),
        "present_records": len(present),
        "completed_records": len(complete),
        "validation_labels_read_for_state": not no_label_state,
        "all_identity_safe": identity_safe,
        "screen_records_sha256": _digest({key: records[key] for key in sorted(present)}),
        "passed": len(present) == len(expected) == len(complete) and no_label_state and identity_safe,
        "test_loaded": False,
        "leaderboard_loaded": False,
    }
    return {**body, "sha256": _digest(body)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("cache/features"))
    parser.add_argument("--phase", choices=("screen", "confirmation", "all"), default="screen")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.run_dir / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("protocol_sha256") != protocol["sha256"]:
            raise ValueError("Run registry belongs to another protocol")
    else:
        registry = {
            "version": "ecvoi-v4-matched-state-runner-v1",
            "protocol_sha256": protocol["sha256"],
            "records": {},
            "test_loaded": False,
            "leaderboard_loaded": False,
        }
        _atomic_json(registry_path, registry)

    agent = ADMETAgent(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        device="cpu",
        n_jobs=args.workers,
        feature_cache_dir=args.feature_cache_dir,
        persist_discovery_models=False,
    )
    phases = ("screen", "confirmation") if args.phase == "all" else (args.phase,)
    failures = 0
    for phase in phases:
        if phase == "screen":
            seeds = protocol["screen_seeds"]
        else:
            current = _integrity_decision(registry, protocol)
            decision_path = args.run_dir / "screen_integrity_decision.json"
            if decision_path.exists():
                frozen = json.loads(decision_path.read_text(encoding="utf-8"))
                if frozen != current:
                    raise RuntimeError("Frozen screen integrity decision changed")
            else:
                _atomic_json(decision_path, current)
            if not current["passed"]:
                raise RuntimeError("Screen integrity gate failed; confirmation is forbidden")
            seeds = protocol["confirmation_seeds"]

        for task in protocol["tasks"]:
            for seed in seeds:
                train, valid = agent.group.get_train_valid_split(
                    benchmark=task, split_type="default", seed=seed
                )
                for state in protocol["states"]:
                    state_train, audit = construct_matched_state(
                        train,
                        valid,
                        state=state,
                        seed=seed,
                        smiles_col="Drug",
                        target_col="Y",
                    )
                    profile = profile_task(
                        state_train, valid, resolve_task(task), "Drug", "Y"
                    )
                    for action in protocol["actions"]:
                        key = f"{task}|{seed}|{state}|{action}"
                        if key in registry["records"]:
                            continue
                        spec = matched_action_spec(
                            task,
                            action=action,
                            seed=seed,
                            workers=args.workers,
                            max_minutes=protocol["max_minutes_per_fit"],
                        )
                        result = agent._execute(
                            spec,
                            state_train,
                            valid,
                            "Drug",
                            "Y",
                            profile,
                            timeout_seconds=60 * protocol["max_minutes_per_fit"],
                        )
                        registry["records"][key] = {
                            "task": task,
                            "seed": seed,
                            "state": state,
                            "action": action,
                            "phase": phase,
                            "state_audit": audit.to_dict(),
                            "profile": profile.to_dict(),
                            "spec": spec.to_dict(),
                            "result": result.to_dict(),
                            "test_loaded": False,
                            "leaderboard_loaded": False,
                        }
                        registry["updated_at"] = datetime.now(timezone.utc).isoformat()
                        _atomic_json(registry_path, registry)
                        print(
                            f"{key}: {result.status} score={result.validation_score:.6f} "
                            f"elapsed={result.elapsed_seconds:.1f}s",
                            flush=True,
                        )
                        failures += result.status != "completed"
        if phase == "screen":
            _atomic_json(
                args.run_dir / "screen_integrity_decision.json",
                _integrity_decision(registry, protocol),
            )
    print(f"records={len(registry['records'])}; failures={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
