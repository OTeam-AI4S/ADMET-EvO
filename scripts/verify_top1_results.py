"""Verify reproduced TDC scores and task-specific finalist freeze hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_runs(run_root: Path, manifest_path: Path, tolerance: float = 5e-4) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_seeds = manifest["protocol"]["seeds"]
    messages = []
    for task, expected in manifest["results"].items():
        result_path = run_root / task / "final" / "results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing reproduced result: {result_path}")
        actual = json.loads(result_path.read_text(encoding="utf-8"))
        if actual["seeds"] != expected_seeds:
            raise AssertionError(
                f"{task}: expected seeds {expected_seeds}, got {actual['seeds']}"
            )
        if actual["freeze_sha256"] != expected["freeze_sha256"]:
            raise AssertionError(
                f"{task}: freeze hash mismatch: {actual['freeze_sha256']}"
            )
        score = actual["results"][expected["dataset_key"]]
        for label, observed, target in (
            ("mean", float(score[0]), float(expected["mean"])),
            ("std", float(score[1]), float(expected["std"])),
        ):
            if abs(observed - target) > tolerance:
                raise AssertionError(
                    f"{task}: {label} mismatch: expected {target}, got {observed}"
                )
        messages.append(
            f"{task}: verified {score[0]:.3f} +/- {score[1]:.3f} "
            f"and freeze {actual['freeze_sha256']}"
        )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/tdc_top1_results.json"),
    )
    parser.add_argument("--tolerance", type=float, default=5e-4)
    args = parser.parse_args()
    for message in verify_runs(args.run_root, args.manifest, args.tolerance):
        print(message)


if __name__ == "__main__":
    main()
