"""Checkpointed batch runner for the agent-development phase.

Each task writes an independent JSON artifact. A failed task writes a separate
error artifact and does not prevent the remaining tasks from running.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from run_all22_adaptive_agent_data import TASKS, _prepare_task, run_development


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("data/external/chembl/chembl_37/tdc22_cohorts"),
    )
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=Path(
            "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for task in options.tasks:
        output = options.output_dir / f"{task}.json"
        if output.exists() and not options.overwrite:
            print(json.dumps({"task": task, "status": "skipped_existing"}), flush=True)
            continue
        task_args = SimpleNamespace(
            task=task,
            phase="development",
            output=output,
            data_dir=options.data_dir,
            cohort_dir=options.cohort_dir,
            admission_report=options.admission_report,
            n_jobs=options.n_jobs,
            development_artifact=None,
            agent_confirmation_artifact=None,
        )
        try:
            prepared = _prepare_task(task_args, task)
            result = run_development(task_args, prepared)
            print(
                json.dumps(
                    {
                        "task": task,
                        "status": result["status"],
                        "selected_ratio": result.get("selection", {}).get(
                            "selected_ratio"
                        ),
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # retain a machine-readable checkpoint
            error_path = options.output_dir / f"{task}.error.json"
            error_path.write_text(
                json.dumps(
                    {
                        "task": task,
                        "status": "execution_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                json.dumps(
                    {"task": task, "status": "execution_failed", "error": str(exc)}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
