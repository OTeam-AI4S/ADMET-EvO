#!/usr/bin/env python3
"""Run a development task for a specified seed shard without changing the main runner."""

from __future__ import annotations

import argparse
import sys

import run_all22_optimal_recipe_adaptive_data as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    runner.DEVELOPMENT_SEEDS = tuple(args.seeds)
    sys.argv = [
        "run_all22_optimal_recipe_adaptive_data.py",
        "--phase",
        "development",
        "--tasks",
        args.task,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--n-jobs",
        str(args.n_jobs),
        "--overwrite",
    ]
    runner.main()


if __name__ == "__main__":
    main()
