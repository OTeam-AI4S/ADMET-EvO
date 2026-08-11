import argparse
import json
from pathlib import Path

from .guard import validate_experiment
from .schemas import ExperimentSpec
from .tasks import TASKS, resolve_task


def _selection(args) -> list[str]:
    names = [item.name for item in TASKS.values()] if args.all else (args.task or [])
    if not names:
        raise SystemExit("Choose --all or provide at least one --task.")
    return [resolve_task(name).name for name in names]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="admet-agent")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the canonical 22 tasks")
    commands.add_parser("tools", help="List registered representation and model tools")

    search = commands.add_parser("search", help="Run validation-only model search")
    search.add_argument("--task", action="append", help="Task name; repeat as needed")
    search.add_argument("--all", action="store_true", help="Search all 22 tasks")
    search.add_argument("--seed", type=int, default=1)
    search.add_argument("--rounds", type=int, default=2, help="Sequential plan/evaluate/refine rounds")
    search.add_argument("--validation-seeds", default=None,
                        help="Comma-separated validation seeds for robust finalist selection")
    search.add_argument("--robust-top-k", type=int, default=3,
                        help="Candidates promoted to multi-seed validation")
    search.add_argument("--no-ensemble", action="store_true",
                        help="Disable validation-prediction blending")
    search.add_argument("--data-dir", type=Path, default=Path("data"))
    search.add_argument("--run-dir", type=Path, default=Path("runs/v1"))
    search.add_argument("--brain", choices=("rules", "openai", "hybrid"), default="rules")
    search.add_argument(
        "--api-config", type=Path,
        default=Path.home() / ".config" / "admet-agent" / "openai.env",
    )
    search.add_argument("--model", default="gpt-5.6-sol")
    search.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    search.add_argument("--workers", type=int, default=8)

    final = commands.add_parser("finalize", help="Run frozen finalists on the TDC test protocol")
    final.add_argument("--task", action="append", help="Task name; repeat as needed")
    final.add_argument("--all", action="store_true", help="Finalize all 22 tasks")
    final.add_argument("--seeds", default="1,2,3,4,5")
    final.add_argument("--finalists", type=Path, required=True,
                       help="JSON mapping tasks to frozen experiment specs (model strings remain supported)")
    final.add_argument("--data-dir", type=Path, default=Path("data"))
    final.add_argument("--run-dir", type=Path, default=Path("runs/v1"))
    final.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    final.add_argument("--workers", type=int, default=8)

    validate = commands.add_parser("validate-spec", help="Validate an experiment JSON without running it")
    validate.add_argument("path", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "list":
        for spec in TASKS.values():
            print(f"{spec.name:40} {spec.task_type:14} {spec.metric:8} {spec.category}")
        return
    if args.command == "tools":
        from .registry import planner_catalog

        print(json.dumps(planner_catalog(), indent=2))
        return
    if args.command == "validate-spec":
        experiment = ExperimentSpec.from_dict(json.loads(args.path.read_text(encoding="utf-8")))
        validate_experiment(experiment)
        print("valid")
        return

    names = _selection(args)
    from .runner import ADMETAgent

    brain = None
    if args.command == "search" and args.brain in {"openai", "hybrid"}:
        from .openai_brain import OpenAIPlanner

        planner = OpenAIPlanner(
            config_path=args.api_config, model=args.model,
            audit_path=args.run_dir / "api_usage.jsonl",
        )
        if args.brain == "hybrid":
            from .brain import HybridBrain

            brain = HybridBrain(planner)
        else:
            brain = planner
    agent = ADMETAgent(args.data_dir, args.run_dir, brain=brain, device=args.device, n_jobs=args.workers)
    agent.environment_manifest()
    if args.command == "search":
        validation_seeds = (
            [int(value) for value in args.validation_seeds.split(",")]
            if args.validation_seeds else [args.seed]
        )
        summaries = {
            name: agent.search(
                name,
                seed=args.seed,
                rounds=args.rounds,
                validation_seeds=validation_seeds,
                robust_top_k=args.robust_top_k,
                ensembles=not args.no_ensemble,
            )
            for name in names
        }
        print(json.dumps(summaries, indent=2))
        return
    finalists_raw = json.loads(args.finalists.read_text(encoding="utf-8"))
    finalists = {resolve_task(name).name: model for name, model in finalists_raw.items()}
    missing = set(names) - set(finalists)
    if missing:
        raise SystemExit(f"Missing frozen finalists for: {sorted(missing)}")
    seeds = [int(value) for value in args.seeds.split(",")]
    result = agent.finalize(names, seeds, finalists)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
