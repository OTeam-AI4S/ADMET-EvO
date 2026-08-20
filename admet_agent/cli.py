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

    diagnose = commands.add_parser(
        "diagnose", help="Profile endpoint hardness without training or test access"
    )
    diagnose.add_argument("--task", action="append", help="Task name; repeat as needed")
    diagnose.add_argument("--all", action="store_true", help="Diagnose all 22 tasks")
    diagnose.add_argument("--seed", type=int, default=1)
    diagnose.add_argument("--data-dir", type=Path, default=Path("data"))
    diagnose.add_argument("--run-dir", type=Path, default=Path("runs/diagnosis"))
    diagnose.add_argument(
        "--feature-cache-dir", type=Path, default=Path("cache/features")
    )
    diagnose.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    diagnose.add_argument("--workers", type=int, default=8)
    diagnose.add_argument("--effect-priority", type=float, default=0.80)
    diagnose.add_argument(
        "--quiet", action="store_true", help="Print task names instead of full packets"
    )

    autonomy = commands.add_parser(
        "benchmark-agent",
        help="Score autonomous research episodes over the fixed 22-task suite",
    )
    autonomy.add_argument(
        "episodes", type=Path, help="JSON list or {episodes: [...]} file"
    )
    autonomy.add_argument("--output", type=Path, default=None)

    report = commands.add_parser(
        "summarize-diagnosis",
        help="Aggregate discovery-only Hardness over all 22 tasks",
    )
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--seed", type=int, default=1)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)

    search_report = commands.add_parser(
        "summarize-search", help="Aggregate discovery-only multi-seed search evidence"
    )
    search_report.add_argument("--run-dir", type=Path, required=True)
    search_report.add_argument("--output-json", type=Path, required=True)
    search_report.add_argument("--output-markdown", type=Path, required=True)

    preregister = commands.add_parser(
        "init-policy-benchmark",
        help="Write the immutable TDC-22 policy benchmark protocol",
    )
    preregister.add_argument("path", type=Path)

    compare = commands.add_parser(
        "compare-policies", help="Compare completed policy arms over all 22 tasks"
    )
    compare.add_argument("results", type=Path)
    compare.add_argument("--agent-arm", default="agent")
    compare.add_argument("--output", type=Path, default=None)

    collect = commands.add_parser(
        "collect-policy-results",
        help="Collect policy-arm search results under one protocol",
    )
    collect.add_argument("--protocol", type=Path, required=True)
    collect.add_argument(
        "--arm-run",
        action="append",
        required=True,
        help="ARM=RUN_DIR; repeat for each completed arm",
    )
    collect.add_argument("--output", type=Path, required=True)

    faults = commands.add_parser(
        "benchmark-faults", help="Score hidden scientific/data fault challenges"
    )
    faults.add_argument("challenges", type=Path)
    faults.add_argument("--output", type=Path, default=None)

    injected_faults = commands.add_parser(
        "run-hidden-fault-suite",
        help="Run deterministic discovery-only fault injections",
    )
    injected_faults.add_argument("--data-dir", type=Path, default=Path("data"))
    injected_faults.add_argument("--output", type=Path, required=True)

    freeze_audit = commands.add_parser(
        "audit-freeze",
        help="Authorize a complete discovery-only 22-task finalist freeze",
    )
    freeze_audit.add_argument("--run-dir", type=Path, required=True)
    freeze_audit.add_argument("--protocol", type=Path, required=True)
    freeze_audit.add_argument("--output-audit", type=Path, required=True)
    freeze_audit.add_argument("--output-finalists", type=Path, required=True)

    curate = commands.add_parser(
        "curate-external",
        help="Validate, clean and quarantine an external endpoint table",
    )
    curate.add_argument("--task", required=True)
    curate.add_argument("--table", type=Path, required=True)
    curate.add_argument("--manifest", type=Path, required=True)
    curate.add_argument("--smiles-col", required=True)
    curate.add_argument("--target-col", required=True)
    curate.add_argument("--output-dir", type=Path, required=True)

    search = commands.add_parser("search", help="Run validation-only model search")
    search.add_argument("--task", action="append", help="Task name; repeat as needed")
    search.add_argument("--all", action="store_true", help="Search all 22 tasks")
    search.add_argument("--seed", type=int, default=1)
    search.add_argument(
        "--rounds", type=int, default=2, help="Sequential plan/evaluate/refine rounds"
    )
    search.add_argument(
        "--candidates-per-round",
        type=int,
        default=6,
        help=(
            "Maximum primary-seed configurations proposed per round; formal "
            "benchmark commands are checked against max_actions_per_task before training"
        ),
    )
    search.add_argument(
        "--validation-seeds",
        default=None,
        help="Comma-separated validation seeds for robust finalist selection",
    )
    search.add_argument(
        "--robust-top-k",
        type=int,
        default=6,
        help="Candidates promoted to multi-seed validation",
    )
    search.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable validation-prediction blending",
    )
    search.add_argument("--data-dir", type=Path, default=Path("data"))
    search.add_argument("--run-dir", type=Path, default=Path("runs/v1"))
    search.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("cache/features"),
        help="Shared label-free molecular feature cache across policy arms",
    )
    search.add_argument(
        "--persist-discovery-models",
        action="store_true",
        help=(
            "Persist fitted discovery models for debugging; disabled by default "
            "because certification retrains the frozen recipe"
        ),
    )
    search.add_argument(
        "--brain",
        choices=(
            "rules",
            "openai",
            "hybrid",
            "agent",
            "random-control",
            "fixed-rules-control",
            "automl-control",
            "expanded-automl-control",
            "strong-fixed-control",
            "agent-no-memory",
            "agent-no-hardness",
            "agent-no-data",
            "committee",
        ),
        default="rules",
    )
    search.add_argument("--policy-seed", type=int, default=20260819)
    search.add_argument(
        "--benchmark-protocol",
        type=Path,
        default=None,
        help="Cryptographically bind the search to a preregistered protocol",
    )
    search.add_argument(
        "--api-config",
        type=Path,
        default=Path.home() / ".config" / "admet-agent" / "openai.env",
    )
    search.add_argument("--model", default="gpt-5.6-sol")
    search.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    search.add_argument("--workers", type=int, default=8)
    search.add_argument(
        "--selection-policy",
        choices=("score", "lcb", "cost-aware-lcb"),
        default="score",
        help="Validation finalist policy; score is the evidence-backed effect-first default",
    )
    search.add_argument(
        "--uncertainty-penalty",
        type=float,
        default=0.5,
        help="Validation standard-deviation penalty for LCB policies",
    )
    search.add_argument(
        "--cost-penalty",
        type=float,
        default=0.005,
        help="Normalized log-compute penalty for cost-aware LCB",
    )
    search.add_argument(
        "--effect-priority",
        type=float,
        default=0.80,
        help="Research-action utility weight on expected effect; range 0.5-1.0",
    )

    final = commands.add_parser(
        "finalize", help="Run frozen finalists on the TDC test protocol"
    )
    final.add_argument("--task", action="append", help="Task name; repeat as needed")
    final.add_argument("--all", action="store_true", help="Finalize all 22 tasks")
    final.add_argument("--seeds", default="1,2,3,4,5")
    final.add_argument(
        "--finalists",
        type=Path,
        required=True,
        help=(
            "JSON mapping tasks to frozen experiment, ensemble, or recipe specs "
            "(model strings remain supported)"
        ),
    )
    final.add_argument("--data-dir", type=Path, default=Path("data"))
    final.add_argument("--run-dir", type=Path, default=Path("runs/v1"))
    final.add_argument("--feature-cache-dir", type=Path, default=Path("cache/features"))
    final.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    final.add_argument("--workers", type=int, default=8)
    final.add_argument(
        "--freeze-audit",
        type=Path,
        default=None,
        help="Required authorization generated by audit-freeze for --all certification",
    )

    validate = commands.add_parser(
        "validate-spec", help="Validate an experiment JSON without running it"
    )
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
        experiment = ExperimentSpec.from_dict(
            json.loads(args.path.read_text(encoding="utf-8"))
        )
        validate_experiment(experiment)
        print("valid")
        return
    if args.command == "benchmark-agent":
        from .agent_evaluation import evaluate_episode_file

        print(json.dumps(evaluate_episode_file(args.episodes, args.output), indent=2))
        return
    if args.command == "init-policy-benchmark":
        from .benchmark_protocol import write_default_protocol

        print(json.dumps(write_default_protocol(args.path), indent=2))
        return
    if args.command == "compare-policies":
        from .benchmark_protocol import PolicyTaskResult, compare_policy_results

        payload = json.loads(args.results.read_text(encoding="utf-8"))
        rows = payload["results"] if isinstance(payload, dict) else payload
        comparison = compare_policy_results(
            [PolicyTaskResult.from_dict(item) for item in rows],
            agent_arm=args.agent_arm,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps(comparison, indent=2))
        return
    if args.command == "collect-policy-results":
        from .policy_results import collect_policy_results

        arm_dirs = {}
        for value in args.arm_run:
            arm, separator, path = value.partition("=")
            if not separator or not arm or not path:
                raise SystemExit("--arm-run must use ARM=RUN_DIR")
            arm_dirs[arm] = Path(path)
        print(
            json.dumps(
                collect_policy_results(args.protocol, arm_dirs, args.output),
                indent=2,
            )
        )
        return
    if args.command == "benchmark-faults":
        from .hidden_fault_benchmark import evaluate_fault_file

        print(json.dumps(evaluate_fault_file(args.challenges, args.output), indent=2))
        return
    if args.command == "run-hidden-fault-suite":
        from .hidden_fault_suite import run_hidden_fault_suite

        print(json.dumps(run_hidden_fault_suite(args.data_dir, args.output), indent=2))
        return
    if args.command == "audit-freeze":
        from .freeze_audit import audit_global_freeze

        audit = audit_global_freeze(
            args.run_dir,
            args.protocol,
            args.output_audit,
            args.output_finalists,
        )
        print(json.dumps(audit, indent=2))
        if not audit["passed"]:
            raise SystemExit("Global freeze audit failed; test remains locked.")
        return
    if args.command == "summarize-diagnosis":
        from .diagnosis_report import write_suite_report

        summary = write_suite_report(
            args.run_dir,
            args.output_json,
            args.output_markdown,
            seed=args.seed,
        )
        if not summary["complete"]:
            raise SystemExit(
                f"Incomplete TDC-22 diagnosis; missing {summary['missing_tasks']}"
            )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "summarize-search":
        from .search_report import write_search_report

        summary = write_search_report(
            args.run_dir,
            args.output_json,
            args.output_markdown,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "curate-external":
        from .data_governance import (
            LocalTableSource,
            MolecularDataCurator,
            SourceManifest,
        )
        from .endpoint_contracts import endpoint_contract

        manifest = SourceManifest(
            **json.loads(args.manifest.read_text(encoding="utf-8"))
        )
        _, packet = MolecularDataCurator().ingest_external(
            LocalTableSource(args.table, manifest),
            endpoint_contract(args.task),
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            output_dir=args.output_dir,
        )
        print(json.dumps(packet, indent=2))
        return

    names = _selection(args)
    from .runner import ADMETAgent

    brain = None
    if args.command == "search" and args.brain in {"openai", "hybrid"}:
        from .openai_brain import OpenAIPlanner

        planner = OpenAIPlanner(
            config_path=args.api_config,
            model=args.model,
            audit_path=args.run_dir / "api_usage.jsonl",
        )
        if args.brain == "hybrid":
            from .brain import HybridBrain

            brain = HybridBrain(planner)
        else:
            brain = planner
    if args.command == "search" and args.brain in {
        "agent",
        "random-control",
        "fixed-rules-control",
        "automl-control",
        "expanded-automl-control",
        "strong-fixed-control",
        "agent-no-memory",
        "agent-no-hardness",
        "agent-no-data",
        "committee",
    }:
        from .policy_brains import benchmark_brain

        arm = {
            "agent": "agent",
            "random-control": "random",
            "fixed-rules-control": "rules",
            "automl-control": "automl",
            "expanded-automl-control": "expanded_automl",
            "strong-fixed-control": "strong_fixed",
            "agent-no-memory": "agent_no_memory",
            "agent-no-hardness": "agent_no_hardness",
            "agent-no-data": "agent_no_data",
            "committee": "committee",
        }[args.brain]
        brain = benchmark_brain(
            arm,
            max_candidates=args.candidates_per_round,
            policy_seed=args.policy_seed,
        )
    benchmark_protocol = None
    if args.command == "search" and args.benchmark_protocol is not None:
        from .benchmark_protocol import load_protocol, validate_search_budget_request

        benchmark_protocol = load_protocol(args.benchmark_protocol)
        selected_arm = getattr(brain, "benchmark_arm", "agent")
        protocol_arms = {item["name"]: item for item in benchmark_protocol["arms"]}
        if not set(names).issubset(set(benchmark_protocol["tasks"])):
            raise SystemExit("Requested tasks are outside the preregistered protocol")
        if selected_arm not in protocol_arms:
            raise SystemExit(f"Protocol does not define benchmark arm {selected_arm}")
        requested_seeds = (
            [int(value) for value in args.validation_seeds.split(",")]
            if args.validation_seeds
            else [args.seed]
        )
        if requested_seeds != benchmark_protocol["discovery_seeds"]:
            raise SystemExit(
                "Validation seeds must exactly match the preregistered discovery seeds"
            )
        if (
            benchmark_protocol["primary_selection"]
            == "effect_first_score_after_multiseed"
            and args.selection_policy != "score"
        ):
            raise SystemExit("Protocol requires --selection-policy score")
        try:
            validate_search_budget_request(
                protocol_arms[selected_arm],
                rounds=args.rounds,
                candidates_per_round=(
                    1 if selected_arm == "strong_fixed" else args.candidates_per_round
                ),
                one_shot=selected_arm == "strong_fixed",
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    agent_kwargs = {
        "brain": brain,
        "device": args.device,
        "n_jobs": args.workers,
        "effect_priority": getattr(args, "effect_priority", 0.80),
        "benchmark_protocol": benchmark_protocol,
        "feature_cache_dir": args.feature_cache_dir,
        "persist_discovery_models": getattr(args, "persist_discovery_models", False),
    }
    if args.command == "search":
        agent_kwargs.update(
            selection_policy=args.selection_policy,
            uncertainty_penalty=args.uncertainty_penalty,
            cost_penalty=args.cost_penalty,
        )
    agent = ADMETAgent(args.data_dir, args.run_dir, **agent_kwargs)
    agent.environment_manifest()
    if args.command == "diagnose":
        diagnoses = {name: agent.diagnose(name, args.seed) for name in names}
        output = (
            {
                "protocol": "discovery_only_no_test_access",
                "diagnosed_tasks": sorted(diagnoses),
                "count": len(diagnoses),
            }
            if args.quiet
            else diagnoses
        )
        print(json.dumps(output, indent=2))
        return
    if args.command == "search":
        validation_seeds = (
            [int(value) for value in args.validation_seeds.split(",")]
            if args.validation_seeds
            else [args.seed]
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
    finalists = {
        resolve_task(name).name: model for name, model in finalists_raw.items()
    }
    if args.all:
        if args.freeze_audit is None:
            raise SystemExit(
                "Full 22-task certification requires --freeze-audit from audit-freeze."
            )
        from .freeze_audit import verify_freeze_authorization

        verify_freeze_authorization(args.freeze_audit, finalists)
    missing = set(names) - set(finalists)
    if missing:
        raise SystemExit(f"Missing frozen finalists for: {sorted(missing)}")
    seeds = [int(value) for value in args.seeds.split(",")]
    result = agent.finalize(names, seeds, finalists)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
