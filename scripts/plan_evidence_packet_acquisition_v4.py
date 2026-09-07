#!/usr/bin/env python3
"""Create endpoint-specific FDA/literature acquisition tickets for weak tasks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.endpoint_contracts import endpoint_contract  # noqa: E402
from admet_agent.evidence_packet_loop import weakest_tasks  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports/evidence_packet_data_axis_v4/acquisition_plan.json"

SOURCE_GUARDS = {
    "AMES": {
        "benchmark_origin": "Hansen/TDC Ames compilation",
        "forbidden_reaggregations": [
            "MathWorks Chemistry-Deep-Learning AMES_Alberga/Hansen derivatives",
            "any renamed copy or split of the TDC/Hansen Ames benchmark",
        ],
        "candidate_leads": [],
    },
    "CYP3A4_Veith": {
        "benchmark_origin": "Veith et al. 2009 qHTS, PubChem AID 1851, DOI 10.1038/nbt.1581",
        "forbidden_reaggregations": [
            "PubChem AID 1851",
            "Veith-derived CYP3A4 classification tables",
        ],
        "candidate_leads": [
            {
                "source": "Gonzalez et al. 2021, DOI 10.1124/dmd.120.000320",
                "records": "NCATS/PubChem AID 1645841",
                "status": "lineage_review_required_before_packet_materialization",
                "reason": "distinct 2021 assay/cohort but shares NCATS lineage with the 2009 benchmark source",
            }
        ],
    },
    "CYP2C9_Veith": {
        "benchmark_origin": "Veith et al. 2009 qHTS, PubChem AID 1851, DOI 10.1038/nbt.1581",
        "forbidden_reaggregations": [
            "PubChem AID 1851",
            "Veith-derived CYP2C9 classification tables",
        ],
        "candidate_leads": [
            {
                "source": "Gonzalez et al. 2021, DOI 10.1124/dmd.120.000320",
                "records": "NCATS/PubChem AID 1645842",
                "status": "lineage_review_required_before_packet_materialization",
                "reason": "distinct 2021 assay/cohort but shares NCATS lineage with the 2009 benchmark source",
            }
        ],
    },
}


def load_discovery_routing_scores(directory: Path) -> dict[str, float]:
    scores = {}
    paths = sorted(directory.glob("*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"incomplete discovery artifact: {path}")
        scores[payload["task"]] = float(
            payload["routing"]["development_summary"][
                "mean_normalized_improvement"
            ]
        )
    if len(paths) != 22 or len(scores) != 22:
        raise ValueError("routing score source must contain exactly 22 completed tasks")
    return scores


def build_plan(
    scores: dict[str, float], *, count: int, score_source: str | None = None
) -> dict:
    targets = weakest_tasks(scores, count=count)
    tickets = []
    for rank, task in enumerate(targets, start=1):
        contract = endpoint_contract(task)
        tickets.append(
            {
                "priority": rank,
                "task": task,
                "current_normalized_improvement": float(scores[task]),
                "endpoint_contract": contract.to_dict(),
                "source_guard": SOURCE_GUARDS.get(
                    task,
                    {
                        "benchmark_origin": contract.source_family,
                        "forbidden_reaggregations": [
                            "the task's own source or any renamed/reaggregated copy"
                        ],
                        "candidate_leads": [],
                    },
                ),
                "permitted_sources": [
                    "FDA label/review/guidance with molecule-level endpoint evidence",
                    "independent peer-reviewed endpoint-matched assay",
                ],
                "search_queries": [
                    f"site:fda.gov {contract.property_name} drug assay table",
                    f'"{contract.property_name}" assay supplementary data molecules',
                ],
                "packet_unit": "one document and one endpoint/assay/label transform",
                "minimum_rows": 1,
                "maximum_rows": 5000,
                "required_review": [
                    "source independence from TDC source family",
                    "direct endpoint and species/system match",
                    "molecule-level record locator",
                    "exact label mapping, direction, transform, and unit",
                ],
                "automatic_download_or_training_allowed": False,
            }
        )
    return {
        "protocol": "admet-evidence-packet-acquisition-plan-v4.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": "bottom normalized-improvement tasks",
        "score_source": score_source,
        "target_count": count,
        "target_tasks": list(targets),
        "tickets": tickets,
        "legacy_bulk_pool_used": False,
        "test_labels_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores", type=Path)
    source.add_argument("--discovery-dir", type=Path)
    parser.add_argument("--count", type=int, choices=(2, 3), default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.scores is not None:
        payload = json.loads(args.scores.read_text(encoding="utf-8"))
        scores = payload.get("normalized_improvement_by_task", payload)
        score_source = str(args.scores.resolve())
    else:
        scores = load_discovery_routing_scores(args.discovery_dir)
        score_source = str(args.discovery_dir.resolve())
    plan = build_plan(scores, count=args.count, score_source=score_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "targets": plan["target_tasks"]}))


if __name__ == "__main__":
    main()
