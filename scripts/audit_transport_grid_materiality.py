"""Audit sentinel hardness and data-action materiality without fitting models."""

from __future__ import annotations

import json
from pathlib import Path

from tdc.benchmark_group import admet_group

from admet_agent.data_governance import MolecularDataCurator
from admet_agent.source_snapshot import production_source_manifest
from admet_agent.tasks import resolve_task
from admet_agent.transport_protocol import load_transport_protocol


def _columns(frame):
    target = "Y" if "Y" in frame else frame.columns[-1]
    smiles = "Drug" if "Drug" in frame else frame.columns[0]
    return smiles, target


def main() -> None:
    root = Path(__file__).parents[1]
    protocol = load_transport_protocol(root / "runs/v5-transport-grid/protocol.json")
    if protocol["source_sha256"] != production_source_manifest()["source_sha256"]:
        raise RuntimeError(
            "Transport protocol source hash no longer matches production"
        )
    group = admet_group(path=str(root / "data"))
    rows = []
    for endpoint in protocol["sentinel_endpoints"]:
        task = resolve_task(endpoint)
        for seed in protocol["validation_seeds"]:
            train, valid = group.get_train_valid_split(
                benchmark=endpoint, split_type="default", seed=seed
            )
            smiles_col, target_col = _columns(train)
            curated, _audit = MolecularDataCurator(smiles_col, target_col).curate(
                train,
                task_type=task.task_type,
                policy="quality_filtered",
                dry_run=False,
            )
            changed_rows = len(train) - len(curated)
            rows.append(
                {
                    "task": endpoint,
                    "category": task.category,
                    "task_type": task.task_type,
                    "seed": seed,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "data_policy": "quality_filtered",
                    "changed_training_rows": changed_rows,
                    "data_action_material": changed_rows > 0,
                    "test_loaded": False,
                }
            )
    material = sum(row["data_action_material"] for row in rows)
    endpoints_material = len(
        {row["task"] for row in rows if row["data_action_material"]}
    )
    report = {
        "version": "transport-grid-materiality-audit-v1",
        "protocol_sha256": protocol["sha256"],
        "source_sha256": protocol["source_sha256"],
        "endpoint_count": len(protocol["sentinel_endpoints"]),
        "split_count": len(rows),
        "material_data_action_splits": material,
        "material_data_action_endpoints": endpoints_material,
        "all_endpoints_have_material_data_action": endpoints_material
        == len(protocol["sentinel_endpoints"]),
        "test_loaded": False,
        "rows": rows,
    }
    (root / "reports/transport_grid_materiality_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Prospective transport-grid materiality audit",
        "",
        f"- Protocol SHA-256: `{protocol['sha256']}`",
        f"- Production source SHA-256: `{protocol['source_sha256']}`",
        f"- Sentinel endpoints: {report['endpoint_count']}",
        f"- Validation splits audited: {report['split_count']}",
        (
            "- Data action changes training membership: "
            f"{material}/{report['split_count']} seed-specific splits"
        ),
        (
            "- Endpoints with at least one material data action: "
            f"{endpoints_material}/{report['endpoint_count']}"
        ),
        "- Model fits: 0",
        "- Test loaded: false",
        "",
        "This is a preflight audit only. It verifies that the data intervention is not",
        "a universal no-op before authorizing the 150-fit complete grid. It does not",
        "estimate action effects and cannot make a cross-task memory claim.",
        "",
    ]
    (root / "reports/transport_grid_materiality_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
