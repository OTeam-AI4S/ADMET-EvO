"""Freeze the single-ticket ChEMBL hERG source-heldout protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path) -> dict:
    sources = [
        root / "admet_agent/external_herg.py",
        root / "scripts/acquire_chembl_herg.py",
        root / "scripts/run_chembl_herg_source_holdout.py",
        root / "reports/test_firewall_incident_2026-08-20.json",
    ]
    return {
        "protocol": "tdc22-herg-chembl-source-heldout-v2",
        "objective": "Test whether an agent-curated real external hERG source improves document-heldout external-domain discrimination without touching TDC test or mutating the frozen TDC recipe.",
        "endpoint": {
            "tdc_task": "hERG",
            "tdc_semantics": "heterogeneous binary hERG blocker label; exact threshold not asserted",
            "external_target": "CHEMBL240 human KCNH2/hERG",
            "external_label": "IC50 < 10000 nM is blocker",
            "alignment": "related auxiliary liability endpoint, not asserted identical",
        },
        "source": {
            "api": "https://www.ebi.ac.uk/chembl/api/data/activity.json",
            "server_filters": {"target_chembl_id": "CHEMBL240", "standard_type": "IC50", "standard_units": "nM"},
            "license": "CC BY-SA 3.0 Unported",
            "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "attribution": "ChEMBL current data API, accessed 2026-08-20",
            "acquisition_correction": "v1 failed closed on HTTP 500; v2 requests only required fields with 250-row explicit-offset pages",
        },
        "local_filters": {
            "target_and_species_exact": True,
            "standard_relation": "=",
            "standard_flag": 1,
            "potential_duplicate": 0,
            "data_validity_comment": None,
            "assay_type_in": ["F", "T"],
            "finite_positive_ic50": True,
            "valid_largest_fragment_canonical_smiles": True,
        },
        "split": {
            "unit": "document_chembl_id",
            "heldout_fraction": 0.2,
            "assignment": "sha256 frozen salt modulo 5",
            "drop_molecules_crossing_partitions": True,
            "drop_all_TDC_train_overlap": True,
            "drop_within_partition_label_conflicts": True,
        },
        "admission_gates": {
            "minimum_external_train_molecules": 500,
            "minimum_source_heldout_molecules": 100,
            "both_positive_rates_between": [0.05, 0.95],
            "molecule_and_document_overlap": 0,
        },
        "matched_evaluation": {
            "control": "TDC train_val only",
            "treatment": "same TDC train_val plus admitted ChEMBL source-train",
            "only_changed_axis": "external training data",
            "representation": ["morgan", "rdkit2d"],
            "model": "ExtraTreesClassifier frozen hyperparameters",
            "seeds": [1, 2, 3, 4, 5],
            "primary_metric": "AUROC on ChEMBL source-heldout",
            "secondary_metrics": ["AUPRC", "balanced_accuracy"],
            "gain_gate": "paired bootstrap 95% CI lower bound > 0 and mean AUROC delta >= 0.01",
        },
        "firewall": {
            "tdc_input": "data/admet_group/herg/train_val.csv only",
            "known_incident": "tdc-test-access-2026-08-20-001",
            "test_accessed_in_session": True,
            "test_feedback_used": False,
            "frozen_recipe_mutation_allowed": False,
        },
        "source_hashes": {str(path.relative_to(root)): _sha(path) for path in sources},
        "status": "frozen_before_acquisition",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = build(root)
    output = root / "runs/chembl-herg-source-heldout-v2/protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
