"""Preregister fresh source acquisition before loading candidate records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    candidates = [
        {"loader": "Tox", "dataset": "ToxCast", "source_id": "epa_toxcast_2016", "max_endpoints": 4, "license_status": "TDC_explicit_CC_BY_4.0"},
        {"loader": "Tox", "dataset": "hERG_Karim", "source_id": "karim_cardiotox_2021", "max_endpoints": 1, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "Tox", "dataset": "hERG_Central", "source_id": "du_hergcentral_2011", "max_endpoints": 3, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "HTS", "dataset": "HIV", "source_id": "nci_dtp_hiv", "max_endpoints": 1, "license_status": "TDC_explicit_CC_BY_4.0"},
        {"loader": "HTS", "dataset": "SARSCoV2_Vitro_Touret", "source_id": "touret_sarscov2_2020", "max_endpoints": 1, "license_status": "TDC_explicit_CC_BY_4.0"},
        {"loader": "HTS", "dataset": "SARSCoV2_3CLPro_Diamond", "source_id": "diamond_3clpro_2020", "max_endpoints": 1, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "HTS", "dataset": "orexin1_receptor_butkiewicz", "source_id": "butkiewicz_pubchem_2013", "max_endpoints": 1, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "HTS", "dataset": "m1_muscarinic_receptor_agonists_butkiewicz", "source_id": "butkiewicz_pubchem_2013", "max_endpoints": 1, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "HTS", "dataset": "potassium_ion_channel_kir2.1_butkiewicz", "source_id": "butkiewicz_pubchem_2013", "max_endpoints": 1, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "HTS", "dataset": "serine_threonine_kinase_33_butkiewicz", "source_id": "butkiewicz_pubchem_2013", "max_endpoints": 1, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "QM", "dataset": "QM7b", "source_id": "montavon_qm7b_2013", "max_endpoints": 2, "license_status": "TDC_CC_BY_4.0_upstream_unspecified"},
        {"loader": "QM", "dataset": "QM8", "source_id": "ramakrishnan_qm8_2015", "max_endpoints": 2, "license_status": "TDC_explicit_CC_BY_4.0"},
        {"loader": "QM", "dataset": "QM9", "source_id": "ramakrishnan_qm9_2014", "max_endpoints": 2, "license_status": "TDC_explicit_CC_BY_4.0"},
    ]
    payload = {
        "version": "ecvoi-v6-fresh-source-registry-v1",
        "frozen_at": "2026-08-23",
        "candidate_records_loaded_before_freeze": False,
        "candidate_action_outcomes_available_before_freeze": False,
        "registry_role": "source acquisition and deterministic admission only; not a model-result protocol",
        "official_documentation": {
            "Tox": "https://tdcommons.ai/single_pred_tasks/tox/",
            "HTS": "https://tdcommons.ai/single_pred_tasks/hts/",
            "QM": "https://tdcommons.ai/single_pred_tasks/qm/",
            "TDC_code": "https://github.com/mims-harvard/TDC",
            "accessed": "2026-08-23",
        },
        "license_policy": {
            "TDC_explicit_CC_BY_4.0": "admit for research modeling and derived-statistic release with attribution",
            "TDC_CC_BY_4.0_upstream_unspecified": "admit for local research modeling; do not redistribute raw rows; disclose upstream ambiguity",
            "missing_TDC_CC_BY_declaration": "quarantine",
        },
        "candidates": candidates,
        "source_grouping": "all endpoints from the same primary study/database family share one source_id",
        "admission": {
            "valid_rdkit_smiles_fraction": ">=0.95",
            "minimum_post_firewall_identities": 400,
            "classification_minority_identities": ">=40",
            "classification_each_outer_fold_per_class": ">=5",
            "regression_finite_fraction": ">=0.95",
            "regression_nonzero_variance": True,
            "endpoint_selection": "canonical label name ascending after admission; at most max_endpoints per source; no model scores",
            "minimum_admitted_sources": 9,
            "minimum_admitted_endpoints": 18,
            "require_classification_and_regression": True,
        },
        "compute_cap": {
            "classification": "retain every minority identity and the 30000 lowest SHA256-ranked majority identities",
            "regression": "retain the 30000 lowest SHA256-ranked identities",
            "hash_salt": "ecvoi-v6-fresh-cohort-cap-v1",
            "labels_used_only_for_classification_minority_retention": True,
            "same_capped_rows_for_every_action": True,
        },
        "split": {
            "outer_folds": 5,
            "seed": 20260823,
            "identity_and_scaffold_component_locked": True,
            "source_id_never_split_for_policy_meta-evaluation": True,
        },
        "firewall": {
            "remove_exact_identity_against_v5_development": True,
            "remove_exact_identity_against_all_TDC22_development_and_protected_test": True,
            "remove_exact_identity_against_OpenADMET_blind_structures": True,
            "remove_near_neighbor_against_protected_TDC22_test_and_OpenADMET_blind": True,
            "near_neighbor_tanimoto_threshold": 0.80,
            "protected_labels_loaded": False,
        },
        "explicit_exclusions": {
            "canonical_TDC_ADMET22_datasets": "all excluded as candidate endpoints",
            "v5_public_sources": ["tox21", "clintox", "b3db", "ncats_pampa", "freesolv"],
            "OpenADMET_CYP": "blind structures only for firewall; labels unavailable",
        },
        "freeze_sequence": [
            "acquire candidates and content-hash raw cache",
            "run deterministic chemistry/license/firewall admission",
            "freeze final cohort manifest and action-screen protocol",
            "only then compute action outcomes",
        ],
        "test_or_leaderboard_feedback_used": False,
        "tdc_19_locked": True,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v6_fresh_source_registry_v1.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
