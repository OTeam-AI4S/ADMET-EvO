"""Freeze raw sources, endpoint contracts and firewalls for the v5 public cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tdc.utils import retrieve_label_name_list


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw_root = Path("data/v5_public_cohort_raw")
    raw_files = {
        path.name: _sha256(path) for path in sorted(raw_root.glob("*.tab"))
    }
    tdc_tests = sorted(Path("data/admet_group").glob("*/test.csv"))
    tdc_development = sorted(Path("data/admet_group").glob("*/train_val.csv"))
    if len(tdc_tests) != 22 or len(tdc_development) != 22:
        raise RuntimeError("Expected exactly 22 frozen TDC task files per partition")
    openadmet_blind = Path(
        "external/openadmet/cyp_challenge_data/cyp-challenge-TEST-BLINDED.csv"
    )
    implementation = [
        Path("admet_agent/cohort_governance.py"),
        Path("scripts/prepare_ecvoi_v5_public_cohort.py"),
        Path("tests/test_cohort_governance.py"),
    ]
    endpoints = []
    for label in retrieve_label_name_list("tox21"):
        endpoints.append(
            {
                "endpoint_id": f"tox21::{label}",
                "source_id": "Tox21_US_Federal_Challenge",
                "raw_file": "tox21.tab",
                "smiles_column": "X",
                "target_column": label,
                "task_type": "classification",
                "metric": "mcc",
                "metric_direction": "maximize",
                "assay_system": "Tox21 qualitative in-vitro pathway assay",
                "species": "not_applicable_in_vitro",
                "unit": "binary_activity",
            }
        )
    endpoints.extend(
        [
            {"endpoint_id":"clintox::CT_TOX","source_id":"MoleculeNet_ClinTox","raw_file":"clintox.tab","smiles_column":"Drug","target_column":"Y","task_type":"classification","metric":"mcc","metric_direction":"maximize","assay_system":"FDA-approved versus clinical-trial-toxicity outcome","species":"human","unit":"binary_clinical_toxicity"},
            {"endpoint_id":"freesolv::experimental_hydration_free_energy","source_id":"FreeSolv","raw_file":"hydrationfreeenergy_freesolv.tab","smiles_column":"Drug","target_column":"Y","task_type":"regression","metric":"mae","metric_direction":"minimize","assay_system":"experimental small-molecule hydration free energy in water","species":"not_applicable_physicochemical","unit":"kcal_per_mol"},
            {"endpoint_id":"ncats_pampa::pH7.4_permeability","source_id":"NCATS_PAMPA","raw_file":"pampa_ncats.tab","smiles_column":"Drug","target_column":"Y","task_type":"classification","metric":"mcc","metric_direction":"maximize","assay_system":"parallel artificial membrane permeability assay pH 7.4","species":"not_applicable_in_vitro","unit":"binary_high_vs_low_moderate_permeability"},
            {"endpoint_id":"b3db::bbb_classification","source_id":"B3DB","raw_file":"b3db_classification.tab","smiles_column":"Drug","target_column":"Y","task_type":"classification","metric":"mcc","metric_direction":"maximize","assay_system":"curated blood-brain barrier permeability evidence","species":"mixed_source_curated","unit":"binary_BBB_plus_minus"},
            {"endpoint_id":"b3db::logBB_regression","source_id":"B3DB","raw_file":"b3db_regression.tab","smiles_column":"Drug","target_column":"Y","task_type":"regression","metric":"mae","metric_direction":"minimize","assay_system":"curated blood-brain concentration ratio","species":"mixed_source_curated","unit":"logBB"},
        ]
    )
    payload = {
        "version": "ecvoi-v5-public-cohort-protocol-v1",
        "frozen_at": "2026-08-23",
        "mechanism_protocol_sha256": "94339422007e1ab77e30a5949e5819747e34a1374885a1b505883382ee1e9b15",
        "raw_root": str(raw_root.resolve()),
        "raw_files_sha256": raw_files,
        "sources": {
            "Tox21_US_Federal_Challenge": {"admission":"modeling_ready_development","license":"CC_BY_4.0_as_distributed_by_TDC; US federal source publicly available","origin":"https://tripod.nih.gov/tox21/pubdata","citation":"Tox21 Data Challenge","redistribution":False},
            "MoleculeNet_ClinTox": {"admission":"modeling_ready_development","license":"MIT_original_MoleculeNet_dataset","origin":"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz","citation":"Wu et al., MoleculeNet, Chemical Science 2018","redistribution":False},
            "FreeSolv": {"admission":"modeling_ready_development","license":"CC_BY_NC_SA_4.0_as_listed_by_TDC (more restrictive record)","origin":"https://github.com/MobleyLab/FreeSolv","citation":"Mobley and Guthrie, JCAMD 2014","redistribution":False},
            "NCATS_PAMPA": {"admission":"modeling_ready_development","license":"CC_BY_4.0_as_distributed_by_TDC; public PubChem AID 1508612","origin":"https://pubchem.ncbi.nlm.nih.gov/bioassay/1508612","citation":"Siramshetty et al., SLAS Discovery 2021","redistribution":False},
            "B3DB": {"admission":"modeling_ready_development","license":"CC0_1.0","origin":"https://github.com/theochem/B3DB","citation":"Meng et al., Scientific Data 2021","redistribution":False},
            "Skin_Reaction": {"admission":"quarantine_license_ambiguous","raw_file":"skin_reaction.tab"},
            "Carcinogens_Lagunin": {"admission":"quarantine_license_ambiguous","raw_file":"carcinogens_lagunin.tab"},
        },
        "endpoints": endpoints,
        "quarantined_raw_files": ["skin_reaction.tab", "carcinogens_lagunin.tab"],
        "standardization": "cleanup_fragment_parent_uncharge_canonical_tautomer_inchikey",
        "deduplication": "endpoint_identity_collapse; conflicting_binary_identity_removed; regression_replicates_mean",
        "firewall": {
            "development_exact_identity_sources": {str(path): _sha256(path) for path in tdc_development},
            "protected_tdc_test_structure_sources": {str(path): _sha256(path) for path in tdc_tests},
            "protected_openadmet_blind_structure_source": {"path":str(openadmet_blind),"sha256":_sha256(openadmet_blind),"contains_labels":False},
            "near_neighbor_fingerprint": "Morgan_radius_2_2048",
            "near_neighbor_tanimoto_threshold": 0.80,
            "test_structure_use": "record_removal_only; counts_cannot_change_source_or_endpoint_admission",
            "test_label_columns_loaded": False,
        },
        "split": {"identity_locked":True,"chemistry_group_locked":True,"outer_folds":5,"seed":20260823,"certification_labels_sealed":False},
        "implementation_sha256": {str(path):_sha256(path) for path in implementation},
        "gate": {"minimum_endpoint_source_units":8,"requires_classification":True,"requires_regression":True,"minimum_source_families":4,"identity_or_group_overlap_between_folds":0,"test_labels_loaded":False},
        "blind_test_opened": False,
        "leaderboard_feedback_used": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v5_public_cohort_protocol_v1.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
