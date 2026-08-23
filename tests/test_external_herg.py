from admet_agent.external_herg import admission_decision, curate_chembl_herg
from scripts.run_chembl_herg_source_holdout import paired_stratified_bootstrap_delta

import numpy as np


def _record(activity_id, molecule, document, value, **overrides):
    row = {
        "activity_id": activity_id,
        "target_chembl_id": "CHEMBL240",
        "target_organism": "Homo sapiens",
        "standard_type": "IC50",
        "standard_units": "nM",
        "standard_relation": "=",
        "standard_flag": 1,
        "potential_duplicate": 0,
        "data_validity_comment": None,
        "assay_type": "F",
        "canonical_smiles": molecule,
        "standard_value": str(value),
        "document_chembl_id": document,
        "assay_chembl_id": f"A{activity_id}",
        "molecule_chembl_id": f"M{activity_id}",
    }
    row.update(overrides)
    return row


def test_curator_filters_non_exact_and_tdc_overlap_without_leakage():
    records = [
        _record(1, "CC", "D1", 100),
        _record(2, "CCC", "D2", 20000),
        _record(3, "CCCC", "D3", 100, standard_relation=">"),
        _record(4, "CCO", "D4", 100),
    ]
    train, heldout, audit = curate_chembl_herg(records, tdc_train_smiles={"CCO"})
    assert set(train["Drug"]).isdisjoint(heldout["Drug"])
    assert "CCO" not in set(train["Drug"]).union(heldout["Drug"])
    assert audit["filter_counts"]["rejected_censored_or_approximate"] == 1
    assert audit["filter_counts"]["rejected_tdc_train_overlap_molecules"] == 1


def test_admission_gate_is_explicit():
    decision = admission_decision(
        {
            "external_train_rows": 600,
            "source_heldout_rows": 120,
            "external_train_positive_rate": 0.5,
            "source_heldout_positive_rate": 0.4,
            "molecule_overlap_between_partitions": 0,
            "document_overlap_between_partitions": 0,
        }
    )
    assert decision["admitted_for_matched_external_evaluation"] is True
    assert decision["tdc_recipe_mutation_allowed"] is False


def test_paired_bootstrap_detects_a_clear_treatment_gain():
    y = np.array([0] * 20 + [1] * 20)
    control = np.full(40, 0.5)
    treatment = np.r_[np.linspace(0.0, 0.4, 20), np.linspace(0.6, 1.0, 20)]
    result = paired_stratified_bootstrap_delta(y, control, treatment, samples=200)
    assert result["lower_95"] > 0
