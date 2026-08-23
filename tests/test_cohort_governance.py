from __future__ import annotations

import pandas as pd

from admet_agent.cohort_governance import (
    assign_identity_locked_folds,
    collapse_endpoint_records,
    protected_structure_firewall,
    standardize_records,
)
from admet_agent.cohort_governance_v2 import standardize_records as standardize_records_v2
from admet_agent.cohort_governance_v3 import protected_structure_firewall as protected_structure_firewall_v3
from admet_agent.cohort_governance_v4 import (
    assign_identity_locked_folds as assign_component_locked_folds,
)


def test_standardization_and_conflicting_binary_labels_fail_closed():
    frame = pd.DataFrame(
        {"Drug":["CCO","OCC","CCC","bad"],"target":[0,1,1,0]}
    )
    standardized, invalid = standardize_records(frame, smiles_column="Drug")
    collapsed, audit = collapse_endpoint_records(
        standardized, task_type="classification"
    )
    assert invalid == 1
    assert audit["conflicting_identities_removed"] == 1
    assert len(collapsed) == 1


def test_protected_firewall_removes_development_exact_test_and_neighbor():
    frame = pd.DataFrame({"Drug":["CCO","c1ccccc1","CCCCCCCC","N#N"]})
    standardized, _ = standardize_records(frame, smiles_column="Drug")
    allowed, audit = protected_structure_firewall(
        standardized,
        development_smiles=["OCC"],
        protected_test_smiles=["c1ccccc1","CCCCCCC"],
        tanimoto_threshold=0.5,
    )
    assert audit["exact_development_overlap_removed"] == 1
    assert audit["exact_test_overlap_removed"] == 1
    assert audit["near_test_neighbor_removed"] >= 1
    assert len(allowed) == 1
    assert audit["test_labels_loaded"] is False


def test_fold_assignment_locks_identity_and_chemistry_group():
    frame = pd.DataFrame(
        {
            "inchikey":["a","b","c","d","e","f"],
            "chemistry_group":["g1","g1","g2","g3","g4","g5"],
        }
    )
    assignment = assign_identity_locked_folds(frame, folds=3, seed=4)
    assert assignment["a"] == assignment["b"]
    assert set(assignment.values()) == {0,1,2}


def test_v2_standardization_retains_valid_and_counts_invalid():
    output, invalid = standardize_records_v2(
        pd.DataFrame({"Drug":["CCO","not-a-smiles"],"Y":[1,0]}),
        smiles_column="Drug",
    )
    assert len(output) == 1
    assert invalid == 1
    assert output.iloc[0]["inchikey"]


def test_v3_firewall_deduplicates_raw_protected_inputs():
    candidates, _ = standardize_records_v2(
        pd.DataFrame({"Drug":["CCO","N#N"],"Y":[1,0]}),
        smiles_column="Drug",
    )
    allowed, audit = protected_structure_firewall_v3(
        candidates,
        development_smiles=[],
        protected_test_smiles=["CCO","CCO","CCC"],
        tanimoto_threshold=0.9,
    )
    assert audit["raw_structure_deduplication_before_standardization"] is True
    assert audit["exact_test_overlap_removed"] == 1
    assert len(allowed) == 1


def test_component_locking_handles_identity_with_multiple_groups():
    rows = pd.DataFrame(
        {
            "inchikey": ["id-a", "id-a", "id-b", "id-c"],
            "chemistry_group": ["g1", "g2", "g2", "g3"],
        }
    )
    first = assign_component_locked_folds(rows, folds=2, seed=17)
    second = assign_component_locked_folds(rows, folds=2, seed=17)
    assert first == second
    assert first["id-a"] == first["id-b"]
