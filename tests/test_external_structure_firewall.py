from pathlib import Path

import pandas as pd
import pytest

from admet_agent.external_structure_firewall import (
    apply_external_structure_firewall,
    load_label_blind_test_structures,
    prepare_structure_firewall_reference,
)


def test_label_blind_loader_returns_only_structures(tmp_path: Path):
    path = tmp_path / "test.csv"
    path.write_text("Drug,Y\nCC,1\nCCC,0\n", encoding="utf-8")
    assert load_label_blind_test_structures(path) == ("CC", "CCC")


def test_exact_and_near_neighbor_exclusions_are_audited():
    external = pd.DataFrame({"Drug": ["CC", "CCC", "c1ccccc1O"], "Y": [1, 0, 1]})
    output, audit = apply_external_structure_firewall(
        external,
        train_smiles=["CC"],
        validation_smiles=["N"],
        test_smiles=["CCC"],
        same_source_test_overlap_limit=1.0,
        test_tanimoto_limit=0.0,
    )
    assert output.empty
    assert audit.exact_train_overlap_rows == 1
    assert audit.exact_test_overlap_rows == 1
    assert audit.near_test_neighbor_rows == 1
    assert audit.test_labels_loaded is False
    assert audit.test_structures_exposed_to_planner is False


def test_dataset_is_wholly_rejected_above_five_percent_test_skeleton_overlap():
    external = pd.DataFrame({"Drug": ["c1ccccc1"], "Y": [1]})
    output, audit = apply_external_structure_firewall(
        external,
        train_smiles=[],
        validation_smiles=[],
        test_smiles=["c1ccccc1", "C1CCCCC1"],
        same_source_test_overlap_limit=0.05,
    )
    assert output.empty
    assert audit.external_test_skeleton_coverage_fraction == 0.5
    assert audit.dataset_rejected is True
    assert audit.rejection_reason == (
        "external_dataset_has_more_than_allowed_test_inchikey_skeleton_overlap"
    )


def test_reusable_reference_is_equivalent_and_input_bound():
    external = pd.DataFrame({"Drug": ["CC", "c1ccncc1"], "Y": [1, 0]})
    inputs = {
        "train_smiles": ["CC"],
        "validation_smiles": ["CCC"],
        "test_smiles": ["c1ccccc1"],
    }
    reference = prepare_structure_firewall_reference(**inputs)
    uncached, uncached_audit = apply_external_structure_firewall(external, **inputs)
    cached, cached_audit = apply_external_structure_firewall(
        external, **inputs, reference=reference
    )
    pd.testing.assert_frame_equal(cached, uncached)
    assert cached_audit == uncached_audit

    with pytest.raises(ValueError, match="does not match"):
        apply_external_structure_firewall(
            external,
            train_smiles=["CO"],
            validation_smiles=inputs["validation_smiles"],
            test_smiles=inputs["test_smiles"],
            reference=reference,
        )


def test_exactly_five_percent_is_allowed_because_rule_says_more_than():
    test = ["C" * length for length in range(1, 21)]
    output, audit = apply_external_structure_firewall(
        pd.DataFrame({"Drug": ["C"], "Y": [0]}),
        train_smiles=[],
        validation_smiles=[],
        test_smiles=test,
        same_source_test_overlap_limit=0.05,
    )
    assert audit.external_test_skeleton_coverage_fraction == 0.05
    assert audit.dataset_rejected is False
    assert output.empty  # the row is still removed by the exact-identity layer
    assert audit.tanimoto_rejection_threshold == 0.9


def test_standard_identity_catches_salt_charge_and_tautomer_variants():
    external = pd.DataFrame({"Drug": ["CC(=O)[O-].[Na+]"], "Y": [1]})
    output, audit = apply_external_structure_firewall(
        external,
        train_smiles=["CC(=O)O"],
        validation_smiles=[],
        test_smiles=["c1ccccc1"],
    )
    assert output.empty
    assert audit.exact_train_overlap_rows == 1


def test_same_source_fraction_counts_test_molecules_by_inchikey_skeleton():
    output, audit = apply_external_structure_firewall(
        pd.DataFrame({"Drug": ["c1ccccc1F"], "Y": [1]}),
        train_smiles=[],
        validation_smiles=[],
        test_smiles=["c1ccccc1F", "CCO"],
        same_source_test_overlap_limit=0.49,
    )
    assert output.empty
    assert audit.test_molecule_count == 2
    assert audit.external_test_skeleton_coverage_count == 1
    assert audit.external_test_skeleton_coverage_fraction == 0.5
