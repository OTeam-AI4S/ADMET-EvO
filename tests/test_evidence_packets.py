import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from admet_agent.evidence_packets import (
    SCHEMA_VERSION,
    TEST_TANIMOTO_LIMIT,
    assemble_evidence_training_data,
    prepare_evidence_packet,
    validate_evidence_packet,
)


def _write_packet(
    root: Path,
    *,
    packet_id: str = "paper-123-hia-assay-a",
    rows: list[dict] | None = None,
    source_kind: str = "independent_peer_reviewed_assay",
    status: str = "modeling_ready",
) -> Path:
    root.mkdir(parents=True)
    records = root / "records.csv"
    pd.DataFrame(
        rows or [{"Drug": "CCO", "Y": 1, "source_record_id": "table2-row1"}]
    ).to_csv(records, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "task": "HIA_Hou",
        "task_type": "classification",
        "source": {
            "kind": source_kind,
            "name": "Independent HIA study",
            "title": "Measured human intestinal absorption",
            "citation_url": "https://doi.org/10.1000/example-hia",
            "citation_type": "doi",
            "stable_identifier": "doi:10.1000/example-hia",
            "license_or_access_terms": "facts extracted with citation",
            "independence_status": "verified_independent",
            "independence_evidence": "not listed in the TDC source bibliography",
        },
        "assay": {
            "endpoint_definition": "fraction absorbed in humans",
            "assay_type": "clinical absorption measurement",
            "organism": "Homo sapiens",
            "biological_system": "human clinical pharmacokinetics",
            "source_record_locator": "Table 2",
        },
        "label": {
            "source_value_definition": "reported fraction absorbed",
            "mapping_rule": "Y=1 when reported fraction absorbed >= 0.30; otherwise 0",
            "direction": "higher means absorbed",
            "transform": "threshold at 0.30",
            "unit": "fraction",
        },
        "review": {
            "status": status,
            "reviewer": "data-steward",
            "reviewed_at": "2026-08-30T00:00:00Z",
            "endpoint_match": "direct",
            "notes": "assay and label mapping independently checked",
            "native_priority_external_weight": 0.75,
        },
        "records": {
            "path": "records.csv",
            "sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
            "row_count": len(pd.read_csv(records)),
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_precise_one_row_packet_is_allowed_without_minimum_size(tmp_path: Path):
    packet = validate_evidence_packet(_write_packet(tmp_path / "packet"))
    assert len(packet.records) == 1
    assert packet.packet_id == "paper-123-hia-assay-a"


def test_legacy_bulk_pool_cannot_be_relabelled_as_packet(tmp_path: Path):
    path = _write_packet(tmp_path / "packet", source_kind="chembl_endpoint_bulk_pool")
    with pytest.raises(ValueError, match="allowed FDA/independent-literature"):
        validate_evidence_packet(path)


def test_missing_citation_review_or_mutated_records_fail_closed(tmp_path: Path):
    path = _write_packet(tmp_path / "packet")
    manifest = json.loads(path.read_text())
    manifest["source"]["citation_url"] = ""
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="citation_url"):
        validate_evidence_packet(path)

    path = _write_packet(tmp_path / "pending", status="semantic_review_required")
    with pytest.raises(ValueError, match="modeling-ready review"):
        validate_evidence_packet(path)

    path = _write_packet(tmp_path / "mutated")
    (path.parent / "records.csv").write_text(
        "Drug,Y,source_record_id\nCCN,0,changed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_evidence_packet(path)


def test_packet_firewall_uses_point_nine_and_removes_near_test_analog(
    tmp_path: Path,
):
    packet = validate_evidence_packet(
        _write_packet(
            tmp_path / "packet",
            rows=[
                {"Drug": "CCCCCCCC", "Y": 1, "source_record_id": "near"},
                {"Drug": "c1ccncc1", "Y": 0, "source_record_id": "safe"},
            ],
        )
    )
    prepared = prepare_evidence_packet(
        packet,
        train_smiles=["CCO"],
        validation_smiles=["CCN"],
        test_smiles=["CCCCCCC", "C1CCCCC1"],
    )
    assert TEST_TANIMOTO_LIMIT == 0.90
    assert prepared.audit["structure_firewall"]["tanimoto_rejection_threshold"] == 0.90
    assert prepared.audit["structure_firewall"]["near_test_neighbor_rows"] == 1
    assert prepared.records["source_record_id"].tolist() == ["safe"]
    assert prepared.audit["test_labels_loaded"] is False
    assert prepared.records["evidence_base_weight"].unique().tolist() == [0.75]


def test_same_source_check_precedes_classification_conflict_collapse(tmp_path: Path):
    packet = validate_evidence_packet(
        _write_packet(
            tmp_path / "packet-conflict",
            rows=[
                {"Drug": "c1ccccc1F", "Y": 0, "source_record_id": "conflict-a"},
                {"Drug": "c1ccccc1F", "Y": 1, "source_record_id": "conflict-b"},
            ],
        )
    )
    prepared = prepare_evidence_packet(
        packet,
        train_smiles=[],
        validation_smiles=[],
        test_smiles=[
            "c1ccccc1F",
            "CCO",
            "CCN",
            "CCC",
            "C1CCCCC1",
        ],
    )
    assert prepared.admitted is False
    assert prepared.audit["structure_firewall"]["dataset_rejected"] is True


def test_more_than_five_percent_test_skeleton_overlap_rejects_whole_packet(
    tmp_path: Path,
):
    packet = validate_evidence_packet(
        _write_packet(
            tmp_path / "packet",
            rows=[{"Drug": "c1ccccc1F", "Y": 1, "source_record_id": "overlap"}],
        )
    )
    prepared = prepare_evidence_packet(
        packet,
        train_smiles=[],
        validation_smiles=[],
        test_smiles=["c1ccccc1F", "CCO", "CCN", "CCC", "C1CCCCC1"],
    )
    assert prepared.admitted is False
    assert prepared.records.empty
    assert prepared.audit["structure_firewall"]["dataset_rejected"] is True


def test_equal_weight_primary_and_native_priority_control(tmp_path: Path):
    packet = validate_evidence_packet(_write_packet(tmp_path / "packet"))
    prepared = prepare_evidence_packet(
        packet,
        train_smiles=["CC"],
        validation_smiles=["CCC"],
        test_smiles=["c1ccccc1"],
    )
    native = pd.DataFrame({"Drug": ["CC", "CCC"], "Y": [0, 1]})
    equal, equal_audit = assemble_evidence_training_data(
        native, [prepared], weight_mode="paper_equal"
    )
    assert equal["sample_weight"].tolist() == [1.0, 1.0, 1.0]
    assert equal_audit["coreset_selection_used"] is False

    weighted, weighted_audit = assemble_evidence_training_data(
        native, [prepared], weight_mode="native_priority"
    )
    assert weighted["sample_weight"].tolist() == [1.0, 1.0, 0.75]
    assert weighted_audit["native_strictly_higher_than_external"] is True
    assert weighted_audit["legacy_bulk_pool_used"] is False


def test_unfingerprintable_standardized_structure_is_invalid_not_a_crash(
    tmp_path: Path,
):
    packet = validate_evidence_packet(
        _write_packet(
            tmp_path / "packet-hypervalent",
            rows=[
                {
                    "Drug": "CCCCNC(C)(C)[P+](=O)O",
                    "Y": 1,
                    "source_record_id": "PubChem:AID1645841:SID225144345",
                },
                {"Drug": "c1ccncc1", "Y": 0, "source_record_id": "safe"},
            ],
        )
    )
    prepared = prepare_evidence_packet(
        packet,
        train_smiles=[],
        validation_smiles=[],
        test_smiles=["C1CCCCC1"],
    )
    assert prepared.audit["structure_firewall"]["invalid_external_smiles"] == 1
    assert prepared.records["source_record_id"].tolist() == ["safe"]


def test_smiles_triple_bond_is_not_treated_as_a_csv_comment(tmp_path: Path):
    path = _write_packet(
        tmp_path / "packet-nitrile",
        rows=[{"Drug": "N#CC", "Y": 1, "source_record_id": "nitrile-row"}],
    )
    assert validate_evidence_packet(path).records["source_record_id"].tolist() == [
        "nitrile-row"
    ]


def test_fda_regulatory_dataset_requires_fda_document_citation(tmp_path: Path):
    path = _write_packet(
        tmp_path / "fda-packet",
        source_kind="fda_regulatory_dataset",
    )
    manifest = json.loads(path.read_text())
    manifest["source"].update(
        {
            "citation_url": "https://www.fda.gov/science-research/example-dataset",
            "citation_type": "fda_document",
            "stable_identifier": "FDA:example-dataset-v2",
        }
    )
    path.write_text(json.dumps(manifest))
    assert validate_evidence_packet(path).source_kind == "fda_regulatory_dataset"

    manifest["source"].update(
        {
            "citation_url": "https://doi.org/10.1000/example",
            "citation_type": "doi",
            "stable_identifier": "doi:10.1000/example",
        }
    )
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="FDA packet source kinds"):
        validate_evidence_packet(path)
