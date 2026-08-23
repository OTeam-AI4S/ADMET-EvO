from admet_agent.endpoint_contracts import endpoint_contract
from admet_agent.external_data_scout import (
    ExternalDataScout,
    ExternalDatasetCandidate,
    judge_endpoint,
)


def _candidate(**changes):
    values = {
        "source_id": "source",
        "endpoint_key": "herg_blockade",
        "property_name": "hERG blockade",
        "task_type": "classification",
        "label_rule": "hERG blocker when IC50 < 10000 nM",
        "species": "human",
        "assay_system": "functional KCNH2 IC50",
        "unit": "nM",
        "transform": "IC50 nM < 10000",
        "provenance": "document-level public records",
        "license": "CC BY",
        "access_url": "https://example.test/data",
        "source_family": "external",
        "target_id": "CHEMBL240",
        "source_overlap_risk": "medium",
    }
    values.update(changes)
    return ExternalDatasetCandidate(**values)


def test_related_thresholded_herg_endpoint_requires_bridge_not_direct_merge():
    result = judge_endpoint(_candidate(), endpoint_contract("hERG"))
    assert result.decision == "bridge_required"
    assert "label_semantics_or_threshold_not_proven_identical" in result.bridge_requirements
    assert result.next_action.startswith("run_structure_firewall")


def test_species_mismatch_and_missing_license_are_hard_rejections():
    result = judge_endpoint(_candidate(species="rat", license=""), endpoint_contract("hERG"))
    assert result.decision == "reject"
    assert "species_mismatch" in result.hard_rejection_reasons
    assert "missing_license" in result.hard_rejection_reasons


def test_scout_finds_multiple_sources_without_automatic_download():
    result = ExternalDataScout().discover("hERG")
    assert result["candidate_count"] == 3
    assert {row["judgment"]["decision"] for row in result["candidates"]} == {"bridge_required"}
    assert result["automatic_download_allowed"] is False
    assert result["test_labels_loaded"] is False


def test_connector_failure_is_isolated():
    class Broken:
        name = "broken"

        def search(self, contract):
            raise RuntimeError("offline")

    result = ExternalDataScout([Broken()]).discover("hERG")
    assert result["candidate_count"] == 0
    assert result["connector_failures"] == [{"connector": "broken", "error": "RuntimeError"}]
