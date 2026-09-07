#!/usr/bin/env python3
"""Build the auditable all-22 FDA/literature Evidence Packet source register."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.endpoint_contracts import endpoint_contract
from admet_agent.evidence_packet_loop import CANONICAL_TASKS

OUTPUT = ROOT / "reports/evidence_packet_data_axis_v5/all22_source_register.json"


LEADS = {
    "Caco2_Wang": [
        {
            "source": "O'Hagan and Kell 2015, PeerJ",
            "stable_identifier": "doi:10.7717/peerj.1405; PMID:26618081",
            "url": "https://doi.org/10.7717/peerj.1405",
            "reported_rows": 696,
            "status": "materialized_as_10_per_paper_packets_9_passed_firewall_confirmation_failed",
            "note": "Ten direct-assay source papers were split into separate packets; nine passed packet firewalls, yielding 79 identities after cross-paper reconciliation. The selected configuration failed independent confirmation with pooled-score mean normalized improvement -0.9252%.",
        },
        {
            "source": "Li et al. 2018 Caco-2 permeability collection",
            "stable_identifier": "PMID:30321548",
            "url": "https://pubmed.ncbi.nlm.nih.gov/30321548/",
            "reported_rows": 1366,
            "status": "supplement_download_and_lineage_audit_required",
            "note": "768 compounds and 1,366 measurements; assay/unit harmonization is required.",
        },
    ],
    "HIA_Hou": [
        {
            "source": "Plano et al. 2011 selenocyanate/diselenide ChEMBL HIA values",
            "stable_identifier": "doi:10.1016/j.ejmech.2011.04.054; ChEMBL37:assay758367",
            "url": "https://doi.org/10.1016/j.ejmech.2011.04.054",
            "reported_rows": 6,
            "status": "rejected_computational_or_secondary_hia_not_direct_human_assay",
            "note": "The compounds were newly synthesized antileishmanial agents and were not administered to humans. Six 97.64-100% HIA values therefore cannot be direct human measurements and fail direct-assay admission despite the generic ChEMBL description.",
        }
    ],
    "Pgp_Broccatelli": [
        {
            "source": "FDA clinical P-gp inhibitor table",
            "stable_identifier": "FDA:CYP-transporter-table:P-gp-inhibitors",
            "url": "https://www.fda.gov/drugs/drug-interactions-labeling/healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems",
            "reported_rows": 23,
            "status": "confirmation_failed_rolled_back",
            "note": "Combination products were excluded; 19 identities passed the firewall. The selected 10-row low-mass configuration had pooled-score confirmation mean -0.1034% despite 2/3 positive seeds, so it was rolled back.",
        }
    ],
    "Bioavailability_Ma": [
        {
            "source": "Four independent medicinal-chemistry oral-bioavailability studies",
            "stable_identifier": (
                "doi:10.1016/j.bmcl.2019.02.017; "
                "doi:10.1371/journal.pntd.0009490; "
                "doi:10.1021/acs.jmedchem.0c02083; "
                "doi:10.1021/acsmedchemlett.1c00195"
            ),
            "url": "https://doi.org/10.1016/j.bmcl.2019.02.017",
            "reported_rows": 111,
            "status": "four_paper_packets_confirmation_passed_active_incumbent",
            "note": "Exact measured rat/mouse oral F% values were thresholded at 20% within each explicitly documented assay; impossible values above 100% and nonnumeric records were excluded. All 111 unique structures passed the firewall. The low-mass 28-row configuration confirmed at pooled-score mean +1.8999%, with 2/3 frozen seeds positive.",
        }
    ],
    "Lipophilicity_AstraZeneca": [
        {
            "source": "Lombardo et al. 2002 experimental ElogD(oct) at pH 7.4",
            "stable_identifier": "doi:10.1021/jm0100990; ChEMBL37:assay16545",
            "url": "https://doi.org/10.1021/jm0100990",
            "reported_rows": 90,
            "status": "firewall_admitted_confirmation_failed_rolled_back",
            "note": "Thirty-nine identities survived full structure decontamination. Development was positive on 5/5 seeds, but frozen confirmation was negative on 3/3 seeds with pooled-score mean -0.8537%, indicating assay or chemical-series domain shift.",
        }
    ],
    "Solubility_AqSolDB": [
        {
            "source": "Granberg et al. 2024 pH 7.4 kinetic solubility assay",
            "stable_identifier": "doi:10.1021/acs.jmedchem.3c02184; ChEMBL37:assay2442720",
            "url": "https://doi.org/10.1021/acs.jmedchem.3c02184",
            "reported_rows": 46,
            "status": "confirmation_passed_active_incumbent",
            "note": "Only exact experimental logS values were retained; 43 identities survived the firewall. The selected full packet at base weight 0.70 x multiplier 0.25 confirmed at pooled-score mean +0.6616%, 2/3 seeds positive.",
        }
    ],
    "BBB_Martins": [
        {
            "source": "B3DB external-validation and source tables",
            "stable_identifier": "doi:10.1038/s41597-021-01069-5",
            "url": "https://github.com/theochem/B3DB",
            "reported_rows": 175,
            "status": "aggregate_rejected_no_direct_source_packet",
            "note": "The external table is predominantly DrugBank/review-derived categorical evidence rather than a uniform direct BBB assay. The 50-source main table is a heterogeneous reaggregation, so neither was relabeled as a paper-style training packet.",
        }
    ],
    "PPBR_AZ": [
        {
            "source": "Sutherland et al. 2023 multi-source human plasma-protein-binding compilation",
            "stable_identifier": "doi:10.1038/s41467-023-40064-9; PMID:37468498; PMCID:PMC10356841",
            "url": "https://doi.org/10.1038/s41467-023-40064-9",
            "reported_rows": 1007,
            "status": "four_source_packets_audited_development_no_robust_gain_rolled_back",
            "note": "Only exact (=) PPB records were used. The 560-row NIBR packet was rejected at 5.5556% test-skeleton coverage; Smit, DrugCentral, and Pharmapendium packets admitted 45, 84, and 251 identities, but all tested source subsets had negative development means and were rolled back before confirmation.",
        },
        {
            "source": "European Commission JRC EURL ECVAM equilibrium-dialysis human plasma-protein-binding assay",
            "stable_identifier": "doi:10.2905/JRC.SR09628",
            "url": "https://doi.org/10.2905/JRC.SR09628",
            "reported_rows": 25,
            "status": "firewall_admitted_development_no_robust_gain_rolled_back",
            "note": "Twenty-two identities survived strict recovery/variance quality filters and the structure firewall; the best development mean was -0.1388%, so confirmation was not opened.",
        },
        {
            "source": "Ghafourian and Amin 2013 plasma protein binding collection",
            "stable_identifier": "doi:10.5681/bi.2013.011; PMID:23678466",
            "url": "https://pubmed.ncbi.nlm.nih.gov/23678466/",
            "reported_rows": 794,
            "status": "supplement_availability_and_lineage_audit_required",
            "note": "Human plasma-protein binding values collated from literature; document-level provenance must be retained.",
        },
        {
            "source": "Votano et al. 2006 human serum protein binding",
            "stable_identifier": "doi:10.1021/jm051245v; PMID:17125269",
            "url": "https://pubmed.ncbi.nlm.nih.gov/17125269/",
            "reported_rows": 1008,
            "status": "supplement_availability_and_lineage_audit_required",
            "note": "Public-source measurements; may overlap later AstraZeneca compilation and therefore requires identity/source audit.",
        },
    ],
    "VDss_Lombardo": [
        {
            "source": "2017 whole-body PBPK volume-of-distribution dataset",
            "stable_identifier": "PMC:PMC5285473",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5285473/",
            "reported_rows": 63,
            "status": "rejected_complete_native_identity_overlap",
            "note": "Official Table S2 was audited: all 63 drug identities exactly match the native Lombardo train/test name union (54 train, 9 test), so no row can survive the mandatory exact-identity firewall; test labels were not loaded.",
        }
    ],
    "CYP2C9_Veith": [
        {
            "source": "Gonzalez et al. 2021 CYP2C9 P450-Glo qHTS, PubChem AID 1645842",
            "stable_identifier": "doi:10.1124/dmd.120.000320; PubChem:AID1645842",
            "url": "https://pubchem.ncbi.nlm.nih.gov/bioassay/1645842",
            "reported_rows": 4464,
            "status": "rejected_test_skeleton_coverage",
            "note": "Candidate audit coverage 5.7036% exceeded the preregistered 5% packet-rejection threshold.",
        }
    ],
    "CYP2D6_Veith": [
        {
            "source": "Gonzalez et al. 2021 CYP2D6 P450-Glo qHTS, PubChem AID 1645840",
            "stable_identifier": "doi:10.1124/dmd.120.000320; PubChem:AID1645840",
            "url": "https://pubchem.ncbi.nlm.nih.gov/bioassay/1645840",
            "reported_rows": 4278,
            "status": "rejected_test_skeleton_coverage",
            "note": "Candidate audit coverage 7.8867% exceeded the preregistered 5% packet-rejection threshold.",
        }
    ],
    "CYP3A4_Veith": [
        {
            "source": "Gonzalez et al. 2021 CYP3A4 P450-Glo qHTS, PubChem AID 1645841",
            "stable_identifier": "doi:10.1124/dmd.120.000320; PubChem:AID1645841",
            "url": "https://pubchem.ncbi.nlm.nih.gov/bioassay/1645841",
            "reported_rows": 4349,
            "status": "modeling_ready_confirmation_passed",
            "note": "3,794 rows admitted; selected 25% quantity at weight 0.25; independent pooled-score confirmation mean +0.08293%, 2/3 positive.",
        }
    ],
    "CYP2C9_Substrate_CarbonMangels": [
        {
            "source": "FDA clinical CYP2C9 sensitive and moderate-sensitive substrates",
            "stable_identifier": "FDA:CYP-DDI-table:2C9",
            "url": "https://www.fda.gov/drugs/drug-interactions-labeling/healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems",
            "reported_rows": 5,
            "status": "confirmation_failed_rolled_back",
            "note": "The sensitive packet was fully removed by exact-overlap filtering; one moderate-sensitive identity remained. Pooled-score confirmation mean -1.0935%, 1/3 positive, so it was rolled back.",
        }
    ],
    "CYP2D6_Substrate_CarbonMangels": [
        {
            "source": "FDA clinical CYP2D6 sensitive and moderate-sensitive substrates",
            "stable_identifier": "FDA:CYP-DDI-table:2D6",
            "url": "https://www.fda.gov/drugs/drug-interactions-labeling/healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems",
            "reported_rows": 17,
            "status": "confirmation_passed_active_incumbent",
            "note": "Five identities survived packet firewalls; selected four rows with tier-aware base weights and a 0.75 multiplier. Pooled-score confirmation mean +3.3306%, 2/3 seeds positive.",
        }
    ],
    "CYP3A4_Substrate_CarbonMangels": [
        {
            "source": "FDA clinical CYP3A sensitive and moderate-sensitive substrates",
            "stable_identifier": "FDA:CYP-DDI-table:3A",
            "url": "https://www.fda.gov/drugs/drug-interactions-labeling/healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems",
            "reported_rows": 60,
            "status": "confirmation_passed_active_incumbent",
            "note": "Fifty identities survived packet firewalls; selected 12 rows with tier-aware base weights. Pooled-score confirmation mean +0.9480%, 3/3 seeds positive.",
        }
    ],
    "Half_Life_Obach": [
        {
            "source": "Three original healthy-subject clinical PK studies",
            "stable_identifier": "doi:10.1128/aac.00567-09; doi:10.1128/aac.00682-09; doi:10.1128/aac.00951-07",
            "url": "https://doi.org/10.1128/aac.00682-09",
            "reported_rows": 3,
            "status": "three_paper_packets_confirmation_passed_active_incumbent",
            "note": "S-013420, nemonoxacin, and maribavir human terminal/elimination half-lives passed all structure firewalls. Repeated nemonoxacin dose cohorts were reconciled by a within-paper median. The 2-row low-mass configuration confirmed on fresh v5.3 seeds at pooled mean +8.3889%, 2/3 seeds positive.",
        }
    ],
    "Clearance_Hepatocyte_AZ": [
        {
            "source": "European Commission JRC EURL ECVAM uniform human hepatocyte clearance assay",
            "stable_identifier": "doi:10.2905/JRC.SR09628",
            "url": "https://doi.org/10.2905/JRC.SR09628",
            "reported_rows": 87,
            "status": "confirmation_failed_rolled_back",
            "note": "41 identities survived structure firewall and concentration-replicate reconciliation. Development selected 25% quantity x 0.5 multiplier; pooled-score confirmation mean -0.1826% but only 1/3 seeds positive.",
        },
        {
            "source": "Pirovano et al. 2016 human hepatocyte clearance",
            "stable_identifier": "doi:10.1016/j.etap.2016.01.017; PMID:26874337",
            "url": "https://pubmed.ncbi.nlm.nih.gov/26874337/",
            "reported_rows": 118,
            "status": "supplement_availability_and_unit_alignment_required",
            "note": "Literature-derived human hepatocyte CLint; source-level record provenance and AZ-unit alignment required.",
        }
    ],
    "Clearance_Microsome_AZ": [
        {
            "source": "Granberg et al. 2024 human liver microsome intrinsic-clearance assay",
            "stable_identifier": "doi:10.1021/acs.jmedchem.3c02184; ChEMBL37:assay2442728",
            "url": "https://doi.org/10.1021/acs.jmedchem.3c02184",
            "reported_rows": 45,
            "status": "confirmation_passed_active_incumbent_high_variance",
            "note": "Interval-censored rows were excluded and exact values were mapped to the native 3-150 mL.min-1.g-1 range. Thirty-nine identities survived the firewall; a 10-row low-mass configuration confirmed at pooled-score mean +4.3036%, 2/3 seeds positive.",
        },
        {
            "source": "Pirovano et al. 2016 human liver microsome clearance",
            "stable_identifier": "doi:10.1016/j.etap.2016.01.017; PMID:26874337",
            "url": "https://pubmed.ncbi.nlm.nih.gov/26874337/",
            "reported_rows": 115,
            "status": "supplement_availability_and_unit_alignment_required",
            "note": "Literature-derived human microsomal CLint; source-level record provenance and AZ-unit alignment required.",
        }
    ],
    "LD50_Zhu": [
        {
            "source": "EPA/NICEATM Helman et al. 2019 rat oral LD50 dataset",
            "stable_identifier": "doi:10.1186/s13321-019-0383-2; genra-py:3e9b68f",
            "url": "https://github.com/i-shah/genra-py",
            "reported_rows": 7011,
            "status": "rejected_same_source_lineage_and_test_skeleton_coverage",
            "note": "The label-blind audit found 4,472 exact native-development overlaps, 1,112 exact test overlaps, and 70.8731% test-skeleton coverage versus the 5% limit. The aggregate includes TEST/ChemIDplus-family data and was rejected whole before training.",
        }
    ],
    "hERG": [
        {
            "source": "Dorn et al. 2005 PDSP hERG fluorescence assay, PubChem AID 376",
            "stable_identifier": "doi:10.1177/1087057104272045; PMID:15964935; PubChem:AID376",
            "url": "https://pubchem.ncbi.nlm.nih.gov/bioassay/376",
            "reported_rows": 1960,
            "status": "modeling_ready_confirmation_not_robust",
            "note": "1,909 rows admitted; selected 50% quantity at weight 0.25; pooled-score confirmation mean +0.8941% but only 1/3 seeds strictly positive.",
        },
        {
            "source": "NCGC hERG thallium-flux assay, PubChem AID 588834",
            "stable_identifier": "PubChem:AID588834",
            "url": "https://pubchem.ncbi.nlm.nih.gov/bioassay/588834",
            "reported_rows": 2688,
            "status": "paper_identity_and_lineage_audit_required",
            "note": "Potential second assay packet; must verify peer-reviewed citation and independence from Wang 2016 before materialization.",
        },
    ],
    "AMES": [
        {
            "source": "Two independent peer-reviewed TA98/TA100/TAMix Ames assay panels",
            "stable_identifier": "doi:10.1016/j.bmcl.2020.127662; doi:10.1016/j.bmcl.2012.10.076",
            "url": "https://doi.org/10.1016/j.bmcl.2020.127662",
            "reported_rows": 220,
            "status": "two_paper_packets_confirmation_failed_rolled_back",
            "note": "Paper-level any-positive aggregation yielded 58 compounds, all firewall-clean. Development mean was +0.3180%, but frozen pooled-score confirmation had mean +0.0101% and only 1/3 positive seeds, so both packets were rolled back.",
        },
        {
            "source": "2019 Ames/QSAR International Challenge",
            "stable_identifier": "PMC:PMC6402315",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6402315/",
            "reported_rows": 12140,
            "status": "blocked_proprietary_compound_table",
            "note": "Independent chemicals are reported but the molecule-level challenge table is not publicly released; cannot train from summary counts.",
        }
    ],
    "DILI": [
        {
            "source": "FDA DILIrank 2.0",
            "stable_identifier": "FDA:DILIrank2.0",
            "url": "https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset",
            "reported_rows": 1336,
            "status": "strict_temporal_new_packet_confirmation_passed",
            "note": "Only version-2 rows marked New were eligible; Most-DILI-concern mapped to 1, No-DILI-concern to 0, and Less/Ambiguous were excluded. The 128 admitted identities used a selected 50% quantity at effective row weight 0.20; independent pooled-score confirmation mean +2.1624%, 3/3 seeds positive.",
        },
        {
            "source": "2024 DILI ML gold-standard compilation",
            "stable_identifier": "PMC:PMC11337212",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11337212/",
            "reported_rows": 1111,
            "status": "supplement_available_lineage_audit_required",
            "note": "SMILES supplement exists; DILIrank/DILIst lineage overlap and label semantics must be separated before admission.",
        },
    ],
}


SAME_SOURCE_NOTES = {
    "Caco2_Wang": "Do not reuse Wang 2016 Caco-2 benchmark tables or renamed copies.",
    "HIA_Hou": "Do not reuse Hou HIA compilation or admetSAR/Hou reaggregations.",
    "Pgp_Broccatelli": "Do not reuse Broccatelli transporter compilation, including its supplement.",
    "Bioavailability_Ma": "Do not reuse Ma oral-bioavailability compilation or renamed copies.",
    "hERG": "Do not use the Wang 2016 648-compound table or an integrated database containing it without record-level lineage removal.",
    "AMES": "Do not reuse Hansen/TDC Ames or AMES_Alberga derivatives.",
}


def build() -> dict:
    tasks = []
    for task in CANONICAL_TASKS:
        contract = endpoint_contract(task)
        leads = LEADS.get(task, [])
        tasks.append(
            {
                "task": task,
                "endpoint_contract": contract.to_dict(),
                "benchmark_source_family": contract.source_family,
                "same_source_guard": SAME_SOURCE_NOTES.get(
                    task,
                    "Do not use the benchmark source family or any renamed/reaggregated copy.",
                ),
                "candidate_sources": leads,
                "candidate_count": len(leads),
                "acquisition_status": (
                    "candidate_sources_identified" if leads else "search_required"
                ),
                "search_queries": [
                    f'"{contract.property_name}" supplementary data SMILES assay',
                    f'site:fda.gov "{contract.property_name}" molecule table',
                ],
                "admission_requirements": [
                    "stable FDA/DOI/PubMed citation and molecule-level locator",
                    "direct endpoint/species/system match with explicit unit or binary threshold",
                    "verified independence from benchmark source lineage",
                    "whole-development exact-identity removal",
                    "label-blind test exact-identity and Tanimoto >=0.90 removal",
                    "reject whole packet when test skeleton coverage exceeds 0.05",
                    "training-only use after modeling_ready review",
                ],
            }
        )
    if len(tasks) != 22 or {item["task"] for item in tasks} != set(CANONICAL_TASKS):
        raise RuntimeError("source register must cover exactly the canonical 22 tasks")
    return {
        "protocol": "admet-evidence-packet-all22-source-register-v5.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": 22,
        "tasks": tasks,
        "source_policy": {
            "priority": [
                "FDA primary document",
                "independent peer-reviewed assay",
                "public regulatory scientific assay report",
            ],
            "packet_unit": "one document plus one endpoint/assay/label transform",
            "multi_paper_merge": "canonical-identity deduplication; classification conflicts dropped; regression duplicates median-aggregated with packet IDs retained",
            "confirmation_metric_aggregation": "ratio of mean baseline/challenger scores; never mean of seed-wise ratios",
            "legacy_bulk_pool_used": False,
            "test_labels_loaded": False,
        },
    }


def main() -> None:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    identified = sum(bool(item["candidate_sources"]) for item in payload["tasks"])
    print(json.dumps({"output": str(OUTPUT), "tasks": 22, "with_leads": identified}))


if __name__ == "__main__":
    main()
