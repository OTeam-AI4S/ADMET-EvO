"""Auditable external-data plans for all 22 TDC ADMET tasks.

The registry separates endpoint-matched observations from bridge/pretraining
data.  It is intentionally a source plan, not permission to merge a dataset:
every acquired row must still pass its endpoint contract and structure
firewall before use.
"""

from __future__ import annotations

from .endpoint_contracts import endpoint_contract
from .tasks import TASKS

FIREWALL_POLICY = {
    "identity_key": "standardized parent InChIKey",
    "exact_overlap_action": "remove candidate",
    "fingerprint": "Morgan radius=2, 2048 bits",
    "similarity_rule": "keep only max Tanimoto < 0.80",
    "threshold": 0.80,
    "protected_structures": "task evaluation structures and evaluator-owned test structures when available",
    "test_labels_accessible": False,
    "source_holdout_required": True,
    "same_lineage_as_benchmark": "exclude from external-gain claim",
}


PRIMARY_REFERENCES = {
    "Caco2_Wang": "Wang et al., J Chem Inf Model 2016, 56, 763-773. doi:10.1021/acs.jcim.5b00642",
    "HIA_Hou": "Hou et al., J Chem Inf Model 2007, 47, 208-218. doi:10.1021/ci600343x",
    "Pgp_Broccatelli": "Broccatelli et al., J Med Chem 2011, 54, 1740-1751. doi:10.1021/jm101421d",
    "Bioavailability_Ma": "Ma et al., J Pharm Biomed Anal 2008, 47, 677-682. doi:10.1016/j.jpba.2008.03.023",
    "Lipophilicity_AstraZeneca": "AstraZeneca experimental in-vitro DMPK and physicochemical dataset (2016), redistributed through MoleculeNet/TDC.",
    "Solubility_AqSolDB": "Sorkun et al., Sci Data 2019, 6, 143. doi:10.1038/s41597-019-0151-1",
    "BBB_Martins": "Martins et al., J Chem Inf Model 2012, 52, 1686-1697. doi:10.1021/ci300124c",
    "PPBR_AZ": "AstraZeneca experimental in-vitro DMPK and physicochemical dataset (2016), redistributed through MoleculeNet/TDC.",
    "VDss_Lombardo": "Lombardo and Jing, J Chem Inf Model 2016, 56, 2042-2052. doi:10.1021/acs.jcim.6b00044",
    "CYP2C9_Veith": "Veith et al., Nat Biotechnol 2009, 27, 1050-1055. doi:10.1038/nbt.1581",
    "CYP2D6_Veith": "Veith et al., Nat Biotechnol 2009, 27, 1050-1055. doi:10.1038/nbt.1581",
    "CYP3A4_Veith": "Veith et al., Nat Biotechnol 2009, 27, 1050-1055. doi:10.1038/nbt.1581",
    "CYP2C9_Substrate_CarbonMangels": "Carbon-Mangels and Hutter, Mol Inform 2011, 30, 885-895. doi:10.1002/minf.201100069",
    "CYP2D6_Substrate_CarbonMangels": "Carbon-Mangels and Hutter, Mol Inform 2011, 30, 885-895. doi:10.1002/minf.201100069",
    "CYP3A4_Substrate_CarbonMangels": "Carbon-Mangels and Hutter, Mol Inform 2011, 30, 885-895. doi:10.1002/minf.201100069",
    "Half_Life_Obach": "Obach et al., Drug Metab Dispos 2008, 36, 1385-1405. doi:10.1124/dmd.108.020479",
    "Clearance_Hepatocyte_AZ": "AstraZeneca experimental in-vitro DMPK dataset (2016), redistributed through MoleculeNet/TDC.",
    "Clearance_Microsome_AZ": "Di et al., Eur J Med Chem 2012, 57, 441-448. doi:10.1016/j.ejmech.2012.09.021",
    "LD50_Zhu": "Zhu et al., Chem Res Toxicol 2009, 22, 1913-1921. doi:10.1021/tx900189p",
    "hERG": "Wang et al., Mol Pharm 2016, 13, 2855-2866. doi:10.1021/acs.molpharmaceut.6b00150",
    "AMES": "Xu et al., J Chem Inf Model 2012, 52, 2840-2847. doi:10.1021/ci300400a",
    "DILI": "Xu et al., J Chem Inf Model 2015, 55, 2085-2093. doi:10.1021/acs.jcim.5b00238",
}


def _source(source_id, role, url, acquisition, guard, evidence, status="acquisition_pending"):
    return {
        "source_id": source_id,
        "role": role,
        "url": url,
        "acquisition": acquisition,
        "endpoint_guard": guard,
        "evidence": evidence,
        "status": status,
    }


CHEMBL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest"
PKDB_URL = "https://pk-db.com/"
EPA_URL = "https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data"


TASK_SOURCE_PLANS = {
    "Caco2_Wang": [
        _source("chembl_caco2", "direct", CHEMBL_URL, "bulk ChEMBL assay/activity tables; Caco-2 Papp queries", "Caco-2 monolayer, direction, pH and units must match", "ChEMBL is a manually curated bioactivity database."),
        _source("pubchem_caco2", "direct", PUBCHEM_URL, "BioAssay bulk export for Caco-2 permeability", "quantitative Papp only; normalize direction and units", "PubChem BioAssay provides depositor assay records."),
        _source("ncats_pampa", "bridge", "https://tdcommons.ai/single_pred_tasks/adme/#pampa-permeability-ncats", "local data/v5_public_cohort_raw/pampa_ncats.tab", "PAMPA is not Caco-2; representation pretraining only", "Cell-free passive permeability is related but not endpoint-identical.", "local_source_ready"),
    ],
    "HIA_Hou": [
        _source("pkdb_hia", "direct", PKDB_URL, "PK-DB API/dump plus linked study metadata", "human in-vivo absorbed fraction or explicit HIA class", "PK-DB is an open pharmacokinetics database; Grzegorzewski et al., NAR 2021."),
        _source("chembl_hia", "bridge", CHEMBL_URL, "bulk ADME permeability/absorption assays", "in-vitro permeability cannot be relabeled as human HIA", "Mechanistically related ADME assays support transfer only."),
    ],
    "Pgp_Broccatelli": [
        _source("chembl_abcb1", "direct", CHEMBL_URL, "target CHEMBL4302/ABCB1 activity records", "inhibition only; keep substrate/transport labels separate", "ChEMBL target and assay annotations support endpoint filtering."),
        _source("pubchem_pgp", "direct", PUBCHEM_URL, "ABCB1/P-gp inhibition BioAssays", "exclude cytotoxicity-only and substrate-only assays", "PubChem provides assay protocols and outcomes."),
    ],
    "Bioavailability_Ma": [
        _source("pkdb_bioavailability", "direct", PKDB_URL, "human/animal oral bioavailability observations", "species, route and F% required; map labels only by preregistered cutoff", "PK-DB stores study-level PK observations with provenance."),
        _source("fda_labels_bioavailability", "bridge", "https://open.fda.gov/apis/drug/label/", "downloadable openFDA drug labels", "text-mined claims require manual/dual validation", "FDA labels provide clinical pharmacology evidence."),
    ],
    "Lipophilicity_AstraZeneca": [
        _source("chembl_lipophilicity", "direct", CHEMBL_URL, "bulk calculated_properties and assay records", "experimental logD/logP only; preserve pH and method", "ChEMBL records measured physicochemical properties."),
        _source("pubchem_xlogp", "bridge", "https://pubchem.ncbi.nlm.nih.gov/docs/compound-property", "PubChem compound property bulk export", "computed XlogP is a feature/teacher, not an experimental label", "PubChem exposes standardized compound properties."),
    ],
    "Solubility_AqSolDB": [
        _source("chembl_solubility", "direct", CHEMBL_URL, "bulk solubility assays", "experimental aqueous solubility; normalize mol/L, pH and temperature", "ChEMBL contains curated ADME assay measurements."),
        _source("pubchem_solubility", "direct", PUBCHEM_URL, "compound/assay bulk records with solubility provenance", "exclude predicted values and unresolved mixtures", "PubChem records depositor experimental annotations."),
    ],
    "BBB_Martins": [
        _source("b3db", "direct", "https://github.com/theochem/B3DB", "local data/v5_public_cohort_raw/b3db_classification.tab", "use B3DB BBB+/BBB- evidence; audit Martins lineage overlaps", "B3DB is curated from 50 published resources; Meng et al., Sci Data 2021, doi:10.1038/s41597-021-01069-5.", "local_source_ready"),
        _source("b3db_logbb", "bridge", "https://github.com/theochem/B3DB", "local data/v5_public_cohort_raw/b3db_regression.tab", "continuous logBB is auxiliary unless label mapping is preregistered", "B3DB supplies experimental logBB values.", "local_source_ready"),
    ],
    "PPBR_AZ": [
        _source("chembl_ppb", "direct", CHEMBL_URL, "bulk plasma-protein-binding assays", "species, plasma matrix, free/bound convention and units required", "ChEMBL carries assay-level ADME provenance."),
        _source("pkdb_ppb", "direct", PKDB_URL, "PK-DB protein-binding observations", "retain study/species and harmonize fraction unbound vs bound", "PK-DB links PK values to studies."),
    ],
    "VDss_Lombardo": [
        _source("pkdb_vdss", "direct", PKDB_URL, "steady-state volume-of-distribution observations", "in-vivo VDss only; species and L/kg normalization required", "PK-DB is an open study-level PK resource."),
        _source("fda_labels_vdss", "bridge", "https://open.fda.gov/apis/drug/label/", "clinical pharmacology label extraction", "manual validation and drug-to-structure mapping required", "FDA labels can corroborate human PK values."),
    ],
    "CYP2C9_Veith": [], "CYP2D6_Veith": [], "CYP3A4_Veith": [],
    "CYP2C9_Substrate_CarbonMangels": [], "CYP2D6_Substrate_CarbonMangels": [], "CYP3A4_Substrate_CarbonMangels": [],
    "Half_Life_Obach": [
        _source("pkdb_half_life", "direct", PKDB_URL, "terminal/elimination half-life observations", "route, species and half-life definition required", "PK-DB provides PK observations with study provenance."),
        _source("chembl_half_life", "direct", CHEMBL_URL, "PK assays including ChEMBL assay 1614674", "exclude assay 1614674/Obach lineage from independent-gain evidence", "ChEMBL exposes assay provenance and references."),
    ],
    "Clearance_Hepatocyte_AZ": [
        _source("chembl_hepatocyte_cl", "direct", CHEMBL_URL, "hepatocyte intrinsic-clearance assays", "hepatocyte only; species and units required", "ChEMBL stores DMPK assay descriptions and values."),
        _source("pkdb_clearance", "bridge", PKDB_URL, "in-vivo clearance observations", "systemic in-vivo clearance is auxiliary, not hepatocyte Cl_int", "PK-DB provides complementary PK evidence."),
    ],
    "Clearance_Microsome_AZ": [
        _source("chembl_microsome_cl", "direct", CHEMBL_URL, "liver-microsome intrinsic-clearance assays", "microsome only; species, protein normalization and units required", "ChEMBL stores DMPK assay descriptions and values."),
        _source("pkdb_clearance", "bridge", PKDB_URL, "in-vivo clearance observations", "systemic clearance cannot be relabeled as microsomal Cl_int", "PK-DB provides complementary PK evidence."),
    ],
    "LD50_Zhu": [
        _source("epa_toxvaldb_ld50", "direct", EPA_URL, "ToxValDB bulk download", "oral rat acute LD50 preferred; species, route and mg/kg required", "EPA ToxValDB aggregates in-vivo toxicity values with source provenance."),
        _source("epa_toxrefdb_ld50", "direct", EPA_URL, "ToxRefDB bulk download", "acute single-dose LD50 only", "EPA ToxRefDB provides curated guideline-study data."),
    ],
    "hERG": [
        _source("hergcentral", "direct", "https://tripod.nih.gov/hergcentral/", "local external/ecvoi_v6_tdc_cache/herg_central.tab", "binary hERGCentral assay label; remove Karim/TDC lineage and near neighbors", "hERGCentral is a large public electrophysiology screening resource.", "local_source_ready"),
        _source("chembl_herg", "direct", CHEMBL_URL, "target CHEMBL240 quantitative activities", "preregister IC50 cutoff; assay type and units required", "ChEMBL provides KCNH2 target activities."),
    ],
    "AMES": [
        _source("epa_ames", "direct", EPA_URL, "CompTox/ToxValDB Ames records", "strain and +/-S9 retained; preregister compound-level aggregation", "EPA downloadable toxicology data preserve source provenance."),
        _source("pubchem_ames", "direct", PUBCHEM_URL, "Ames BioAssay bulk export", "exclude predictions; preserve protocol and strain", "PubChem provides depositor assay descriptions and outcomes."),
    ],
    "DILI": [
        _source("fda_dilirank2", "direct", "https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset", "FDA DILIrank 2.0 download", "map severity categories to binary only in sensitivity analysis; retain unmapped", "Olubamiwa et al., Drug Discov Today 2025, 30, 104485."),
        _source("livertox", "bridge", "https://www.ncbi.nlm.nih.gov/books/NBK547852/", "drug-level literature evidence with explicit versioning", "manual adjudication; not a row-wise automatic label source", "NIH LiverTox provides expert-reviewed clinical narratives."),
    ],
}


for isoform in ("2C9", "2D6", "3A4"):
    TASK_SOURCE_PLANS[f"CYP{isoform}_Veith"] = [
        _source(f"chembl_cyp{isoform.lower()}_inhibition", "direct", CHEMBL_URL, f"CYP{isoform} inhibition assays", "inhibition only; exclude Veith assay lineage; retain probe and units", "ChEMBL assay provenance enables lineage exclusion."),
        _source(f"pubchem_cyp{isoform.lower()}_inhibition", "direct", PUBCHEM_URL, f"CYP{isoform} inhibition BioAssays", "exclude substrate/metabolic-stability endpoints", "PubChem provides assay protocol metadata."),
    ]
    TASK_SOURCE_PLANS[f"CYP{isoform}_Substrate_CarbonMangels"] = [
        _source(f"chembl_cyp{isoform.lower()}_substrate", "direct", CHEMBL_URL, f"CYP{isoform} substrate/metabolism records", "substrate evidence only; exclude inhibition-only records", "ChEMBL metabolism assays and references support endpoint filtering."),
        _source(f"fda_label_cyp{isoform.lower()}", "bridge", "https://open.fda.gov/apis/drug/label/", f"validated CYP{isoform} substrate statements", "manual validation; absence of a statement is not a negative", "FDA drug labels contain clinical interaction evidence."),
    ]


def build_task_data_registry() -> dict:
    """Return the complete, validated machine-readable source registry."""

    expected = {spec.name for spec in TASKS.values()}
    observed = set(TASK_SOURCE_PLANS)
    if expected != observed:
        raise RuntimeError(
            f"Task source registry mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    tasks = []
    for task in (spec.name for spec in TASKS.values()):
        sources = TASK_SOURCE_PLANS[task]
        tasks.append(
            {
                "task": task,
                "endpoint_contract": endpoint_contract(task).to_dict(),
                "benchmark_reference": PRIMARY_REFERENCES[task],
                "sources": sources,
                "local_direct_ready": [
                    item["source_id"] for item in sources
                    if item["role"] == "direct" and item["status"] == "local_source_ready"
                ],
                "paper_claim_status": "pilot_runnable" if any(
                    item["role"] == "direct" and item["status"] == "local_source_ready"
                    for item in sources
                ) else "acquisition_required",
            }
        )
    return {
        "registry_version": "tdc22-task-conditioned-data-scaling-v1",
        "firewall_policy": FIREWALL_POLICY,
        "claim_rule": "Only independent, endpoint-matched direct data may support the external-data gain claim; bridge data require a separate ablation.",
        "tasks": tasks,
    }


__all__ = [
    "FIREWALL_POLICY",
    "PRIMARY_REFERENCES",
    "TASK_SOURCE_PLANS",
    "build_task_data_registry",
]
