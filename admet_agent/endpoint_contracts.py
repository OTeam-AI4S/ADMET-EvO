"""Scientific contracts for the canonical 22 TDC ADMET endpoints.

The contracts deliberately preserve TDC-native labels.  External observations
must declare their assay, species, unit and transformation before they can be
merged; a matching endpoint name alone is never considered sufficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .tasks import TASKS, TaskSpec, resolve_task


@dataclass(frozen=True)
class EndpointContract:
    tdc_name: str
    endpoint_id: str
    property_name: str
    category: str
    task_type: str
    primary_metric: str
    metric_direction: str
    label_semantics: str
    tdc_native_unit: str | None = None
    species: str | None = None
    assay_system: str | None = None
    source_family: str | None = None
    external_merge_policy: str = "explicit_assay_and_unit_alignment"
    discovery_split: str = "tdc_default_multi_seed"
    certification_split: str = "tdc_evaluator_owned_test"

    def to_dict(self) -> dict:
        return asdict(self)


_DETAILS = {
    "Caco2_Wang": ("caco2_permeability", "Caco-2 permeability", "continuous TDC-native permeability value", None, "human", "Caco-2 cell monolayer permeability", "Wang"),
    "HIA_Hou": ("human_intestinal_absorption", "human intestinal absorption", "TDC binary HIA label", None, "human", "curated in-vivo absorption", "Hou"),
    "Pgp_Broccatelli": ("pgp_inhibition", "P-glycoprotein inhibition", "TDC binary inhibitor label", None, "human", "P-gp inhibition assay", "Broccatelli"),
    "Bioavailability_Ma": ("oral_bioavailability", "oral bioavailability", "TDC binary bioavailability label", None, None, "curated in-vivo bioavailability", "Ma"),
    "Lipophilicity_AstraZeneca": ("lipophilicity", "lipophilicity", "continuous TDC-native lipophilicity value", None, None, "AstraZeneca lipophilicity assay", "AstraZeneca"),
    "Solubility_AqSolDB": ("aqueous_solubility", "aqueous solubility", "continuous TDC-native solubility value", None, None, "heterogeneous aqueous-solubility records", "AqSolDB"),
    "BBB_Martins": ("blood_brain_barrier", "blood-brain barrier penetration", "TDC binary BBB label", None, None, "curated BBB penetration evidence", "Martins"),
    "PPBR_AZ": ("plasma_protein_binding", "plasma protein binding rate", "continuous TDC-native PPBR value", None, None, "plasma protein binding assay", "AstraZeneca"),
    "VDss_Lombardo": ("steady_state_volume_distribution", "steady-state volume of distribution", "continuous TDC-native VDss value", None, None, "curated in-vivo pharmacokinetics", "Lombardo"),
    "CYP2C9_Veith": ("cyp2c9_inhibition", "CYP2C9 inhibition", "TDC binary inhibitor label", None, "human", "CYP inhibition assay", "Veith"),
    "CYP2D6_Veith": ("cyp2d6_inhibition", "CYP2D6 inhibition", "TDC binary inhibitor label", None, "human", "CYP inhibition assay", "Veith"),
    "CYP3A4_Veith": ("cyp3a4_inhibition", "CYP3A4 inhibition", "TDC binary inhibitor label", None, "human", "CYP inhibition assay", "Veith"),
    "CYP2C9_Substrate_CarbonMangels": ("cyp2c9_substrate", "CYP2C9 substrate status", "TDC binary substrate label", None, "human", "curated CYP substrate evidence", "Carbon-Mangels"),
    "CYP2D6_Substrate_CarbonMangels": ("cyp2d6_substrate", "CYP2D6 substrate status", "TDC binary substrate label", None, "human", "curated CYP substrate evidence", "Carbon-Mangels"),
    "CYP3A4_Substrate_CarbonMangels": ("cyp3a4_substrate", "CYP3A4 substrate status", "TDC binary substrate label", None, "human", "curated CYP substrate evidence", "Carbon-Mangels"),
    "Half_Life_Obach": ("half_life", "elimination half-life", "continuous TDC-native half-life value", None, None, "curated pharmacokinetic measurements", "Obach"),
    "Clearance_Hepatocyte_AZ": ("hepatocyte_clearance", "hepatocyte clearance", "continuous TDC-native clearance value", None, None, "hepatocyte clearance assay", "AstraZeneca"),
    "Clearance_Microsome_AZ": ("microsome_clearance", "microsomal clearance", "continuous TDC-native clearance value", None, None, "microsomal clearance assay", "AstraZeneca"),
    "LD50_Zhu": ("acute_toxicity_ld50", "acute toxicity LD50", "continuous TDC-native LD50 value", None, None, "curated acute-toxicity measurements", "Zhu"),
    "hERG": ("herg_blockade", "hERG blockade", "TDC binary hERG label", None, "human", "heterogeneous hERG assays", "TDC hERG"),
    "AMES": ("ames_mutagenicity", "Ames mutagenicity", "TDC binary mutagenicity label", None, None, "Ames bacterial mutagenicity assay", "TDC AMES"),
    "DILI": ("drug_induced_liver_injury", "drug-induced liver injury", "TDC binary DILI label", None, "human", "curated clinical/preclinical DILI evidence", "TDC DILI"),
}


def _build_contract(spec: TaskSpec) -> EndpointContract:
    details = _DETAILS[spec.name]
    return EndpointContract(
        tdc_name=spec.name,
        endpoint_id=details[0],
        property_name=details[1],
        category=spec.category,
        task_type=spec.task_type,
        primary_metric=spec.metric,
        metric_direction="maximize" if spec.maximize else "minimize",
        label_semantics=details[2],
        tdc_native_unit=details[3],
        species=details[4],
        assay_system=details[5],
        source_family=details[6],
    )


ENDPOINT_CONTRACTS = {
    name: _build_contract(spec) for name, spec in ((value.name, value) for value in TASKS.values())
}


def endpoint_contract(name: str) -> EndpointContract:
    return ENDPOINT_CONTRACTS[resolve_task(name).name]


def validate_contract_registry() -> None:
    task_names = {item.name for item in TASKS.values()}
    contract_names = set(ENDPOINT_CONTRACTS)
    if task_names != contract_names:
        raise RuntimeError(
            f"Endpoint contract mismatch: missing={sorted(task_names - contract_names)}, "
            f"extra={sorted(contract_names - task_names)}"
        )


validate_contract_registry()
