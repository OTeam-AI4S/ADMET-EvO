"""Canonical metadata for all 22 TDC ADMET Group benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    metric: str
    category: str

    @property
    def maximize(self) -> bool:
        return self.metric != "mae"


_ROWS = [
    ("Caco2_Wang", "regression", "mae", "absorption"),
    ("HIA_Hou", "classification", "auroc", "absorption"),
    ("Pgp_Broccatelli", "classification", "auroc", "absorption"),
    ("Bioavailability_Ma", "classification", "auroc", "absorption"),
    ("Lipophilicity_AstraZeneca", "regression", "mae", "absorption"),
    ("Solubility_AqSolDB", "regression", "mae", "absorption"),
    ("BBB_Martins", "classification", "auroc", "distribution"),
    ("PPBR_AZ", "regression", "mae", "distribution"),
    ("VDss_Lombardo", "regression", "spearman", "distribution"),
    ("CYP2C9_Veith", "classification", "auprc", "metabolism"),
    ("CYP2D6_Veith", "classification", "auprc", "metabolism"),
    ("CYP3A4_Veith", "classification", "auprc", "metabolism"),
    ("CYP2C9_Substrate_CarbonMangels", "classification", "auprc", "metabolism"),
    ("CYP2D6_Substrate_CarbonMangels", "classification", "auprc", "metabolism"),
    ("CYP3A4_Substrate_CarbonMangels", "classification", "auroc", "metabolism"),
    ("Half_Life_Obach", "regression", "spearman", "excretion"),
    ("Clearance_Hepatocyte_AZ", "regression", "spearman", "excretion"),
    ("Clearance_Microsome_AZ", "regression", "spearman", "excretion"),
    ("LD50_Zhu", "regression", "mae", "toxicity"),
    ("hERG", "classification", "auroc", "toxicity"),
    ("AMES", "classification", "auroc", "toxicity"),
    ("DILI", "classification", "auroc", "toxicity"),
]

TASKS = {row[0].lower(): TaskSpec(*row) for row in _ROWS}


def resolve_task(name: str) -> TaskSpec:
    try:
        return TASKS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown task {name!r}; use `admet-agent list`.") from exc
