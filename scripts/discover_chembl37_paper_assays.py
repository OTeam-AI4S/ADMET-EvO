#!/usr/bin/env python3
"""Inventory ChEMBL 37 cohorts at original-paper and assay granularity.

This is a discovery-only step.  It never promotes records or reads test labels.
The heterogeneous endpoint cohorts remain quarantined; downstream materializers
must bind one DOI and one assay to each Evidence Packet and retrieve the raw
activity fields from the frozen ChEMBL SQLite database.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BENCHMARK_DOIS = {
    "HIA_Hou": "10.1021/ci600343x",
    "Bioavailability_Ma": "10.1016/j.jpba.2008.03.023",
    "Lipophilicity_AstraZeneca": None,
    "Solubility_AqSolDB": "10.1038/s41597-019-0151-1",
    "Half_Life_Obach": "10.1124/dmd.108.020479",
    "Clearance_Microsome_AZ": "10.1016/j.ejmech.2012.09.021",
    "LD50_Zhu": "10.1021/tx900189p",
    "AMES": "10.1021/ci300400a",
}

TASKS = tuple(BENCHMARK_DOIS)
PREDICTION_TERMS = re.compile(
    r"\b(qsar|qspr|predict(?:ed|ion)?|comput(?:ed|ational)|model(?:ing|led)?|"
    r"in silico|calculated)\b",
    re.IGNORECASE,
)


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _doi(value: object) -> str:
    text = _clean_text(value).lower()
    text = text.removeprefix("https://doi.org/").removeprefix("doi:")
    return text.rstrip(". ")


def _semantic_flags(task: str, row: dict) -> dict[str, bool]:
    description = row["description"].lower()
    title = row["title"].lower()
    organism = row["assay_organism"].lower()
    units = row["standard_units"].lower()
    direct = False
    if task == "Lipophilicity_AstraZeneca":
        direct = "logd" in description and not PREDICTION_TERMS.search(description)
    elif task == "Solubility_AqSolDB":
        direct = "solub" in description and not PREDICTION_TERMS.search(description)
    elif task == "Clearance_Microsome_AZ":
        direct = (
            "microsom" in description
            and "clearance" in description
            and organism == "homo sapiens"
            and units == "ml.min-1.g-1"
            and not PREDICTION_TERMS.search(description)
        )
    elif task == "LD50_Zhu":
        direct = (
            "ld50" in description
            and organism == "rattus norvegicus"
            and "oral" in description
            and units in {"mg.kg-1", "umol.kg-1"}
            and not PREDICTION_TERMS.search(description)
        )
    elif task == "AMES":
        direct = (
            "ames" in description
            and ("salmonella" in description or "mutagen" in description)
            and not PREDICTION_TERMS.search(description)
        )
    elif task == "Half_Life_Obach":
        direct = (
            "half-life" in description
            and organism == "homo sapiens"
            and units == "hr"
            and any(term in description for term in ("elimination", "terminal", "in vivo"))
            and not any(term in description for term in ("microsom", "plasma stability", "hydrolysis"))
        )
    elif task == "HIA_Hou":
        direct = (
            "intestinal absorption" in description
            and organism == "homo sapiens"
            and any(
                term in title
                for term in ("absorption", "pharmacokinetic", "bioavailability")
            )
            and not PREDICTION_TERMS.search(f"{title} {description}")
        )
    elif task == "Bioavailability_Ma":
        direct = "bioavailability" in description and "oral" in description
    return {
        "description_direct_semantics": bool(direct),
        "prediction_language": bool(PREDICTION_TERMS.search(f"{title} {description}")),
    }


def _summarize_task(path: Path, task: str) -> list[dict]:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame["doi_normalized"] = frame["doi"].map(_doi)
    for column in (
        "title",
        "description",
        "assay_organism",
        "standard_units",
        "standard_relation",
        "standard_type",
        "assay_chembl_id",
    ):
        frame[column] = frame[column].map(_clean_text)
    rows = []
    keys = ["doi_normalized", "assay_id"]
    for (doi, assay_id), group in frame.groupby(keys, dropna=False, sort=False):
        first = group.iloc[0]
        relations = sorted(set(group["standard_relation"]))
        values = pd.to_numeric(group["Y"], errors="coerce")
        benchmark = BENCHMARK_DOIS[task]
        record = {
            "task": task,
            "doi": doi,
            "assay_id": int(assay_id),
            "assay_chembl_id": first["assay_chembl_id"],
            "title": first["title"],
            "description": first["description"],
            "assay_organism": first["assay_organism"],
            "standard_type": first["standard_type"],
            "standard_units": first["standard_units"],
            "standard_relations": "|".join(relations),
            "rows": len(group),
            "unique_structures": int(group["X"].nunique()),
            "y_min": float(values.min()),
            "y_max": float(values.max()),
            "y_mean": float(values.mean()),
            "positive_rows": int((values == 1).sum()) if task in {"HIA_Hou", "Bioavailability_Ma", "AMES"} else None,
            "negative_rows": int((values == 0).sum()) if task in {"HIA_Hou", "Bioavailability_Ma", "AMES"} else None,
            "has_stable_doi": bool(doi.startswith("10.")),
            "peer_reviewed_doi": bool(
                doi.startswith("10.") and not doi.startswith("10.6019/chembl")
            ),
            "benchmark_lineage": bool(benchmark and doi == benchmark),
            "exact_or_approximate_only": set(relations) <= {"=", "~", ""},
            "test_labels_loaded": False,
        }
        record.update(_semantic_flags(task, record))
        record["discovery_candidate"] = bool(
            record["peer_reviewed_doi"]
            and not record["benchmark_lineage"]
            and record["exact_or_approximate_only"]
            and record["description_direct_semantics"]
            and not record["prediction_language"]
        )
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("data/external/chembl/chembl_37/tdc22_cohorts"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/evidence_packet_data_axis_v5/chembl37_paper_assay_discovery.json"),
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("reports/evidence_packet_data_axis_v5/chembl37_paper_assay_discovery.tsv"),
    )
    args = parser.parse_args()
    records = []
    for task in TASKS:
        records.extend(_summarize_task(args.cohort_dir / f"{task}.tab", task))
    records.sort(
        key=lambda row: (
            row["task"],
            not row["discovery_candidate"],
            -row["unique_structures"],
            row["doi"],
            row["assay_id"],
        )
    )
    table = pd.DataFrame(records)
    summary = {}
    for task in TASKS:
        subset = table.loc[table["task"].eq(task)]
        candidates = subset.loc[subset["discovery_candidate"]]
        summary[task] = {
            "paper_assay_groups": len(subset),
            "groups_with_doi": int(subset["has_stable_doi"].sum()),
            "discovery_candidate_groups": len(candidates),
            "candidate_rows": int(candidates["rows"].sum()),
            "candidate_unique_structures_upper_bound": int(candidates["unique_structures"].sum()),
        }
    payload = {
        "protocol": "chembl37-original-paper-single-assay-discovery-v1",
        "purpose": "discovery_only_not_training_data",
        "cohort_policy": "bulk cohorts remain quarantined; one DOI and one assay per downstream packet",
        "test_labels_loaded": False,
        "summary": summary,
        "groups": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    table.to_csv(args.output_tsv, sep="\t", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
