"""Extract endpoint-conditioned external cohorts from frozen ChEMBL 37.

This script performs deterministic SQL selection and endpoint-scale mapping.
It does not inspect any TDC test file.  The resulting cohorts retain assay and
document provenance; structural and lineage firewalls are applied downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

TARGETS = {"2C9": "CHEMBL3397", "2D6": "CHEMBL289", "3A4": "CHEMBL340"}

BASE = """
select act.activity_id as ID, cs.canonical_smiles as X,
       act.standard_type, act.standard_value, act.standard_units,
       act.standard_relation, act.standard_text_value, act.activity_comment,
       ass.assay_id, ass.chembl_id as assay_chembl_id, ass.description,
       ass.assay_organism, ass.assay_cell_type, ass.assay_subcellular_fraction,
       doc.doc_id, doc.doi, doc.pubmed_id, doc.title,
       td.chembl_id as target_chembl_id, td.pref_name as target_name,
       cp.mw_freebase
from activities act
join assays ass on ass.assay_id=act.assay_id
join compound_structures cs on cs.molregno=act.molregno
left join compound_properties cp on cp.molregno=act.molregno
left join docs doc on doc.doc_id=act.doc_id
left join target_dictionary td on td.tid=ass.tid
where cs.canonical_smiles is not null
  and act.data_validity_comment is null
  and coalesce(act.potential_duplicate,0)=0
  and ({where})
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(connection, where: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(BASE.format(where=where), connection, params=params)


def _exact_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(frame["standard_value"], errors="coerce")
    relation = frame["standard_relation"].fillna("=").astype(str).str.strip()
    output = frame.loc[values.notna() & relation.isin(["=", "~"])].copy()
    output["value"] = values.loc[output.index].astype(float)
    return output


def _text(row) -> str:
    return " | ".join(
        str(value).strip().lower()
        for value in (row.standard_text_value, row.activity_comment)
        if pd.notna(value)
    )


def _caco2(frame: pd.DataFrame) -> pd.DataFrame:
    work = _exact_numeric(frame)
    units = work["standard_units"].fillna("").str.lower().str.replace(" ", "", regex=False)
    offsets = {
        "10^-6cm/s": -6, "10'-6cm/s": -6, "ucm/s": -6, "10'6cm/s": -6,
        "nm/s": -7, "cm/s": 0, "cms-1": 0,
        "10^-7cm/s": -7, "10'-7cm/s": -7, "10^-5cm/s": -5,
        "10'-5cm/s": -5, "10^-8cm/s": -8,
    }
    work["offset"] = units.map(offsets)
    work = work.loc[work["offset"].notna() & (work["value"] > 0)].copy()
    work["Y"] = work["value"].map(math.log10) + work["offset"]
    return work.loc[work["Y"].between(-12.0, -2.0)].copy()


def _solubility(frame: pd.DataFrame) -> pd.DataFrame:
    work = _exact_numeric(frame)
    kind = work["standard_type"].fillna("").str.lower()
    units = work["standard_units"].fillna("").str.lower()
    log_s = kind.eq("log s") & units.eq("")
    nm = kind.eq("solubility") & units.eq("nm") & (work["value"] > 0)
    work = work.loc[log_s | nm].copy()
    work["Y"] = work["value"]
    work.loc[nm.loc[work.index], "Y"] = work.loc[nm.loc[work.index], "value"].map(math.log10) - 9.0
    return work.loc[work["Y"].between(-15.0, 3.0)].copy()


def _identity_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    work = _exact_numeric(frame)
    work["Y"] = work["value"]
    return work


def _hia(frame: pd.DataFrame) -> pd.DataFrame:
    work = _exact_numeric(frame)
    kind = work["standard_type"].fillna("").str.lower()
    pct = pd.Series(index=work.index, dtype=float)
    pct.loc[kind.eq("hia")] = work.loc[kind.eq("hia"), "value"]
    transformed = kind.str.startswith("log10(%hia")
    pct.loc[transformed] = 10 ** work.loc[transformed, "value"] - 10.0
    work = work.loc[pct.notna()].copy()
    work["Y"] = (pct.loc[work.index] >= 30.0).astype(int)
    return work


def _bioavailability(frame: pd.DataFrame) -> pd.DataFrame:
    work = _exact_numeric(frame)
    kind = work["standard_type"].fillna("").str.lower()
    units = work["standard_units"].fillna("").str.lower()
    pct = pd.Series(index=work.index, dtype=float)
    mask_pct = kind.isin(["f", "ba"]) & units.eq("%")
    mask_fraction = kind.eq("f_fraction") & units.eq("")
    pct.loc[mask_pct] = work.loc[mask_pct, "value"]
    pct.loc[mask_fraction] = 100.0 * work.loc[mask_fraction, "value"]
    work = work.loc[pct.notna()].copy()
    work["Y"] = (pct.loc[work.index] >= 20.0).astype(int)
    return work


def _inhibition(frame: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for row in frame.itertuples(index=False):
        text = _text(row)
        kind = str(row.standard_type or "").lower()
        units = str(row.standard_units or "").lower()
        value = pd.to_numeric(pd.Series([row.standard_value]), errors="coerce").iloc[0]
        label = None
        if "not active" in text or "inactive" in text or "is not an inhibitor" in text:
            label = 0
        elif "inhibitor [+]" in text:
            label = 1
        elif kind in {"ic50", "ac50"} and units == "nm" and pd.notna(value):
            label = int(float(value) <= 10_000.0)
        labels.append(label)
    work = frame.copy()
    work["Y"] = labels
    return work.loc[work["Y"].notna()].assign(Y=lambda x: x.Y.astype(int))


def _substrate(frame: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for row in frame.itertuples(index=False):
        text = _text(row)
        if "metabolite not detected" in text or "is not a substrate" in text:
            labels.append(0)
        elif "metabolite detected" in text or "substrate [+]" in text or text == "substrate":
            labels.append(1)
        else:
            labels.append(None)
    work = frame.copy()
    work["Y"] = labels
    return work.loc[work["Y"].notna()].assign(Y=lambda x: x.Y.astype(int))


def _ames(frame: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for row in frame.itertuples(index=False):
        text = _text(row)
        if any(token in text for token in ("not toxic", "non-toxic", "non-toxic", "not active")):
            labels.append(0)
        elif text in {"toxic", "active"} or text.startswith("toxic |") or text.startswith("active |"):
            labels.append(1)
        else:
            labels.append(None)
    work = frame.copy()
    work["Y"] = labels
    return work.loc[work["Y"].notna()].assign(Y=lambda x: x.Y.astype(int))


def _ld50(frame: pd.DataFrame) -> pd.DataFrame:
    work = _exact_numeric(frame)
    units = work["standard_units"].fillna("").str.lower()
    target = pd.Series(index=work.index, dtype=float)
    umol = units.eq("umol.kg-1") & (work["value"] > 0)
    mg = units.eq("mg.kg-1") & (work["value"] > 0) & pd.to_numeric(work["mw_freebase"], errors="coerce").gt(0)
    target.loc[umol] = -work.loc[umol, "value"].map(math.log10) + 6.0
    target.loc[mg] = -(work.loc[mg, "value"] / (1000.0 * pd.to_numeric(work.loc[mg, "mw_freebase"]))).map(math.log10)
    work = work.loc[target.notna()].copy()
    work["Y"] = target.loc[work.index]
    return work


def _plans() -> dict:
    plans = {
        "Caco2_Wang": ("lower(coalesce(ass.description,'')) like '%caco%' and lower(coalesce(act.standard_type,'')) in ('papp','permeability')", (), _caco2),
        "HIA_Hou": ("lower(coalesce(ass.description,'')) like '%intestinal absorption%' and lower(coalesce(act.standard_type,'')) in ('hia','log10(%hia +10)')", (), _hia),
        "Pgp_Broccatelli": ("td.chembl_id='CHEMBL4302' and lower(coalesce(act.standard_type,'')) in ('ic50','ac50','activity','inhibition')", (), _inhibition),
        "Bioavailability_Ma": ("lower(coalesce(ass.description,'')) like '%bioavailability%' and lower(coalesce(act.standard_type,'')) in ('f','f_fraction','ba')", (), _bioavailability),
        "Lipophilicity_AstraZeneca": ("lower(coalesce(act.standard_type,'')) in ('logd','logd7.4') and (lower(coalesce(ass.description,'')) like '%logd%' or lower(coalesce(ass.description,'')) like '%lipophil%')", (), lambda f: _identity_numeric(f).loc[lambda x: x["Y"].between(-5.0, 10.0)].copy()),
        "Solubility_AqSolDB": ("lower(coalesce(ass.description,'')) like '%solub%' and lower(coalesce(act.standard_type,'')) in ('solubility','log s')", (), _solubility),
        "PPBR_AZ": ("lower(coalesce(ass.description,'')) like '%plasma protein bind%' and lower(coalesce(act.standard_type,''))='ppb' and act.standard_units='%' and ass.assay_organism='Homo sapiens'", (), lambda f: _identity_numeric(f).loc[lambda x: x["Y"].between(0.0, 100.0)].copy()),
        "VDss_Lombardo": ("lower(coalesce(ass.description,'')) like '%volume of distribution%' and lower(coalesce(act.standard_type,''))='vdss' and act.standard_units='L.kg-1' and ass.assay_organism='Homo sapiens'", (), lambda f: _identity_numeric(f).loc[lambda x: x["Y"].between(0.001, 1000.0)].copy()),
        "Half_Life_Obach": ("lower(coalesce(ass.description,'')) like '%half-life%' and lower(coalesce(act.standard_type,''))='t1/2' and act.standard_units='hr' and ass.assay_organism='Homo sapiens' and ass.assay_id<>1614674", (), lambda f: _identity_numeric(f).loc[lambda x: x["Y"].between(0.001, 2000.0)].copy()),
        "Clearance_Hepatocyte_AZ": ("lower(coalesce(ass.description,'')) like '%hepatocyte%clearance%' and lower(coalesce(act.standard_type,''))='cl' and ass.assay_organism='Homo sapiens' and act.standard_units in ('uL.min-1.(10^6cells)-1','mL.min-1.g-1')", (), lambda f: _identity_numeric(f).loc[lambda x: x["Y"].between(3.0, 150.0)].copy()),
        "Clearance_Microsome_AZ": ("lower(coalesce(ass.description,'')) like '%microsom%clearance%' and lower(coalesce(act.standard_type,''))='cl' and ass.assay_organism='Homo sapiens' and act.standard_units='mL.min-1.g-1'", (), lambda f: _identity_numeric(f).loc[lambda x: x["Y"].between(3.0, 150.0)].copy()),
        "LD50_Zhu": ("lower(coalesce(ass.description,'')) like '%ld50%' and lower(coalesce(act.standard_type,''))='ld50' and act.standard_units in ('mg.kg-1','umol.kg-1')", (), _ld50),
        "AMES": ("lower(coalesce(ass.description,'')) like '%ames%'", (), _ames),
    }
    for isoform, target in TARGETS.items():
        plans[f"CYP{isoform}_Veith"] = (f"td.chembl_id='{target}' and lower(coalesce(act.standard_type,'')) in ('ic50','ac50')", (), _inhibition)
        plans[f"CYP{isoform}_Substrate_CarbonMangels"] = (f"td.chembl_id='{target}' and lower(coalesce(act.standard_type,'')) in ('drug metabolism','activity')", (), _substrate)
    return plans


def run(args: argparse.Namespace) -> dict:
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.execute("pragma query_only=on")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "chembl37-tdc22-endpoint-conditioned-extraction-v1",
        "database": str(args.database.resolve()),
        "database_sha256": _sha256(args.archive),
        "test_labels_loaded": False,
        "cohorts": {},
    }
    for task, (where, params, transform) in _plans().items():
        raw = _read(connection, where, params)
        cohort = transform(raw)
        keep = [
            "ID", "X", "Y", "standard_type", "standard_units", "standard_relation",
            "assay_id", "assay_chembl_id", "description", "assay_organism",
            "assay_cell_type", "assay_subcellular_fraction", "doc_id", "doi",
            "pubmed_id", "title", "target_chembl_id", "target_name",
        ]
        cohort = cohort[[column for column in keep if column in cohort]].copy()
        output = args.output_dir / f"{task}.tab"
        cohort.to_csv(output, sep="\t", index=False)
        manifest["cohorts"][task] = {
            "sql_candidate_rows": len(raw),
            "mapped_rows": len(cohort),
            "mapped_identities_before_structure_standardization": int(cohort["X"].nunique()),
            "positive_rate": float(cohort["Y"].mean()) if task not in {
                "Caco2_Wang", "Lipophilicity_AstraZeneca", "Solubility_AqSolDB",
                "PPBR_AZ", "VDss_Lombardo", "Half_Life_Obach",
                "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ", "LD50_Zhu",
            } and len(cohort) else None,
            "path": str(output),
            "sha256": _sha256(output),
        }
        print(json.dumps({"task": task, **manifest["cohorts"][task]}), flush=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/external/chembl/chembl_37")
    parser.add_argument("--database", type=Path, default=root / "chembl_37/chembl_37_sqlite/chembl_37.db")
    parser.add_argument("--archive", type=Path, default=root / "chembl_37_sqlite.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "tdc22_cohorts")
    parser.add_argument("--manifest", type=Path, default=root / "tdc22_cohorts_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps({"tasks": len(payload["cohorts"])}, indent=2))
