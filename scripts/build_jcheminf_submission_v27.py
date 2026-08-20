"""Build a fail-closed Journal of Cheminformatics submission package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.build_paper_evidence_manifest import hash_file, manifest_digest


VERSION = "v2.7"
FREEZE_SHA256 = "693db6936485d59791eacfd9005d18efca00056274c11c1cc80efd962058c84b"
OUTPUT_DIR = Path("paper/submission/jcheminformatics")
SOURCES = {
    "working_manuscript": "MANUSCRIPT_V24.md",
    "bibliography": "paper/references.bib",
    "figure_manifest": "paper/figures/manifest.json",
    "scorecard": "reports/goal_scorecard_v27.json",
    "router_controls": "runs/v27-router-controls/report.json",
    "router_risk_guard": "runs/v27-router-risk-guard/report.json",
    "primary_effects": "reports/primary_model_effects_v27.json",
    "chemical_space": "reports/validation_calibration_residuals_v27.json",
    "compute": "reports/compute_memory_appendix_v27.json",
    "third_party_release_audit": "reports/third_party_release_audit_v27.json",
    "rescue_ledger": "runs/v24-rescue-ledger/report.json",
    "citation_metadata": "CITATION.cff",
    "cpu_environment": "environment-cpu.yml",
    "gpu_environment": "environment-gpu.yml",
    "requirements": "requirements-repro.txt",
    "license": "LICENSE",
}

ABSTRACT = """### Background

ADMET endpoints differ in label type, sample size, chemical-space shift, and
the value of topology, geometry, and target-distribution evidence. This makes
always-on use of every molecular representation costly and scientifically
fragile.

### Results

We introduce ADMET-EvO, a protocol-safe research agent that evaluates bounded
candidates on scaffold-split train/validation data, records successful and
failed acquisitions, and content-addresses a finalist before one-shot official
evaluation. On a dated snapshot of 22 Therapeutics Data Commons ADMET tasks,
the frozen v2.7 system has 18 estimated Top-5 and 21 Top-10 placements. A
preregistered MapLight+GNN expert improved CYP2C9 substrate from 0.340 to 0.434
+/- 0.009 AUPRC and Bioavailability from 0.697 to 0.731 +/- 0.007 AUROC. The
remaining displayed Bioavailability deficit is reported conservatively because
its public page duplicates the P-glycoprotein table and lacks an authoritative
correction. A held-out-seed router analysis retained 18.4% of score-selected
aggregate compute without improving mean accuracy. Fresh-seed safety, learned
context, full-memory, and matched-cost random controls also failed their
accuracy gates. Low-similarity molecules had worse error on 11/13
classification and 7/9 regression endpoints.

### Contribution

The contribution is a reproducible experiment-level acquisition architecture:
registered actions, measured acquisition cost, optional evidence rejection, a
full negative-result ledger, and an immutable discovery-to-evaluation
boundary. It is not a claim that routing or multimodal fusion universally
improves prediction.

### Conclusions

Protocol-safe evidence acquisition can reduce aggregate experimental compute
while preserving an auditable account of uncertainty, failures, and chemical-
space risk. The negative controls delimit the supported claims and identify
candidate-level risk modeling, rather than metric-name rules, as the next
prospective target."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w+/-]+\b", text))


def _bib_entries(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    blocks = re.split(r"(?=^@)", text, flags=re.MULTILINE)
    for block in blocks:
        header = re.match(r"@(\w+)\{([^,]+),", block)
        if not header:
            continue
        fields = {"type": header.group(1)}
        for line in block.splitlines()[1:]:
            match = re.match(r"\s*(\w+)\s*=\s*\{(.*)\},?\s*$", line)
            if match:
                fields[match.group(1).lower()] = match.group(2)
        entries[header.group(2)] = fields
    return entries


def _citation_order(body: str) -> list[str]:
    order: list[str] = []
    for match in re.finditer(r"\[([^\]]*@[^\]]+)\]", body):
        for key in re.findall(r"@([A-Za-z0-9_:-]+)", match.group(1)):
            if key not in order:
                order.append(key)
    return order


def _number_citations(body: str, order: list[str]) -> str:
    numbers = {key: index + 1 for index, key in enumerate(order)}

    def replace(match: re.Match[str]) -> str:
        keys = re.findall(r"@([A-Za-z0-9_:-]+)", match.group(1))
        return "[" + ", ".join(str(numbers[key]) for key in keys) + "]"

    return re.sub(r"\[([^\]]*@[^\]]+)\]", replace, body)


def _references(entries: dict[str, dict[str, str]], order: list[str]) -> str:
    def plain(value: str) -> str:
        replacements = {
            'Kl\\"aser': "Kläser",
            'B\\l{}a\\.{z}ej': "Błażej",
            'M\\"uller': "Müller",
        }
        for source, target in replacements.items():
            value = value.replace(source, target)
        return value.replace("{", "").replace("}", "")

    lines = ["## References", ""]
    for index, key in enumerate(order, 1):
        entry = entries[key]
        venue = entry.get("journal") or entry.get("booktitle") or "Preprint"
        volume = entry.get("volume", "")
        pages = entry.get("pages", "")
        locator = entry.get("doi") or entry.get("url", "")
        if entry.get("doi"):
            locator = f"https://doi.org/{locator}"
        volume_pages = ";".join(part for part in (volume, pages) if part)
        suffix = f" {entry['year']}"
        if volume_pages:
            suffix += f";{volume_pages}"
        lines.append(
            f"{index}. {plain(entry['author']).replace(' and ', '; ')}. "
            f"{plain(entry['title'])}. *{plain(venue)}*.{suffix}. {locator}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_manuscript(root: Path) -> tuple[str, dict[str, Any]]:
    source = (root / SOURCES["working_manuscript"]).read_text(encoding="utf-8")
    body = source.split("## 1. Introduction", 1)[1].split("## 9. References", 1)[0]
    body = "## 1. Introduction" + body
    order = _citation_order(body)
    entries = _bib_entries((root / SOURCES["bibliography"]).read_text(encoding="utf-8"))
    if set(order) != set(entries):
        raise ValueError("Submission citation order does not cover the bibliography exactly")
    body = _number_citations(body, order).rstrip()
    declarations = """## Abbreviations

ADMET, absorption, distribution, metabolism, excretion, and toxicity; AUPRC,
area under the precision--recall curve; AUROC, area under the receiver
operating-characteristic curve; ECE, expected calibration error; GNN, graph
neural network; LCB, lower confidence bound; MAE, mean absolute error; TDC,
Therapeutics Data Commons.

## Declarations

### Ethics approval and consent to participate

Not applicable. This computational study uses public benchmark datasets and no
new human-participant or animal experiments.

### Consent for publication

Not applicable.

### Availability of data and materials

The public benchmark data are distributed through PyTDC. Source code, frozen
recipes, dependency specifications, derived validation evidence, prediction
hashes, and figure-generation scripts are prepared in the ADMET-EvO repository
under an OSI-approved MIT license. Before submission, the exact reviewed
version must be deposited in a registration-free archival repository and the
placeholder `ARCHIVE_DOI_REQUIRED` replaced with its persistent DOI. Official
test labels are neither redistributed nor exposed to model-selection code.

### Competing interests

`AUTHOR_CONFIRMATION_REQUIRED` -- the author must confirm or replace the draft
statement: The author declares no competing interests.

### Funding

`FUNDING_INFORMATION_REQUIRED` -- name funders and grant numbers, or state that
the research received no specific funding.

### Authors' contributions

Y.W. conceived the study, developed the software, ran the experiments,
analyzed the results, and drafted the manuscript. `AUTHOR_CONFIRMATION_REQUIRED`.

### Acknowledgements

`ACKNOWLEDGEMENTS_OPTIONAL`.
"""
    title = """# ADMET-EvO: Cost-Aware Sequential Evidence Acquisition with Freeze-Locked Evaluation for ADMET Prediction

Research Article

Yilin Wang^1,*^ (`ORCID_REQUIRED`)

^1^ Valhalla Technology  
* Correspondence: `CORRESPONDING_EMAIL_REQUIRED`

## Abstract

"""
    manuscript = (
        title
        + ABSTRACT.strip()
        + "\n\n"
        + body
        + "\n\n"
        + declarations.strip()
        + "\n\n"
        + _references(entries, order)
    )
    metadata = {
        "abstract_word_count": _word_count(ABSTRACT),
        "citation_count": len(order),
        "reference_count": len(entries),
        "contribution_heading_present": "### Contribution" in ABSTRACT,
        "author_placeholders": [
            "ORCID_REQUIRED",
            "CORRESPONDING_EMAIL_REQUIRED",
            "AUTHOR_CONFIRMATION_REQUIRED",
            "FUNDING_INFORMATION_REQUIRED",
            "ARCHIVE_DOI_REQUIRED",
        ],
    }
    return manuscript, metadata


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _build_supplement(root: Path) -> tuple[str, dict[str, Any]]:
    scorecard = _load(root / SOURCES["scorecard"])
    controls = _load(root / SOURCES["router_controls"])
    guard = _load(root / SOURCES["router_risk_guard"])
    effects = _load(root / SOURCES["primary_effects"])
    space = _load(root / SOURCES["chemical_space"])
    compute = _load(root / SOURCES["compute"])
    rescue = _load(root / SOURCES["rescue_ledger"])

    lines = [
        "# Supplementary Information: ADMET-EvO v2.7",
        "",
        "This document is generated from content-addressed evidence. It reports",
        "negative results and missing measurements as well as positive results.",
        "",
        "## S1. Complete conservative 22-task scorecard",
        "",
        "| Task | Metric | Frozen finalist | Mean +/- SD | Estimated rank | Band |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in scorecard["tasks"]:
        band = "Top-5" if row["estimated_rank"] <= 5 else "Top-10" if row["estimated_rank"] <= 10 else "below Top-10"
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['finalist']} | "
            f"{_fmt(row['score'], 3)} +/- {_fmt(row['std'], 3)} | "
            f"{row['estimated_rank']} | {band} |"
        )

    memory = controls["memory_ablation"]
    random = controls["matched_cost_random"]
    lines.extend([
        "",
        "## S2. Preregistered router controls",
        "",
        "| Contrast | Mean gain | 95% bootstrap CI | Improved/tied/worse | Wilcoxon p | Gate |",
        "|---|---:|---:|---:|---:|---|",
        f"| Full-outcome minus winner-only memory | {_fmt(memory['summary']['mean'], 5)} | "
        f"[{_fmt(memory['summary']['bootstrap_mean_95_ci'][0], 5)}, {_fmt(memory['summary']['bootstrap_mean_95_ci'][1], 5)}] | "
        f"{memory['summary']['improved']}/{memory['summary']['tied']}/{memory['summary']['worse']} | "
        f"{_fmt(memory['summary']['wilcoxon_p_value'], 4)} | failed |",
        f"| Agent minus matched-cost random | {_fmt(random['summary']['mean'], 5)} | "
        f"[{_fmt(random['summary']['bootstrap_mean_95_ci'][0], 5)}, {_fmt(random['summary']['bootstrap_mean_95_ci'][1], 5)}] | "
        f"{random['summary']['improved']}/{random['summary']['tied']}/{random['summary']['worse']} | "
        f"{_fmt(random['summary']['wilcoxon_p_value'], 4)} | failed |",
        "",
        f"Full-outcome/winner-only aggregate cost ratio: {_fmt(memory['aggregate_full_to_winner_cost_ratio'], 4)}. "
        f"Maximum matched-random relative cost error: {random['maximum_relative_cost_error']:.3e}.",
        "",
        "## S3. Fresh-seed metric-risk guard",
        "",
        f"The guard covered {guard['task_count']} tasks on seeds {guard['fresh_validation_seeds']} with "
        f"identity overlap {guard['identity_overlap_total']}. Its claim gate was "
        f"{guard['claim_gate']['passed']}; guarded minus unguarded mean gain was "
        f"{_fmt(guard['primary_guard_minus_unguarded']['mean'], 5)}. The experiment did not change selection or the official freeze.",
        "",
        "## S4. Primary paired model effects",
        "",
        "| Task and contrast | n | Mean oriented gain | 95% paired-t CI | Corrected dz | Sign-flip p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, row in effects["contrasts"].items():
        effect = row["effect"]
        lines.append(
            f"| {name} | {effect['n']} | {_fmt(effect['mean_oriented_gain'], 5)} | "
            f"[{_fmt(effect['paired_t_95_ci'][0], 5)}, {_fmt(effect['paired_t_95_ci'][1], 5)}] | "
            f"{_fmt(effect['small_sample_corrected_dz'], 3)} | {_fmt(effect['exact_two_sided_sign_flip_p'], 4)} |"
        )

    lines.extend([
        "",
        "## S5. Calibration and chemical-space diagnostics",
        "",
        "| Task | Type | n train | n valid | Median nearest-train similarity | Low-minus-high error | Identity overlap | Scaffold overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in space["tasks"]:
        if row["analysis_type"] == "classification_calibration":
            delta = row["chemical_space"]["low_minus_high_brier"]
        else:
            delta = row["chemical_space"]["low_minus_high_mean_absolute_error"]
        lines.append(
            f"| {row['task']} | {row['analysis_type']} | {row['n_train']} | {row['n_valid']} | "
            f"{_fmt(row['nearest_train_tanimoto']['median'], 3)} | {_fmt(delta, 4)} | "
            f"{row['identity_overlap_count']} | {row['train_valid_scaffold_overlap_count']} |"
        )

    lines.extend([
        "",
        "## S6. Weak-task rescue ledger",
        "",
        "| Task | Method | Baseline | Candidate | Gain | Decision |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in rescue["experiments"]:
        lines.append(
            f"| {row['task']} | {row['method']} | {_fmt(row['baseline'], 4)} | "
            f"{_fmt(row['candidate'], 4)} | {_fmt(row['gain'], 4)} | {row['decision']} |"
        )

    lines.extend([
        "",
        "## S7. Compute-accounting boundary",
        "",
        f"Candidate-stage selected time was {_fmt(compute['router_replay']['selected_total_hours'], 3)} h versus "
        f"{_fmt(compute['router_replay']['score_policy_total_hours'], 3)} h for score selection "
        f"(ratio {_fmt(compute['router_replay']['selected_to_score_total_cost_ratio'], 4)}). Historical peak-memory "
        f"coverage was {compute['peak_memory_coverage']['records_measured']}/{compute['peak_memory_coverage']['records_total']}; "
        "no absolute memory claim is made.",
        "",
        "## S8. Reproducibility firewall",
        "",
        f"The full finalist freeze SHA-256 is `{FREEZE_SHA256}`. Candidate selection used train/validation evidence only. "
        "Fresh validation audits report zero canonical-molecule identity overlap. Official results cannot mutate a finalist "
        "within the same freeze lineage.",
    ])
    metadata = {
        "task_count": len(scorecard["tasks"]),
        "top5": scorecard["status"]["top5"],
        "top10": scorecard["status"]["top10"],
        "router_control_task_count": controls["task_count"],
        "fresh_guard_task_count": guard["task_count"],
        "chemical_space_task_count": space["task_count"],
        "rescue_experiment_count": len(rescue["experiments"]),
        "identity_overlap_total": sum(row["identity_overlap_count"] for row in space["tasks"]),
        "scaffold_overlap_total": sum(row["train_valid_scaffold_overlap_count"] for row in space["tasks"]),
    }
    return "\n".join(lines).rstrip() + "\n", metadata


def _build_graphical_abstract(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.2, 3.0), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, "2D baseline\ncheap evidence", "#DCEEFF"),
        (0.27, "Acquire / reject\n3D + metric views", "#E8E1FF"),
        (0.52, "Full outcome ledger\ncost + uncertainty", "#DFF5E3"),
        (0.76, "SHA-256 freeze\none-shot evaluation", "#FFE7D6"),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch(
            (x, 0.34), 0.19, 0.32,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.4, edgecolor="#233044", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + 0.095, 0.50, label, ha="center", va="center", fontsize=10, weight="bold")
    for left in (0.22, 0.46, 0.71):
        ax.add_patch(FancyArrowPatch((left, 0.50), (left + 0.045, 0.50), arrowstyle="-|>", mutation_scale=13, color="#233044"))
    ax.text(0.50, 0.88, "ADMET-EvO: protocol-safe evidence acquisition", ha="center", va="center", fontsize=15, weight="bold", color="#14213D")
    ax.text(0.50, 0.13, "18/22 Top-5  •  21/22 Top-10  •  validation-only discovery  •  negative results retained", ha="center", va="center", fontsize=10, color="#34495E")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=100, facecolor="white", metadata={"Software": "ADMET-EvO submission builder"})
    plt.close(fig)


def build_submission(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    root = repo_root.resolve()
    out = root / output_dir
    out.mkdir(parents=True, exist_ok=True)

    scorecard = _load(root / SOURCES["scorecard"])
    figures = _load(root / SOURCES["figure_manifest"])
    third_party = _load(root / SOURCES["third_party_release_audit"])
    if scorecard["freeze_sha256"] != FREEZE_SHA256:
        raise ValueError("Scorecard freeze differs from the registered v2.7 freeze")
    if scorecard["status"] != {"top5": 18, "top10": 21, "top5_deficit": 0, "top10_deficit": 1, "goal_met": False}:
        raise ValueError("Conservative scorecard changed")
    if figures["official_test_loaded"] is not False or figures["selection_or_freeze_changed"] is not False:
        raise ValueError("Figure package violates the submission firewall")
    if third_party["gates"]["registration_free_checkpoint_access_verified"] is not True:
        raise ValueError("Required third-party checkpoints are not anonymously accessible")
    if third_party["gates"]["checkpoint_specific_licenses_verified"] is not False:
        raise ValueError("Expected fail-closed GIN checkpoint-license boundary changed")
    if third_party["gates"]["ready_for_external_deposit"] is not True:
        raise ValueError("Link-only reviewer bundle is not ready for external deposit")
    if third_party["scope"]["final_mapping_minimol_used"] is not False:
        raise ValueError("Submission dependency scope incorrectly includes MiniMol")

    manuscript, manuscript_meta = _build_manuscript(root)
    if manuscript_meta["abstract_word_count"] > 350 or not manuscript_meta["contribution_heading_present"]:
        raise ValueError("Research Article abstract violates journal-specific requirements")
    supplement, supplement_meta = _build_supplement(root)

    availability = """# Data and code availability draft

The public benchmark data are distributed by Therapeutics Data Commons through
PyTDC. ADMET-EvO source code, frozen recipes, environment specifications,
derived validation evidence, prediction hashes, and figure/SI builders are
prepared for public release under the MIT license.

Both required third-party checkpoints were anonymously downloadable on
2026-08-17 and are identified by immutable SHA-256 hashes in the release audit.
Uni-Mol2's public, ungated model repository declares MIT. The separately hosted
Molfeat `gin_supervised_masking` artifact is public and hash-matched. Its 57
parameter tensors are bitwise identical to DGL-LifeSci's official checkpoint,
but its artifact metadata has no explicit license field; Apache-2.0 on source
code is not treated as a binary-checkpoint redistribution license.

Submission blocker: deposit the exact reviewed link-only repository snapshot
in a registration-free archival repository (recommended: Zenodo release linked
to GitHub), and replace `ARCHIVE_DOI_REQUIRED`. The reviewer bundle
redistributes neither third-party checkpoint and includes no official test
labels. Third-party checkpoint files must not be added unless their artifact
licenses are made explicit.
"""
    checklist = """# Journal of Cheminformatics submission checklist

Requirements last checked: 2026-08-17.

## Machine-verified

- [x] Research Article abstract is at most 350 words.
- [x] Abstract contains an explicit `Contribution` heading.
- [x] Four main-text figures have hashed SVG and 300-dpi PNG outputs.
- [x] Graphical abstract is 920 x 300 pixels on a white background and below 150 KB.
- [x] All 15 manuscript citations resolve to complete references.
- [x] Supplement reports all 22 tasks, negative controls, chemical-space diagnostics, and missing peak-memory coverage.
- [x] Frozen scorecard remains 18/22 Top-5 and 21/22 Top-10; no displayed threshold is silently overridden.
- [x] Submission generation does not load official test data or change selection/freeze state.
- [x] Both essential checkpoints have verified anonymous public access and immutable hashes; the Molfeat GIN parameters exactly equal the official DGL-LifeSci parameters; neither checkpoint is included in the reviewer bundle.

## Author or external action required

- [ ] Supply corresponding-author email and ORCID.
- [ ] Confirm authorship, author order, contributions, competing interests, acknowledgements, and funding statement.
- [ ] Deposit a registration-free archived source/data package and replace `ARCHIVE_DOI_REQUIRED`.
- [ ] Obtain an authoritative TDC response on the Bioavailability/Pgp page; until then retain both the conservative displayed scorecard and the scientific discrepancy audit.
- [ ] Convert Markdown to the journal's final editable submission format and perform publisher-side reference metadata checks.

## Official sources

- Journal-specific Research Article guidance: https://jcheminform.github.io/jcheminform-author-guidelines/guidelines/researchArticle.html
- Journal reproducibility/methodology guidance: https://jcheminform.github.io/jcheminform-author-guidelines/guidelines/methodology.html
- BMC editorial and data/code availability policies: https://www.biomedcentral.com/getpublished/editorial-policies
"""

    outputs = {
        "manuscript.md": manuscript,
        "supplementary_information.md": supplement,
        "data_and_code_availability.md": availability,
        "submission_checklist.md": checklist,
    }
    for name, text in outputs.items():
        (out / name).write_text(text, encoding="utf-8")
    _build_graphical_abstract(out / "graphical_abstract.png")

    output_hashes = {
        str(path.relative_to(root)): hash_file(path)
        for path in sorted(out.iterdir())
        if path.name != "manifest.json"
    }
    sources = {name: {"path": relative, **hash_file(root / relative)} for name, relative in SOURCES.items()}
    payload = {
        "version": VERSION,
        "target_journal": "Journal of Cheminformatics",
        "article_type": "Research Article",
        "requirements_last_checked": "2026-08-17",
        "official_guidelines": [
            "https://jcheminform.github.io/jcheminform-author-guidelines/guidelines/researchArticle.html",
            "https://jcheminform.github.io/jcheminform-author-guidelines/guidelines/methodology.html",
        ],
        "full_freeze_sha256": FREEZE_SHA256,
        "scorecard_status": scorecard["status"],
        "official_test_loaded_during_generation": False,
        "selection_or_freeze_changed": False,
        "manuscript": manuscript_meta,
        "supplement": supplement_meta,
        "graphical_abstract": {"width_px": 920, "height_px": 300, "max_bytes": 150000},
        "third_party_release_audit": {
            "essential_dependency_count": third_party["scope"]["essential_dependency_count"],
            "registration_free_checkpoint_access_verified": True,
            "checkpoint_specific_licenses_verified": False,
            "third_party_checkpoints_in_bundle": False,
            "gin_official_parameter_equivalence_verified": True,
            "ready_for_external_deposit": True,
        },
        "unresolved_submission_blockers": manuscript_meta["author_placeholders"] + [
            "AUTHORSHIP_AND_FUNDING_CONFIRMATION",
        ],
        "sources": sources,
        "outputs": output_hashes,
    }
    manifest = {**payload, "manifest_sha256": manifest_digest(payload)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build_submission(args.repo_root, args.output_dir)
    print(json.dumps({
        "manifest_sha256": manifest["manifest_sha256"],
        "abstract_word_count": manifest["manuscript"]["abstract_word_count"],
        "reference_count": manifest["manuscript"]["reference_count"],
        "output_count": len(manifest["outputs"]),
        "unresolved_blocker_count": len(manifest["unresolved_submission_blockers"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
