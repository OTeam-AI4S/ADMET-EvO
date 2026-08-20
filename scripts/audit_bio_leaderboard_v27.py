"""Build an evidence-gated audit of the TDC Bioavailability leaderboard.

The audit intentionally separates a strong discrepancy finding from an
authoritative correction.  It must never silently replace the displayed TDC
threshold or turn the 22-task goal green without an official correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BIO = "Bioavailability_Ma"
PGP = "Pgp_Broccatelli"
CYP = "CYP2C9_Substrate_CarbonMangels"
AUDIT_REPOSITORY_COMMIT = "3e5472ce159a29b872219ffaaa4a11b4a2660f77"
AUDIT_PAPER = {
    "title": (
        "An End-User Audit of Reproducibility, Data Leakage, and Overfitting "
        "of the Top-Ranked ADMET Prediction Models in TDC Leaderboards"
    ),
    "doi": "10.1021/acs.jcim.6c00819",
    "url": "https://pubs.acs.org/doi/10.1021/acs.jcim.6c00819",
    "published_maplight": {"mean": 0.741, "std": 0.007},
    "published_maplight_gnn": {"mean": 0.731, "std": 0.007},
    "table": 2,
    "extraction": "manual_transcription_from_published_table_2",
}
MINIMOL_PRIMARY_REPORT = {
    "repository": "https://github.com/graphcore-research/minimol",
    "commit_observed_at_audit": "41642328affbb35e4950b7b7a9020aa95919b503",
    "as_of": "June 2024",
    "bio": {
        "tdc_sota": {"mean": 0.748, "std": 0.033},
        "minimol": {"mean": 0.689, "std": 0.020, "rank": 5},
    },
    "pgp": {
        "tdc_sota": {"mean": 0.938, "std": 0.002},
        "minimol": {"mean": 0.942, "std": 0.002, "rank": 1},
    },
    "extraction": "manual_transcription_from_upstream_README_performance_table",
}
TDC_PUBLIC_ISSUE = {
    "number": 394,
    "url": "https://github.com/mims-harvard/TDC/issues/394",
    "title": (
        "[Bug] Score discrepancy and data contamination in "
        "Bioavailability_Ma leaderboard"
    ),
    "created_at": "2026-06-08T04:13:44Z",
    "state_observed_at_audit": "open",
    "maintainer_comments_observed_at_audit": 0,
    "meaning": (
        "The same Bio/Pgp mapping problem was publicly reported before this "
        "work, but an open user report without a maintainer response is not an "
        "authoritative correction."
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_entries(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": str(row["model"]),
            "mean": float(row.get("mean", row.get("metric_mean"))),
            "std": float(row.get("std", row.get("metric_std"))),
        }
        for row in page["entries"]
    ]


def _entry_key(row: dict[str, Any]) -> tuple[str, float, float]:
    return row["model"], row["mean"], row["std"]


def _official_result(payload: dict[str, Any]) -> tuple[float, float]:
    values = next(iter(payload["results"].values()))
    return float(values[0]), float(values[1])


def build_audit(repo_root: Path, snapshot_path: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    snapshot = _load(snapshot_path)
    frozen_path = root / "vendor/tdc-admet-bench-audit/data/tdc_admet_leaderboard.json"
    frozen = _load(frozen_path)
    scorecard_path = root / "reports/goal_scorecard_v27.json"
    scorecard = _load(scorecard_path)
    confirmation_path = root / "runs/v27-maplight-gnn-confirmation/result.json"
    confirmation = _load(confirmation_path)
    bio_result_path = root / "runs/v27-official/Bioavailability_Ma/final/results.json"
    cyp_result_path = (
        root
        / "runs/v27-official/CYP2C9_Substrate_CarbonMangels/final/results.json"
    )
    bio_result = _load(bio_result_path)
    cyp_result = _load(cyp_result_path)

    bio_entries = _page_entries(snapshot["pages"][BIO])
    pgp_entries = _page_entries(snapshot["pages"][PGP])
    bio_set = {_entry_key(row) for row in bio_entries}
    pgp_set = {_entry_key(row) for row in pgp_entries}
    shared = bio_set & pgp_set
    union = bio_set | pgp_set
    frozen_bio = {_entry_key(row) for row in _page_entries(frozen["bioavailability_ma"])}
    frozen_pgp = {_entry_key(row) for row in _page_entries(frozen["pgp_broccatelli"])}

    bio_mean, bio_std = _official_result(bio_result)
    cyp_mean, cyp_std = _official_result(cyp_result)
    paper_bio = AUDIT_PAPER["published_maplight_gnn"]
    cyp_board = next(
        row
        for row in snapshot["pages"][CYP]["entries"]
        if row["model"] == "MapLight + GNN"
    )
    fresh_bio = confirmation["tasks"][BIO]["summary"]
    identity_audit = confirmation["identity_audit"][BIO]
    bio_scorecard = next(row for row in scorecard["tasks"] if row["task"] == BIO)
    current_bio_minimol = next(
        row for row in bio_entries if row["model"] == "MiniMol"
    )

    checks = {
        "all_pgp_rows_exactly_contained_in_bio": shared == pgp_set,
        "bio_pgp_have_different_dataset_sizes": (
            snapshot["pages"][BIO]["size"] != snapshot["pages"][PGP]["size"]
        ),
        "current_snapshot_matches_audit_repository_bio": bio_set == frozen_bio,
        "current_snapshot_matches_audit_repository_pgp": pgp_set == frozen_pgp,
        "local_bio_exactly_matches_published_reproduction": (
            bio_mean == paper_bio["mean"] and bio_std == paper_bio["std"]
        ),
        "current_bio_minimol_exactly_equals_historical_pgp_minimol": (
            current_bio_minimol["mean"]
            == MINIMOL_PRIMARY_REPORT["pgp"]["minimol"]["mean"]
            and current_bio_minimol["std"]
            == MINIMOL_PRIMARY_REPORT["pgp"]["minimol"]["std"]
        ),
        "historical_bio_and_pgp_scores_are_distinct": (
            MINIMOL_PRIMARY_REPORT["bio"]["minimol"]["mean"]
            != MINIMOL_PRIMARY_REPORT["pgp"]["minimol"]["mean"]
        ),
        "cyp_positive_control_within_0_01": abs(cyp_mean - float(cyp_board["mean"]))
        <= 0.01,
        "confirmation_never_loaded_official_test": (
            confirmation["protocol"]["official_test_loaded"] is False
            and confirmation["protocol"]["test_labels_used"] is False
        ),
        "confirmation_has_zero_identity_overlap": all(
            int(row["cross_split_identity_overlap"]) == 0
            for row in identity_audit.values()
        ),
        "goal_scorecard_remains_unmet": scorecard["status"]["goal_met"] is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Bio leaderboard audit failed invariant(s): {failed}")

    return {
        "schema_version": "bio-leaderboard-discrepancy-v27-1",
        "snapshot": {
            "captured_at": snapshot["captured_at"],
            "path": snapshot_path.resolve().relative_to(root).as_posix(),
            "sha256": _sha256(snapshot_path),
            "bio_url": snapshot["pages"][BIO]["url"],
            "pgp_url": snapshot["pages"][PGP]["url"],
        },
        "audit_repository": {
            "commit": AUDIT_REPOSITORY_COMMIT,
            "leaderboard_path": frozen_path.relative_to(root).as_posix(),
            "leaderboard_sha256": _sha256(frozen_path),
            "url": "https://github.com/receptor-ai/tdc-admet-bench",
        },
        "published_audit": AUDIT_PAPER,
        "historical_primary_report": MINIMOL_PRIMARY_REPORT,
        "tdc_public_issue": TDC_PUBLIC_ISSUE,
        "cross_page_evidence": {
            "bio_dataset_size": int(snapshot["pages"][BIO]["size"]),
            "pgp_dataset_size": int(snapshot["pages"][PGP]["size"]),
            "bio_entry_count": len(bio_set),
            "pgp_entry_count": len(pgp_set),
            "exact_shared_model_mean_std": len(shared),
            "pgp_containment": len(shared) / len(pgp_set),
            "jaccard": len(shared) / len(union),
            "identical_rows": [
                {"model": model, "mean": mean, "std": std}
                for model, mean, std in sorted(shared)
            ],
            "same_values_in_audit_repository_snapshot": True,
            "current_bio_minimol": current_bio_minimol,
            "historical_bio_minimol": MINIMOL_PRIMARY_REPORT["bio"]["minimol"],
            "historical_pgp_minimol": MINIMOL_PRIMARY_REPORT["pgp"]["minimol"],
        },
        "local_reproduction": {
            "bio": {
                "method": "MapLight + GNN",
                "mean": bio_mean,
                "std": bio_std,
                "result_path": bio_result_path.relative_to(root).as_posix(),
                "result_sha256": _sha256(bio_result_path),
                "published_reproduction_absolute_delta": abs(
                    bio_mean - paper_bio["mean"]
                ),
                "fresh_validation_mean": float(fresh_bio["maplight_gnn_mean"]),
                "validation_to_official_delta": bio_mean
                - float(fresh_bio["maplight_gnn_mean"]),
                "displayed_tdc_maplight_gnn": next(
                    row["mean"]
                    for row in bio_entries
                    if row["model"] == "MapLight + GNN"
                ),
            },
            "cyp_positive_control": {
                "method": "MapLight + GNN",
                "mean": cyp_mean,
                "std": cyp_std,
                "displayed_tdc_mean": float(cyp_board["mean"]),
                "absolute_delta": abs(cyp_mean - float(cyp_board["mean"])),
                "result_path": cyp_result_path.relative_to(root).as_posix(),
                "result_sha256": _sha256(cyp_result_path),
            },
            "confirmation_path": confirmation_path.relative_to(root).as_posix(),
            "confirmation_sha256": _sha256(confirmation_path),
            "official_test_loaded_during_selection": False,
            "test_labels_used_during_selection": False,
            "cross_split_identity_overlap": 0,
        },
        "goal_effect": {
            "displayed_top10_threshold": float(bio_scorecard["top10_threshold"]),
            "displayed_estimated_rank": int(bio_scorecard["estimated_rank"]),
            "historical_june_2024_rank_upper_bound": (
                MINIMOL_PRIMARY_REPORT["bio"]["minimol"]["rank"]
                if bio_mean > MINIMOL_PRIMARY_REPORT["bio"]["minimol"]["mean"]
                else None
            ),
            "historical_reference_is_not_current_rank_evidence": True,
            "verified_top5_count": int(scorecard["status"]["top5"]),
            "verified_top10_count": int(scorecard["status"]["top10"]),
            "goal_met": False,
            "threshold_override_allowed": False,
        },
        "checks": checks,
        "assessment": {
            "status": "confirmed_material_leaderboard_discrepancy",
            "most_supported_explanation": (
                "The displayed Bioavailability rows are inconsistent with current-data "
                "reproduction and are exactly copied from the distinct Pgp endpoint for "
                "all 16 Pgp entries. Dataset/version or leaderboard-record corruption is "
                "therefore substantially better supported than a local implementation failure."
            ),
            "root_cause_proven": False,
            "authoritative_correction_obtained": False,
            "policy": (
                "Do not replace the displayed 0.922 Top-10 threshold and do not mark the "
                "22-task goal complete until TDC issues an authoritative correction or a "
                "legitimate frozen model reaches the displayed threshold."
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    cross = report["cross_page_evidence"]
    bio = report["local_reproduction"]["bio"]
    cyp = report["local_reproduction"]["cyp_positive_control"]
    goal = report["goal_effect"]
    paper = report["published_audit"]
    historical = report["historical_primary_report"]
    issue = report["tdc_public_issue"]
    lines = [
        "# Bioavailability leaderboard discrepancy audit (v2.7)",
        "",
        f"Status: **{report['assessment']['status']}**",
        "",
        "This is a confirmed material discrepancy, not an authoritative leaderboard "
        "correction. The displayed threshold remains binding for the goal scorecard.",
        "",
        "## Independent evidence", 
        "",
        "| Evidence | Observation |",
        "|---|---|",
        (
            "| Cross-page exact duplication | "
            f"{cross['exact_shared_model_mean_std']}/{cross['pgp_entry_count']} Pgp rows "
            f"are exact `(model, mean, std)` matches on Bio; containment "
            f"{cross['pgp_containment']:.3f}, Jaccard {cross['jaccard']:.3f}. |"
        ),
        (
            "| Distinct data sets | Bio has "
            f"{cross['bio_dataset_size']} molecules; Pgp has {cross['pgp_dataset_size']}. |"
        ),
        (
            "| Published independent reproduction | MapLight+GNN Bio "
            f"{paper['published_maplight_gnn']['mean']:.3f} +/- "
            f"{paper['published_maplight_gnn']['std']:.3f} (Table {paper['table']}). |"
        ),
        (
            "| Historical primary report | MiniMol's upstream June 2024 table reports "
            f"Bio {historical['bio']['minimol']['mean']:.3f} +/- "
            f"{historical['bio']['minimol']['std']:.3f} (rank "
            f"{historical['bio']['minimol']['rank']}) but Pgp "
            f"{historical['pgp']['minimol']['mean']:.3f} +/- "
            f"{historical['pgp']['minimol']['std']:.3f}; the current Bio page displays "
            "the latter Pgp value. |"
        ),
        (
            "| Our pinned reproduction | MapLight+GNN Bio "
            f"{bio['mean']:.3f} +/- {bio['std']:.3f}; exact delta from published "
            f"reproduction {bio['published_reproduction_absolute_delta']:.3f}. |"
        ),
        (
            "| Positive control | CYP2C9 substrate reproduction "
            f"{cyp['mean']:.3f} +/- {cyp['std']:.3f} versus displayed "
            f"{cyp['displayed_tdc_mean']:.3f}; absolute delta {cyp['absolute_delta']:.3f}. |"
        ),
        (
            "| Selection isolation | Fresh validation used no official test labels; "
            "canonical identity overlap across train/validation was zero. |"
        ),
        (
            "| Existing TDC issue | Issue "
            f"#{issue['number']} reported the same mapping problem on "
            f"{issue['created_at'][:10]}; it remained "
            f"{issue['state_observed_at_audit']} with "
            f"{issue['maintainer_comments_observed_at_audit']} maintainer comments "
            "when audited. |"
        ),
        "",
        "## Interpretation boundary",
        "",
        report["assessment"]["most_supported_explanation"],
        "",
        f"Root cause proven: **{report['assessment']['root_cause_proven']}**.  ",
        f"Authoritative correction obtained: **{report['assessment']['authoritative_correction_obtained']}**.  ",
        f"Threshold override allowed: **{goal['threshold_override_allowed']}**.",
        "",
        "## Goal impact",
        "",
        f"The displayed Bio Top-10 threshold remains {goal['displayed_top10_threshold']:.3f}. "
        f"The conservative scorecard remains {goal['verified_top5_count']}/22 Top-5 and "
        f"{goal['verified_top10_count']}/22 Top-10; the full goal is not yet met.",
        "",
        (
            "For context only, the local 0.731 exceeds MiniMol's historical June 2024 "
            f"rank-5 Bio score of {historical['bio']['minimol']['mean']:.3f}, implying a "
            f"historical rank no worse than {goal['historical_june_2024_rank_upper_bound']}. "
            "This is not used as current rank evidence."
        ),
        "",
        "## Sources",
        "",
        f"- Current Bio page: {report['snapshot']['bio_url']}",
        f"- Current Pgp page: {report['snapshot']['pgp_url']}",
        f"- Published audit: {paper['url']} (DOI `{paper['doi']}`)",
        f"- Audit repository: {report['audit_repository']['url']} at `{report['audit_repository']['commit']}`",
        f"- MiniMol primary report: {historical['repository']} at observed HEAD `{historical['commit_observed_at_audit']}`",
        f"- Existing TDC bug report: {issue['url']}",
        "",
        report["assessment"]["policy"],
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/leaderboard_snapshot_2026-08-17.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/bio_leaderboard_discrepancy_v27")
    )
    args = parser.parse_args()
    report = build_audit(args.repo_root, args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["assessment"]["status"],
                "pgp_containment": report["cross_page_evidence"]["pgp_containment"],
                "local_matches_published": report["checks"][
                    "local_bio_exactly_matches_published_reproduction"
                ],
                "threshold_override_allowed": report["goal_effect"][
                    "threshold_override_allowed"
                ],
                "goal_met": report["goal_effect"]["goal_met"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
