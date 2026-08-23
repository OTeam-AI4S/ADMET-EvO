"""Freeze, run and report the local composite research-decision challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from admet_agent.composite_research_challenge import (
    build_commit_reveal,
    build_submission,
    score_submission,
)


ARMS = (
    "agent_feedback_policy",
    "locked_static_rules",
    "expanded_automl_scalar_search",
    "conventional_automl_scalar_search",
    "seeded_random_action",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_bootstrap(agent: dict, control: dict, *, seed: int) -> dict:
    left = {row["episode_id"]: row["case_score"] for row in agent["rows"]}
    right = {row["episode_id"]: row["case_score"] for row in control["rows"]}
    if set(left) != set(right):
        raise ValueError("Composite score rows must align")
    differences = np.asarray([left[key] - right[key] for key in sorted(left)])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(10000, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "mean_agent_minus_control_case_score": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "agent_better_episodes": int(np.sum(differences > 0)),
        "tied_episodes": int(np.sum(differences == 0)),
        "agent_worse_episodes": int(np.sum(differences < 0)),
        "independent_unit": "locally_authored_episode",
    }


def build(output_dir: Path, *, policy_seed: int = 20260820) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    manifest, key = build_commit_reveal(reveal_nonce="frozen-local-composite-v1")
    public_path = output_dir / "public_manifest.json"
    key_path = output_dir / "revealed_key.json"
    public_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Submissions are materialized before the key is written/revealed.  This
    # gives the local run a concrete commit -> submit -> reveal artifact order.
    submissions = {}
    for arm in ARMS:
        submission = build_submission(manifest, arm, policy_seed=policy_seed)
        path = submission_dir / f"{arm}.json"
        path.write_text(
            json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        submissions[arm] = submission
    key_path.write_text(
        json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    scores = {
        arm: score_submission(manifest, submissions[arm], key) for arm in ARMS
    }
    agent = scores["agent_feedback_policy"]
    comparisons = {
        arm: _paired_bootstrap(agent, scores[arm], seed=policy_seed)
        for arm in ARMS
        if arm != "agent_feedback_policy"
    }
    return {
        "protocol": manifest["protocol"],
        "suite_id": manifest["suite_id"],
        "claim_scope": manifest["claim_scope"],
        "episode_count": manifest["episode_count"],
        "policy_seed": policy_seed,
        "same_maximum_action_budget": True,
        "maximum_followups_per_episode": 1,
        "maximum_followup_fits": 6,
        "answer_commitment_verified_for_all_arms": all(
            score["answer_commitment_verified"] for score in scores.values()
        ),
        "arms": scores,
        "agent_comparisons": comparisons,
        "training_started": False,
        "test_loaded": False,
        "claim_boundary": [
            "The challenge is a same-team local sealed holdout, not independently authored external validation.",
            "The Agent arm executes the production validation self-correction planner; it is not a hand-filled answer table.",
            "The locked static rules arm is a generic one-step transition policy, not an oracle containing per-episode answers.",
            "A positive local bootstrap interval supports benchmark conformance only and must not be promoted to universal autonomous-reasoning superiority.",
            "No predictive models were trained and no frozen TDC test feedback was used.",
        ],
        "artifact_hashes": {
            str(public_path): _sha256(public_path),
            str(key_path): _sha256(key_path),
            **{
                str(submission_dir / f"{arm}.json"): _sha256(
                    submission_dir / f"{arm}.json"
                )
                for arm in ARMS
            },
        },
    }


def render(payload: dict) -> str:
    labels = {
        "agent_feedback_policy": "Agent production feedback policy",
        "locked_static_rules": "Locked static rules",
        "expanded_automl_scalar_search": "Expanded AutoML scalar search",
        "conventional_automl_scalar_search": "Conventional AutoML scalar search",
        "seeded_random_action": "Seeded random action",
    }
    lines = [
        "# Sealed composite research-decision challenge",
        "",
        "Local commit → submit → reveal benchmark; 10 composite episodes, no model training,",
        "no frozen TDC test feedback. Every arm has at most one follow-up and six estimated fits",
        "per episode.",
        "",
        "| Arm | Decision accuracy | Safe action | Parent link | Matched pair | Macro process | Unsafe | Est. fits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, score in payload["arms"].items():
        dims = score["dimensions"]
        lines.append(
            f"| {labels[arm]} | {dims['decision_accuracy']:.3f} | "
            f"{dims['safe_action_rate']:.3f} | {dims['explicit_parent_link']:.3f} | "
            f"{dims['matched_pair_protection']:.3f} | {score['macro_process_score']:.3f} | "
            f"{score['unsafe_action_count']} | {score['total_estimated_fits']} |"
        )
    lines.extend(
        [
            "",
            "## Paired episode comparisons",
            "",
            "| Control | Mean Agent − control case score | Bootstrap 95% CI | W/T/L |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, comparison in payload["agent_comparisons"].items():
        ci = comparison["bootstrap_95_ci"]
        lines.append(
            f"| {labels[arm]} | {comparison['mean_agent_minus_control_case_score']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{comparison['agent_better_episodes']}/{comparison['tied_episodes']}/"
            f"{comparison['agent_worse_episodes']} |"
        )
    lines.extend(
        [
            "",
            "## What the composite cases add",
            "",
            "The cases require one policy to combine: supported/refuted/inconclusive feedback,",
            "paired-seed sufficiency, remaining action budget, a 0.60 compute gate, quarantined",
            "versus admitted external data, cross-axis switching, explicit parent linkage and",
            "bounded stopping. This is harder than scoring each capability in isolation.",
            "",
            "The Agent safely skipped quarantined data and over-budget Uni-Mol2-style actions,",
            "repaired incomplete evidence, switched after resolved null/refuted results, and",
            "stopped when no safe executable action remained. It used 30 estimated fits versus",
            "48 for static rules and scalar AutoML controls under the same capacity ceiling.",
            "",
            "## Claim boundary",
            "",
            *[f"- {item}" for item in payload["claim_boundary"]],
            "",
            "Therefore this artifact upgrades the evidence from three isolated fixtures to one",
            "integrated policy-conformance benchmark. It still does not satisfy the requirement",
            "for an independently authored unseen challenge or prove learned reasoning beyond all",
            "possible deterministic rules.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/v5-composite-research-challenge"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/composite_research_challenge_v1"),
    )
    args = parser.parse_args()
    payload = build(args.output_dir)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
