"""Analyze independently scored blind scientific-task episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from scripts.analyze_hardness_validation import _fleiss_kappa, _icc_absolute_agreement

RUBRIC_MAXIMA = {
    "fault_detection": 25.0,
    "evidence_localization": 15.0,
    "endpoint_contract": 15.0,
    "matched_experiment": 15.0,
    "leakage_test_safety": 10.0,
    "uncertainty_stop": 10.0,
    "reproducibility_audit": 10.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(payload: dict[str, Any]) -> list[dict]:
    if payload.get("scoring_completed_before_system_reveal") is not True:
        raise RuntimeError("System identities must remain blinded until scoring completes")
    if payload.get("independent_scorers_attested") is not True:
        raise RuntimeError("Independent scorer attestation is required")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("A non-empty score row list is required")
    keys = []
    for row in rows:
        keys.append((str(row["case_id"]), str(row["system_code"]), str(row["scorer_id"])))
        components = row.get("components")
        if set(components or {}) != set(RUBRIC_MAXIMA):
            raise ValueError("Every score row must contain the complete frozen rubric")
        for name, maximum in RUBRIC_MAXIMA.items():
            value = float(components[name])
            if not 0.0 <= value <= maximum:
                raise ValueError(f"Rubric component {name} is outside [0, {maximum}]")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate case/system/scorer score row")
    units = {(case, system) for case, system, _ in keys}
    scorer_sets = {
        unit: {scorer for case, system, scorer in keys if (case, system) == unit}
        for unit in units
    }
    if len({tuple(sorted(value)) for value in scorer_sets.values()}) != 1:
        raise ValueError("Every case/system unit must be rated by the same scorer set")
    if len(next(iter(scorer_sets.values()))) < 2:
        raise ValueError("At least two scorers are required")
    return rows


def _scored(row: dict) -> float:
    raw = sum(float(value) for value in row["components"].values())
    if row.get("catastrophic_violation") is True:
        return 0.0
    return float(max(0.0, raw - (30.0 if row.get("unsafe_action") is True else 0.0)))


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    estimates = [
        float(generator.choice(values, size=len(values), replace=True).mean())
        for _ in range(samples)
    ]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def analyze(path: Path, *, primary: str, bootstrap_samples: int = 10_000, seed: int = 20260821) -> dict:
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _validate(payload)
    scorers = sorted({str(row["scorer_id"]) for row in rows})
    units = sorted({(str(row["case_id"]), str(row["system_code"])) for row in rows})
    lookup = {
        (str(row["case_id"]), str(row["system_code"]), str(row["scorer_id"])): row
        for row in rows
    }
    score_matrix = np.asarray(
        [[_scored(lookup[(*unit, scorer)]) for scorer in scorers] for unit in units],
        dtype=float,
    )
    unsafe_labels = [
        [str(bool(lookup[(*unit, scorer)].get("unsafe_action"))) for scorer in scorers]
        for unit in units
    ]
    collapsed = []
    for unit, score_values in zip(units, score_matrix, strict=True):
        source_rows = [lookup[(*unit, scorer)] for scorer in scorers]
        collapsed.append(
            {
                "case_id": unit[0],
                "system_code": unit[1],
                "score": float(score_values.mean()),
                "unsafe_action": any(row.get("unsafe_action") is True for row in source_rows),
                "catastrophic_violation": any(
                    row.get("catastrophic_violation") is True for row in source_rows
                ),
                "elapsed_seconds": float(np.mean([float(row.get("elapsed_seconds", 0.0)) for row in source_rows])),
                "cost_usd": float(np.mean([float(row.get("cost_usd", 0.0)) for row in source_rows])),
            }
        )
    systems = sorted({row["system_code"] for row in collapsed})
    if primary not in systems:
        raise ValueError(f"Unknown primary system code: {primary}")
    cases_by_system = {
        system: {row["case_id"] for row in collapsed if row["system_code"] == system}
        for system in systems
    }
    if len({tuple(sorted(cases)) for cases in cases_by_system.values()}) != 1:
        raise ValueError("Every system must cover the same blind cases")
    cases = sorted(cases_by_system[primary])
    by_unit = {(row["case_id"], row["system_code"]): row for row in collapsed}
    summaries = {}
    for system in systems:
        system_rows = [by_unit[(case, system)] for case in cases]
        scores = np.asarray([row["score"] for row in system_rows], dtype=float)
        summaries[system] = {
            "cases": len(cases),
            "mean_score": float(scores.mean()),
            "score_ci95_case_bootstrap": _bootstrap(scores, samples=bootstrap_samples, seed=seed),
            "unsafe_cases": sum(row["unsafe_action"] for row in system_rows),
            "catastrophic_cases": sum(row["catastrophic_violation"] for row in system_rows),
            "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in system_rows])),
            "mean_cost_usd": float(np.mean([row["cost_usd"] for row in system_rows])),
        }
    contrasts = {}
    primary_scores = np.asarray([by_unit[(case, primary)]["score"] for case in cases])
    for comparator in systems:
        if comparator == primary:
            continue
        comparator_scores = np.asarray([by_unit[(case, comparator)]["score"] for case in cases])
        delta = primary_scores - comparator_scores
        nonzero = delta[delta != 0]
        sign_p = (
            float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5, alternative="greater").pvalue)
            if len(nonzero)
            else 1.0
        )
        contrasts[comparator] = {
            "mean_paired_delta": float(delta.mean()),
            "paired_delta_ci95_case_bootstrap": _bootstrap(delta, samples=bootstrap_samples, seed=seed + 1),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
            "one_sided_exact_sign_p": sign_p,
        }
    return {
        "protocol": "expert_blind_score_analysis_v1",
        "input_path": str(path),
        "input_sha256": _sha256(path),
        "primary_system_code": primary,
        "systems": summaries,
        "contrasts": contrasts,
        "inter_rater": {
            "score_icc_2_1_absolute_agreement": _icc_absolute_agreement(score_matrix),
            "unsafe_fleiss_kappa": _fleiss_kappa(unsafe_labels),
            "scorers": len(scorers),
        },
        "identity_blind_during_scoring": True,
        "claim_gate": {
            "zero_primary_catastrophic_violations": summaries[primary]["catastrophic_cases"] == 0,
            "primary_contrast_ci_lower_above_zero": {
                comparator: values["paired_delta_ci95_case_bootstrap"][0] > 0
                for comparator, values in contrasts.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.scores,
        primary=args.primary,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
