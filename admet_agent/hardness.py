"""Multi-dimensional endpoint hardness profiling using discovery data only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .data_governance import MolecularDataCurator, canonical_smiles
from .tasks import TaskSpec


def _bounded(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _scaffold(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)


def _fingerprint(smiles: str | None):
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _sample_indices(length: int, limit: int, seed: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=int)
    return np.sort(np.random.default_rng(seed).choice(length, size=limit, replace=False))


@dataclass(frozen=True)
class HardnessDimension:
    severity: float
    confidence: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardnessCard:
    task: str
    task_type: str
    metric: str
    overall: float
    dominant_factors: tuple[str, ...]
    dimensions: dict[str, HardnessDimension]
    recommended_actions: tuple[str, ...]
    discovery_only: bool = True
    profiler_version: str = "hardness-v1"
    evidence_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = {
            key: value.to_dict() for key, value in self.dimensions.items()
        }
        return payload


def _nearest_neighbor_evidence(
    train_smiles: list[str | None],
    train_y: np.ndarray,
    valid_smiles: list[str | None],
    valid_y: np.ndarray,
    task_type: str,
    *,
    sample_limit: int,
    seed: int,
) -> dict[str, float]:
    train_idx = _sample_indices(len(train_smiles), sample_limit, seed)
    valid_idx = _sample_indices(len(valid_smiles), min(sample_limit, 1000), seed + 1)
    train_records = [
        (_fingerprint(train_smiles[index]), float(train_y[index])) for index in train_idx
    ]
    train_records = [item for item in train_records if item[0] is not None and math.isfinite(item[1])]
    if not train_records:
        return {
            "mean_nearest_similarity": 0.0,
            "p10_nearest_similarity": 0.0,
            "ood_rate_at_0_4": 1.0,
            "local_disagreement": 1.0,
            "evaluated_valid": 0,
        }
    fps = [item[0] for item in train_records]
    labels = np.asarray([item[1] for item in train_records], dtype=float)
    target_scale = float(np.std(train_y)) or 1.0
    similarities = []
    disagreement = []
    for index in valid_idx:
        fp = _fingerprint(valid_smiles[index])
        y_value = float(valid_y[index])
        if fp is None or not math.isfinite(y_value):
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        nearest = int(np.argmax(sims))
        similarities.append(float(sims[nearest]))
        if task_type == "classification":
            disagreement.append(float(labels[nearest] != y_value))
        else:
            disagreement.append(_bounded(abs(labels[nearest] - y_value) / (2.0 * target_scale)))
    if not similarities:
        return {
            "mean_nearest_similarity": 0.0,
            "p10_nearest_similarity": 0.0,
            "ood_rate_at_0_4": 1.0,
            "local_disagreement": 1.0,
            "evaluated_valid": 0,
        }
    values = np.asarray(similarities, dtype=float)
    return {
        "mean_nearest_similarity": float(np.mean(values)),
        "p10_nearest_similarity": float(np.quantile(values, 0.1)),
        "ood_rate_at_0_4": float(np.mean(values < 0.4)),
        "local_disagreement": float(np.mean(disagreement)),
        "evaluated_valid": len(values),
    }


def _activity_cliff_evidence(
    smiles: list[str | None],
    targets: np.ndarray,
    task_type: str,
    *,
    sample_limit: int,
    seed: int,
) -> dict[str, float]:
    indices = _sample_indices(len(smiles), min(sample_limit, 1200), seed + 2)
    records = [(_fingerprint(smiles[index]), float(targets[index])) for index in indices]
    records = [item for item in records if item[0] is not None and math.isfinite(item[1])]
    if len(records) < 2:
        return {"high_similarity_pairs": 0, "cliff_pairs": 0, "cliff_rate": 0.0}
    fps = [item[0] for item in records]
    labels = np.asarray([item[1] for item in records], dtype=float)
    scale = float(np.std(labels)) or 1.0
    high_similarity = 0
    cliffs = 0
    # Nearest-neighbour cliffs capture the local ruggedness without materialising
    # an O(n^2) similarity matrix.
    for index, fp in enumerate(fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps), dtype=float)
        sims[index] = -1.0
        nearest = int(np.argmax(sims))
        if sims[nearest] < 0.75:
            continue
        high_similarity += 1
        if task_type == "classification":
            cliffs += int(labels[index] != labels[nearest])
        else:
            cliffs += int(abs(labels[index] - labels[nearest]) >= scale)
    return {
        "high_similarity_pairs": high_similarity,
        "cliff_pairs": cliffs,
        "cliff_rate": float(cliffs / high_similarity) if high_similarity else 0.0,
    }


def profile_hardness(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: TaskSpec,
    smiles_col: str = "Drug",
    target_col: str = "Y",
    *,
    sample_limit: int = 2000,
    seed: int = 1,
) -> HardnessCard:
    if sample_limit < 100:
        raise ValueError("sample_limit must be at least 100")
    train_smiles = [canonical_smiles(value) for value in train[smiles_col].tolist()]
    valid_smiles = [canonical_smiles(value) for value in valid[smiles_col].tolist()]
    train_y = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float)
    valid_y = pd.to_numeric(valid[target_col], errors="coerce").to_numpy(dtype=float)
    curator = MolecularDataCurator(smiles_col, target_col)
    audit = curator.audit(train, task_type=spec.task_type)

    valid_canonical = {value for value in valid_smiles if value is not None}
    exact_overlap = sum(value in valid_canonical for value in train_smiles if value is not None)
    train_scaffolds = {_scaffold(value) for value in train_smiles}
    train_scaffolds.discard(None)
    valid_scaffolds = [value for value in (_scaffold(item) for item in valid_smiles) if value]
    unseen_scaffold_rate = (
        float(np.mean([value not in train_scaffolds for value in valid_scaffolds]))
        if valid_scaffolds else 0.0
    )
    nn = _nearest_neighbor_evidence(
        train_smiles, train_y, valid_smiles, valid_y, spec.task_type,
        sample_limit=sample_limit, seed=seed,
    )
    cliffs = _activity_cliff_evidence(
        train_smiles, train_y, spec.task_type, sample_limit=sample_limit, seed=seed
    )

    effective_scaffolds = max(1, len(train_scaffolds))
    scarcity_severity = _bounded(1.0 - math.log1p(effective_scaffolds) / math.log1p(3000))
    if spec.task_type == "classification":
        finite_y = train_y[np.isfinite(train_y)]
        positive_rate = float(np.mean(finite_y)) if len(finite_y) else 0.5
        imbalance_severity = _bounded(1.0 - 2.0 * min(positive_rate, 1.0 - positive_rate))
        imbalance_evidence = {"positive_rate": positive_rate, "minority_count": int(min(np.sum(finite_y == 0), np.sum(finite_y == 1)))}
    else:
        finite_y = train_y[np.isfinite(train_y)]
        positive_rate = None
        imbalance_severity = 0.0
        imbalance_evidence = {"not_applicable": True}
    ood_severity = _bounded(
        0.45 * unseen_scaffold_rate
        + 0.35 * nn["ood_rate_at_0_4"]
        + 0.20 * (1.0 - nn["mean_nearest_similarity"])
    )
    roughness_severity = _bounded(0.6 * nn["local_disagreement"] + 0.4 * cliffs["cliff_rate"])
    # Prevalence, rather than the conditional conflict fraction among the few
    # duplicate structures, determines endpoint-level quality severity.  The
    # latter is retained as evidence but must not turn 3 conflicts in 600 rows
    # into maximum hardness.
    invalid_target_rate = (
        audit.invalid_smiles + audit.missing_targets + audit.non_finite_targets
    ) / max(1, len(train))
    if spec.task_type == "classification":
        label_noise_proxy = 4.0 * audit.conflicting_structures / max(1, len(train))
    else:
        replicate_prevalence = audit.replicate_rows / max(1, len(train))
        relative_dispersion = float(audit.replicate_target_dispersion or 0.0)
        label_noise_proxy = replicate_prevalence * min(1.0, relative_dispersion)
    quality_severity = _bounded(invalid_target_rate + label_noise_proxy)
    leakage_severity = _bounded(exact_overlap / max(1, min(len(train), len(valid))))
    dimensions = {
        "scarcity": HardnessDimension(scarcity_severity, 1.0, {"n_train": len(train), "unique_scaffolds": len(train_scaffolds)}),
        "imbalance": HardnessDimension(imbalance_severity, 1.0 if spec.task_type == "classification" else 0.0, imbalance_evidence),
        "domain_shift": HardnessDimension(ood_severity, min(1.0, nn["evaluated_valid"] / 200.0), {"unseen_valid_scaffold_rate": unseen_scaffold_rate, **nn}),
        "local_roughness": HardnessDimension(roughness_severity, min(1.0, nn["evaluated_valid"] / 200.0), {"nearest_neighbor_disagreement": nn["local_disagreement"], **cliffs}),
        "data_quality": HardnessDimension(quality_severity, 1.0, audit.to_dict()),
        "split_leakage_risk": HardnessDimension(leakage_severity, 1.0, {"exact_train_valid_overlap_rows": exact_overlap}),
    }
    weights = {
        "scarcity": 0.18,
        "imbalance": 0.12 if spec.task_type == "classification" else 0.0,
        "domain_shift": 0.25,
        "local_roughness": 0.23,
        "data_quality": 0.12,
        "split_leakage_risk": 0.10,
    }
    total_weight = sum(weights.values())
    overall = sum(dimensions[name].severity * weight for name, weight in weights.items()) / total_weight
    dominant = tuple(
        name for name, _ in sorted(
            ((name, value.severity) for name, value in dimensions.items()),
            key=lambda item: (-item[1], item[0]),
        )[:3]
    )
    actions = []
    if dimensions["split_leakage_risk"].severity > 0:
        actions.append("data:leakage_safe")
    if dimensions["data_quality"].severity >= 0.05:
        actions.append("data:quality_filtered")
    if dimensions["imbalance"].severity >= 0.35:
        actions.append("model:imbalance_aware")
    if dimensions["domain_shift"].severity >= 0.35:
        actions.append("feature:multi_view_and_pretrained")
    if dimensions["local_roughness"].severity >= 0.25:
        actions.append("model:local_kernel_or_ensemble")
    if dimensions["scarcity"].severity >= 0.45:
        actions.append("data:assay_aligned_external_or_transfer")
    actions.append("replicate:multi_seed_lcb")
    evidence_payload = {
        "task": spec.name,
        "dimensions": {key: value.to_dict() for key, value in dimensions.items()},
    }
    evidence_hash = hashlib.sha256(
        json.dumps(evidence_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return HardnessCard(
        task=spec.name,
        task_type=spec.task_type,
        metric=spec.metric,
        overall=float(overall),
        dominant_factors=dominant,
        dimensions=dimensions,
        recommended_actions=tuple(dict.fromkeys(actions)),
        evidence_sha256=evidence_hash,
    )
