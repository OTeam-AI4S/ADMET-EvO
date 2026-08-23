"""Run training-only chemistry-group CV for the OpenADMET CYP M-family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scripts.openadmet_cyp_metrics import official_st_rae
from scripts.openadmet_multifidelity_model import (
    AssayConditionedCYPModel,
    ModelConfig,
    mixed_supervised_loss,
    tdi_consistency_loss,
)
from scripts.openadmet_split_safety import assert_disjoint_training_evaluation

ISOFORMS = {name: index for index, name in enumerate(("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"))}
ASSAY_ARMS = {
    name: index
    for index, name in enumerate(
        (
            "direct_inhibition",
            "TDI_preincubation",
            "TDI_shift",
            "single_concentration_active_preincubation",
            "Veith_qHTS_activity",
        )
    )
}
READOUTS = {
    name: index
    for index, name in enumerate(
        ("pIC50", "is_TDI", "Emax", "log2fc", "binary_activity")
    )
}
FIDELITIES = {
    name: index
    for index, name in enumerate(
        (
            "primary_direct",
            "primary_classification",
            "auxiliary_paired_direct",
            "auxiliary_paired_condition",
            "auxiliary_emax",
            "low_single_concentration",
            "external_bridge",
        )
    )
}
VARIANT_FIDELITIES = {
    "M0": {"primary_direct", "primary_classification"},
    "M1": {"primary_direct", "primary_classification"},
    "M2": {
        "primary_direct",
        "primary_classification",
        "auxiliary_paired_direct",
        "auxiliary_paired_condition",
        "auxiliary_emax",
    },
    "M3": {
        "primary_direct",
        "primary_classification",
        "auxiliary_paired_direct",
        "auxiliary_paired_condition",
        "auxiliary_emax",
        "low_single_concentration",
    },
    "M4": {
        "primary_direct",
        "primary_classification",
        "auxiliary_paired_direct",
        "auxiliary_paired_condition",
        "auxiliary_emax",
        "low_single_concentration",
    },
}
FIDELITY_WEIGHTS = {
    "primary_direct": 1.0,
    "primary_classification": 1.0,
    "auxiliary_paired_direct": 0.5,
    "auxiliary_paired_condition": 0.5,
    "auxiliary_emax": 0.25,
    "low_single_concentration": 0.15,
    "external_bridge": 0.25,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _shuffled_assay_id(identity: str, endpoint_key: str) -> int:
    digest = hashlib.sha256(
        f"openadmet-shuffled-assay-v1:{identity}:{endpoint_key}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") % len(ASSAY_ARMS)


@dataclass(frozen=True)
class Normalizer:
    mean: float
    std: float


class ObservationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        embeddings: Tensor,
        identity_index: dict[str, int],
        normalizers: dict[str, Normalizer],
        context_ablation: str = "none",
    ):
        self.frame = frame.reset_index(drop=True)
        self.embeddings = embeddings
        self.identity = torch.tensor(
            [identity_index[value] for value in self.frame["inchikey"]],
            dtype=torch.long,
        )
        isoform_values = [ISOFORMS[value] for value in self.frame["isoform"]]
        if context_ablation == "no_isoform":
            isoform_values = [0] * len(self.frame)
        self.isoform = torch.tensor(isoform_values, dtype=torch.long)
        assay_values = [ASSAY_ARMS[value] for value in self.frame["assay_arm"]]
        if context_ablation == "shuffled_assay":
            assay_values = [
                _shuffled_assay_id(identity, endpoint)
                for identity, endpoint in zip(
                    self.frame["inchikey"], self.frame["endpoint_key"], strict=True
                )
            ]
        self.assay = torch.tensor(assay_values, dtype=torch.long)
        self.readout = torch.tensor(
            [READOUTS[value] for value in self.frame["readout"]], dtype=torch.long
        )
        self.fidelity = torch.tensor(
            [FIDELITIES[value] for value in self.frame["fidelity"]], dtype=torch.long
        )
        classification = self.frame["task_type"].eq("classification").to_numpy()
        self.classification = torch.tensor(classification, dtype=torch.bool)

        target = self.frame["target"].to_numpy(dtype=np.float32)
        lower = self.frame["credible_low"].to_numpy(dtype=np.float32)
        upper = self.frame["credible_high"].to_numpy(dtype=np.float32)
        measurement_std = self.frame["measurement_std"].to_numpy(dtype=np.float32)
        for index, endpoint in enumerate(self.frame["endpoint_key"]):
            if classification[index]:
                continue
            normalizer = normalizers[endpoint]
            target[index] = (target[index] - normalizer.mean) / normalizer.std
            if np.isfinite(lower[index]):
                lower[index] = (lower[index] - normalizer.mean) / normalizer.std
            if np.isfinite(upper[index]):
                upper[index] = (upper[index] - normalizer.mean) / normalizer.std
            if np.isfinite(measurement_std[index]):
                measurement_std[index] /= normalizer.std
        measurement_variance = np.where(
            np.isfinite(measurement_std), measurement_std**2, 0.0
        ).astype(np.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.lower = torch.tensor(lower, dtype=torch.float32)
        self.upper = torch.tensor(upper, dtype=torch.float32)
        self.measurement_variance = torch.tensor(
            measurement_variance, dtype=torch.float32
        )

        counts = self.frame["endpoint_key"].value_counts().to_dict()
        weights = np.asarray(
            [
                FIDELITY_WEIGHTS[fidelity] / counts[endpoint]
                for endpoint, fidelity in zip(
                    self.frame["endpoint_key"], self.frame["fidelity"], strict=True
                )
            ],
            dtype=np.float32,
        )
        weights /= weights.mean()
        self.row_weight = torch.tensor(weights, dtype=torch.float32)
        positive_weight = np.ones(len(self.frame), dtype=np.float32)
        for endpoint, group in self.frame.loc[
            self.frame["task_type"] == "classification"
        ].groupby("endpoint_key"):
            positives = int((group["target"] == 1).sum())
            negatives = int((group["target"] == 0).sum())
            value = negatives / positives if positives else 1.0
            positive_weight[group.index.to_numpy()] = value
        self.positive_weight = torch.tensor(positive_weight, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "molecule": self.embeddings[self.identity[index]],
            "isoform": self.isoform[index],
            "assay": self.assay[index],
            "readout": self.readout[index],
            "fidelity": self.fidelity[index],
            "target": self.target[index],
            "classification": self.classification[index],
            "lower": self.lower[index],
            "upper": self.upper[index],
            "measurement_variance": self.measurement_variance[index],
            "row_weight": self.row_weight[index],
            "positive_weight": self.positive_weight[index],
            "row_index": torch.tensor(index, dtype=torch.long),
        }


def _normalizers(frame: pd.DataFrame) -> dict[str, Normalizer]:
    output = {}
    regression = frame.loc[frame["task_type"] == "regression"]
    for endpoint, group in regression.groupby("endpoint_key"):
        std = float(group["target"].std(ddof=0))
        output[endpoint] = Normalizer(
            mean=float(group["target"].mean()),
            std=max(std, 1e-6),
        )
    return output


def _batch_loss(
    model: AssayConditionedCYPModel,
    batch: dict[str, Tensor],
    *,
    variant: str,
) -> Tensor:
    output = model(
        batch["molecule"],
        batch["isoform"],
        batch["assay"],
        batch["readout"],
        batch["fidelity"],
    )
    if variant == "M0":
        output["regression_log_variance"] = torch.zeros_like(
            output["regression_mean"]
        )
    return mixed_supervised_loss(
        output,
        target=batch["target"],
        task_is_classification=batch["classification"],
        lower=batch["lower"],
        upper=batch["upper"],
        measurement_variance=(
            batch["measurement_variance"]
            if variant != "M0"
            else torch.zeros_like(batch["measurement_variance"])
        ),
        row_weight=batch["row_weight"],
        interval_weight=1.0 if variant != "M0" else 0.0,
        positive_weight=batch["positive_weight"],
    )


def _consistency_rows(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame.pivot_table(
        index=["inchikey", "isoform"],
        columns=["assay_arm", "readout", "fidelity"],
        values="target",
        aggfunc="first",
    )
    direct_key = ("direct_inhibition", "pIC50", "auxiliary_paired_direct")
    tdi_key = ("TDI_preincubation", "pIC50", "auxiliary_paired_condition")
    class_key = ("TDI_shift", "is_TDI", "primary_classification")
    if not {direct_key, tdi_key, class_key}.issubset(primary.columns):
        return pd.DataFrame()
    output = primary[[direct_key, tdi_key, class_key]].dropna().reset_index()
    output.columns = ["inchikey", "isoform", "direct", "tdi", "label"]
    return output


def _apply_consistency_steps(
    model: AssayConditionedCYPModel,
    optimizer: torch.optim.Optimizer,
    consistency: pd.DataFrame,
    *,
    embeddings: Tensor,
    identity_index: dict[str, int],
    normalizers: dict[str, Normalizer],
    batch_size: int,
    coefficient: float,
    context_ablation: str,
) -> float:
    if consistency.empty or coefficient <= 0:
        return 0.0
    losses = []
    order = torch.randperm(len(consistency))
    for start in range(0, len(consistency), batch_size):
        rows = consistency.iloc[order[start : start + batch_size].numpy()]
        identity = torch.tensor(
            [identity_index[value] for value in rows["inchikey"]], dtype=torch.long
        )
        isoform = torch.tensor([ISOFORMS[value] for value in rows["isoform"]])
        if context_ablation == "no_isoform":
            isoform = torch.zeros_like(isoform)
        molecule = embeddings[identity]
        direct_endpoint = [
            f"{value}|direct_inhibition|pIC50|auxiliary_paired_direct"
            for value in rows["isoform"]
        ]
        tdi_endpoint = [
            f"{value}|TDI_preincubation|pIC50|auxiliary_paired_condition"
            for value in rows["isoform"]
        ]
        if context_ablation == "shuffled_assay":
            direct_assay = torch.tensor(
                [
                    _shuffled_assay_id(identity_value, endpoint)
                    for identity_value, endpoint in zip(
                        rows["inchikey"], direct_endpoint, strict=True
                    )
                ]
            )
            tdi_assay = torch.tensor(
                [
                    _shuffled_assay_id(identity_value, endpoint)
                    for identity_value, endpoint in zip(
                        rows["inchikey"], tdi_endpoint, strict=True
                    )
                ]
            )
        else:
            direct_assay = torch.full_like(
                isoform, ASSAY_ARMS["direct_inhibition"]
            )
            tdi_assay = torch.full_like(isoform, ASSAY_ARMS["TDI_preincubation"])
        direct_output = model(
            molecule,
            isoform,
            direct_assay,
            torch.full_like(isoform, READOUTS["pIC50"]),
            torch.full_like(isoform, FIDELITIES["auxiliary_paired_direct"]),
        )
        tdi_output = model(
            molecule,
            isoform,
            tdi_assay,
            torch.full_like(isoform, READOUTS["pIC50"]),
            torch.full_like(isoform, FIDELITIES["auxiliary_paired_condition"]),
        )
        direct_norm = [
            normalizers[
                f"{value}|direct_inhibition|pIC50|auxiliary_paired_direct"
            ]
            for value in rows["isoform"]
        ]
        tdi_norm = [
            normalizers[
                f"{value}|TDI_preincubation|pIC50|auxiliary_paired_condition"
            ]
            for value in rows["isoform"]
        ]
        loss = tdi_consistency_loss(
            direct_output["regression_mean"],
            tdi_output["regression_mean"],
            torch.tensor(rows["label"].to_numpy(), dtype=torch.float32),
            direct_mean=torch.tensor([item.mean for item in direct_norm]),
            direct_std=torch.tensor([item.std for item in direct_norm]),
            tdi_mean=torch.tensor([item.mean for item in tdi_norm]),
            tdi_std=torch.tensor([item.std for item in tdi_norm]),
        )
        optimizer.zero_grad(set_to_none=True)
        (coefficient * loss).backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def _train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    embeddings: Tensor,
    identity_index: dict[str, int],
    variant: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int | None,
    select_best: bool = True,
    context_ablation: str = "none",
) -> tuple[AssayConditionedCYPModel, int, dict[str, Normalizer]]:
    _seed_everything(seed)
    normalizers = _normalizers(train)
    train_dataset = ObservationDataset(
        train,
        embeddings=embeddings,
        identity_index=identity_index,
        normalizers=normalizers,
        context_ablation=context_ablation,
    )
    validation_dataset = ObservationDataset(
        validation,
        embeddings=embeddings,
        identity_index=identity_index,
        normalizers=normalizers,
        context_ablation=context_ablation,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )
    model = AssayConditionedCYPModel(ModelConfig(molecule_width=embeddings.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    consistency = _consistency_rows(train) if variant == "M4" else pd.DataFrame()
    best_state = None
    best_loss = math.inf
    best_epoch = epochs
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, variant=variant)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if variant == "M4":
            _apply_consistency_steps(
                model,
                optimizer,
                consistency,
                embeddings=embeddings,
                identity_index=identity_index,
                normalizers=normalizers,
                batch_size=batch_size,
                coefficient=0.2,
                context_ablation=context_ablation,
            )
        if not select_best:
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            continue
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                validation_losses.append(float(_batch_loss(model, batch, variant=variant)))
        current = float(np.mean(validation_losses))
        if current < best_loss - 1e-5:
            best_loss = current
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if patience is not None and epoch >= 5 and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, normalizers


def _predict(
    model: AssayConditionedCYPModel,
    frame: pd.DataFrame,
    *,
    embeddings: Tensor,
    identity_index: dict[str, int],
    normalizers: dict[str, Normalizer],
    batch_size: int,
    variant: str,
    context_ablation: str,
) -> pd.DataFrame:
    dataset = ObservationDataset(
        frame,
        embeddings=embeddings,
        identity_index=identity_index,
        normalizers=normalizers,
        context_ablation=context_ablation,
    )
    loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    predictions = np.empty(len(frame), dtype=np.float64)
    uncertainty = np.full(len(frame), np.nan, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["molecule"],
                batch["isoform"],
                batch["assay"],
                batch["readout"],
                batch["fidelity"],
            )
            indices = batch["row_index"].numpy()
            classification = batch["classification"].numpy().astype(bool)
            predictions[indices[classification]] = torch.sigmoid(
                output["classification_logit"][classification]
            ).numpy()
            regression_indices = indices[~classification]
            regression_prediction = output["regression_mean"][~classification].numpy()
            regression_log_variance = output["regression_log_variance"][~classification].numpy()
            endpoints = frame.iloc[regression_indices]["endpoint_key"].tolist()
            for local, (row_index, endpoint) in enumerate(
                zip(regression_indices, endpoints, strict=True)
            ):
                normalizer = normalizers[endpoint]
                predictions[row_index] = (
                    regression_prediction[local] * normalizer.std + normalizer.mean
                )
                if variant != "M0":
                    uncertainty[row_index] = (
                        math.exp(0.5 * regression_log_variance[local]) * normalizer.std
                    )
    output = frame.reset_index(drop=True).copy()
    output["prediction"] = predictions
    output["predictive_std"] = uncertainty
    return output


def _metrics(predictions: pd.DataFrame) -> dict:
    direct = predictions.loc[predictions["fidelity"] == "primary_direct"]
    regression = {}
    for endpoint, group in direct.groupby("endpoint_key"):
        regression[endpoint] = {
            "st_rae": official_st_rae(
                target=group["target"].to_numpy(),
                prediction=group["prediction"].to_numpy(),
                lower=group["credible_low"].to_numpy(),
                upper=group["credible_high"].to_numpy(),
            ),
            "mae": float(np.mean(np.abs(group["prediction"] - group["target"]))),
            "rows": len(group),
        }
    classes = predictions.loc[predictions["fidelity"] == "primary_classification"]
    classification = {}
    for endpoint, group in classes.groupby("endpoint_key"):
        binary = (group["prediction"].to_numpy() >= 0.5).astype(int)
        classification[endpoint] = {
            "mcc": float(matthews_corrcoef(group["target"].astype(int), binary)),
            "rows": len(group),
        }
    return {
        "direct_inhibition": regression,
        "macro_st_rae": (
            float(np.mean([item["st_rae"] for item in regression.values()]))
            if regression
            else None
        ),
        "tdi_classification": classification,
        "macro_mcc": (
            float(np.mean([item["mcc"] for item in classification.values()]))
            if classification
            else None
        ),
    }


def run(args: argparse.Namespace) -> dict:
    if args.variant not in VARIANT_FIDELITIES:
        raise ValueError(f"Unknown variant {args.variant!r}")
    if args.context_ablation not in {"none", "no_isoform", "shuffled_assay"}:
        raise ValueError(f"Unknown context ablation {args.context_ablation!r}")
    torch.set_num_threads(args.threads)
    longform_path = args.longform.resolve()
    longform_manifest_path = longform_path.with_suffix(".manifest.json")
    longform_manifest = json.loads(longform_manifest_path.read_text())
    if longform_manifest.get("training_only") is not True:
        raise RuntimeError("Long-form artifact is not training-only")
    if longform_manifest.get("blind_test_opened") is not False:
        raise RuntimeError("Long-form artifact indicates blind-test access")
    if longform_manifest["output_sha256"] != _sha256(longform_path):
        raise RuntimeError("Long-form table hash mismatch")
    embedding_input_path = args.embedding_input.resolve()
    embedding_manifest_path = args.embedding.with_suffix(".manifest.json").resolve()
    embedding_manifest = json.loads(embedding_manifest_path.read_text())
    if embedding_manifest["input_sha256"] != _sha256(embedding_input_path):
        raise RuntimeError("Embedding input hash mismatch")
    embedding_input = json.loads(embedding_input_path.read_text())
    if embedding_input.get("contains_labels") is not False:
        raise RuntimeError("Embedding input must be label-free")
    if embedding_input.get("official_test_loaded") is not False:
        raise RuntimeError("Embedding input indicates blind-test access")
    external_bridge = None
    external_contract = None
    external_bridge_arg = getattr(args, "external_bridge", None)
    if external_bridge_arg is None:
        if longform_manifest["minimol_input_sha256"] != _sha256(embedding_input_path):
            raise RuntimeError("Long-form manifest does not bind this embedding input")
    else:
        bridge_path = external_bridge_arg.resolve()
        bridge_manifest_path = bridge_path.with_suffix(".manifest.json")
        bridge_manifest = json.loads(bridge_manifest_path.read_text())
        if bridge_manifest.get("training_only") is not True:
            raise RuntimeError("External bridge is not training-only")
        if bridge_manifest.get("blind_test_opened") is not False:
            raise RuntimeError("External bridge indicates blind-test access")
        if bridge_manifest["output_sha256"] != _sha256(bridge_path):
            raise RuntimeError("External bridge hash mismatch")
        parents = embedding_input.get("parents", {})
        if parents.get("main", {}).get("sha256") != longform_manifest[
            "minimol_input_sha256"
        ]:
            raise RuntimeError("Embedding union does not bind the main identity input")
        if parents.get("bridge", {}).get("sha256") != _sha256(bridge_path):
            raise RuntimeError("Embedding union does not bind the external bridge")
        external_bridge = pd.read_csv(bridge_path)
        bridge_contract = {
            "fidelity": {"external_bridge"},
            "task_type": {"classification"},
            "readout": {"binary_activity"},
            "assay_arm": {"Veith_qHTS_activity"},
        }
        for column, allowed in bridge_contract.items():
            observed = set(external_bridge[column].dropna().astype(str))
            if observed != allowed:
                raise RuntimeError(
                    f"External bridge violated {column} contract: {sorted(observed)}"
                )
        if not set(external_bridge["target"].dropna().astype(int)).issubset({0, 1}):
            raise RuntimeError("External bridge contains a non-binary target")
        external_contract = {
            "path": str(bridge_path),
            "sha256": _sha256(bridge_path),
            "manifest_path": str(bridge_manifest_path.resolve()),
            "manifest_sha256": _sha256(bridge_manifest_path),
            "route": bridge_manifest["route"],
            "rows": len(external_bridge),
        }
    embedding_array = np.load(args.embedding.resolve())
    if embedding_manifest.get("output_sha256") != _sha256(args.embedding.resolve()):
        raise RuntimeError("Embedding output hash mismatch")
    if embedding_array.shape != (len(embedding_input["inchikeys"]), 512):
        raise RuntimeError(f"Unexpected embedding shape {embedding_array.shape}")
    embeddings = torch.tensor(embedding_array, dtype=torch.float32)
    identity_index = {
        value: index for index, value in enumerate(embedding_input["inchikeys"])
    }

    longform = pd.read_csv(longform_path)
    longform = longform.loc[
        longform["fidelity"].isin(VARIANT_FIDELITIES[args.variant])
        & longform["partition"].eq("development")
    ].copy()
    outer_folds = [int(value) for value in args.outer_folds.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    prediction_frames = []
    for outer in outer_folds:
        main_outer_train = longform.loc[longform["outer_fold"] != outer].copy()
        outer_validation = longform.loc[longform["outer_fold"] == outer].copy()
        inner_column = f"inner_fold_when_outer_{outer}"
        inner_train = main_outer_train.loc[main_outer_train[inner_column] != 0].copy()
        inner_validation = main_outer_train.loc[
            main_outer_train[inner_column] == 0
        ].copy()
        outer_external = pd.DataFrame()
        if external_bridge is not None:
            excluded = external_bridge["excluded_outer_folds"].fillna("").map(
                lambda value: {
                    int(item) for item in str(value).split(";") if item != ""
                }
            )
            outer_external = external_bridge.loc[
                ~excluded.map(lambda values, fold=outer: fold in values)
            ].copy()
            inner_groups = set(inner_validation["chemistry_group"])
            inner_external = outer_external.loc[
                ~outer_external["chemistry_group"].isin(inner_groups)
            ].copy()
            inner_train = pd.concat([inner_train, inner_external], ignore_index=True)
        outer_train = pd.concat(
            [main_outer_train, outer_external], ignore_index=True
        )
        assert_disjoint_training_evaluation(
            inner_train,
            inner_validation,
            context=f"outer-{outer}/inner-epoch-selection",
        )
        assert_disjoint_training_evaluation(
            outer_train,
            outer_validation,
            context=f"outer-{outer}/refit-evaluation",
        )
        for seed in seeds:
            _, best_epoch, _ = _train_model(
                inner_train,
                inner_validation,
                embeddings=embeddings,
                identity_index=identity_index,
                variant=args.variant,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                context_ablation=args.context_ablation,
            )
            # Refit from the same initialization on every outer-training group
            # for the inner-selected epoch count. The dummy validation is never
            # used for stopping because patience is disabled.
            model, _, normalizers = _train_model(
                outer_train,
                inner_validation,
                embeddings=embeddings,
                identity_index=identity_index,
                variant=args.variant,
                seed=seed,
                epochs=best_epoch,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=None,
                select_best=False,
                context_ablation=args.context_ablation,
            )
            predictions = _predict(
                model,
                outer_validation,
                embeddings=embeddings,
                identity_index=identity_index,
                normalizers=normalizers,
                batch_size=args.batch_size,
                variant=args.variant,
                context_ablation=args.context_ablation,
            )
            predictions["outer_fold_evaluated"] = outer
            predictions["training_seed"] = seed
            metrics = _metrics(predictions)
            records.append(
                {
                    "outer_fold": outer,
                    "seed": seed,
                    "inner_selected_epoch": best_epoch,
                    "external_bridge_rows_inner_train": int(
                        (inner_train["partition"] == "external_bridge").sum()
                    ),
                    "external_bridge_rows_outer_train": int(
                        (outer_train["partition"] == "external_bridge").sum()
                    ),
                    "metrics": metrics,
                }
            )
            prediction_frames.append(predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_path = args.output_dir / f"{args.variant.lower()}_outer_predictions.csv"
    all_predictions.to_csv(prediction_path, index=False)
    macro_st_rae = [
        item["metrics"]["macro_st_rae"]
        for item in records
        if item["metrics"]["macro_st_rae"] is not None
    ]
    macro_mcc = [
        item["metrics"]["macro_mcc"]
        for item in records
        if item["metrics"]["macro_mcc"] is not None
    ]
    result = {
        "version": "openadmet-cyp-multifidelity-cv-v1",
        "variant": args.variant,
        "context_ablation": args.context_ablation,
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "longform_sha256": _sha256(longform_path),
        "longform_manifest_sha256": _sha256(longform_manifest_path),
        "embedding_input_sha256": _sha256(embedding_input_path),
        "embedding_sha256": _sha256(args.embedding.resolve()),
        "embedding_manifest_sha256": _sha256(embedding_manifest_path),
        "external_bridge": external_contract,
        "outer_folds": outer_folds,
        "seeds": seeds,
        "epochs_ceiling": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "threads": args.threads,
        "fidelity_weights": FIDELITY_WEIGHTS,
        "consistency_coefficient": 0.2 if args.variant == "M4" else 0.0,
        "architecture": AssayConditionedCYPModel().manifest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": version("scikit-learn"),
            "cuda_available": torch.cuda.is_available(),
        },
        "fold_seed_results": records,
        "mean_macro_st_rae": float(np.mean(macro_st_rae)),
        "mean_macro_mcc": float(np.mean(macro_mcc)),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    result_path = args.output_dir / f"{args.variant.lower()}_cv_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {**result, "result_path": str(result_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--longform",
        type=Path,
        default=Path("reports/openadmet_cyp_training_longform_v1.csv"),
    )
    parser.add_argument(
        "--embedding-input",
        type=Path,
        default=Path("reports/openadmet_cyp_training_longform_v1.minimol_input.json"),
    )
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--external-bridge", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANT_FIDELITIES), required=True)
    parser.add_argument(
        "--context-ablation",
        choices=("none", "no_isoform", "shuffled_assay"),
        default="none",
    )
    parser.add_argument("--outer-folds", default="0,1,2")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/openadmet-cyp-stage-a")
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
