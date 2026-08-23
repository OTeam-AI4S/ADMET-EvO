"""Assay-conditioned multi-fidelity neural components for OpenADMET CYP."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    molecule_width: int = 512
    trunk_width: int = 192
    context_width: int = 24
    hidden_width: int = 192
    isoforms: int = 4
    assay_arms: int = 5
    readouts: int = 5
    fidelities: int = 7
    dropout: float = 0.15


class AssayConditionedCYPModel(nn.Module):
    """Shared molecule trunk with explicit assay-semantic gated conditioning."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.molecule_trunk = nn.Sequential(
            nn.Linear(config.molecule_width, config.trunk_width),
            nn.LayerNorm(config.trunk_width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.trunk_width, config.trunk_width),
            nn.SiLU(),
        )
        self.isoform_embedding = nn.Embedding(config.isoforms, config.context_width)
        self.assay_embedding = nn.Embedding(config.assay_arms, config.context_width)
        self.readout_embedding = nn.Embedding(config.readouts, config.context_width)
        self.fidelity_embedding = nn.Embedding(config.fidelities, config.context_width)
        concatenated = config.context_width * 4
        self.context_projection = nn.Sequential(
            nn.Linear(concatenated, config.trunk_width),
            nn.SiLU(),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(concatenated, config.trunk_width), nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.trunk_width),
            nn.Linear(config.trunk_width, config.hidden_width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.regression_head = nn.Linear(config.hidden_width, 2)
        self.classification_head = nn.Linear(config.hidden_width, 1)

    def forward(
        self,
        molecule: Tensor,
        isoform: Tensor,
        assay_arm: Tensor,
        readout: Tensor,
        fidelity: Tensor,
    ) -> dict[str, Tensor]:
        context = torch.cat(
            (
                self.isoform_embedding(isoform),
                self.assay_embedding(assay_arm),
                self.readout_embedding(readout),
                self.fidelity_embedding(fidelity),
            ),
            dim=-1,
        )
        trunk = self.molecule_trunk(molecule)
        conditioned = trunk * (1.0 + self.context_gate(context))
        conditioned = conditioned + self.context_projection(context)
        hidden = self.fusion(conditioned)
        regression = self.regression_head(hidden)
        return {
            "regression_mean": regression[:, 0],
            "regression_log_variance": regression[:, 1].clamp(-6.0, 4.0),
            "classification_logit": self.classification_head(hidden)[:, 0],
        }

    def manifest(self) -> dict:
        return {
            "architecture": "assay-conditioned-cyp-gated-trunk-v1",
            "config": asdict(self.config),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }


def interval_distance(
    prediction: Tensor,
    target: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Distance to a credible interval, with point-error fallback."""

    has_interval = torch.isfinite(lower) & torch.isfinite(upper)
    outside = F.relu(lower - prediction) + F.relu(prediction - upper)
    return torch.where(has_interval, outside, torch.abs(prediction - target))


def mixed_supervised_loss(
    output: dict[str, Tensor],
    *,
    target: Tensor,
    task_is_classification: Tensor,
    lower: Tensor,
    upper: Tensor,
    measurement_variance: Tensor,
    row_weight: Tensor,
    interval_weight: float,
    positive_weight: Tensor,
) -> Tensor:
    """Joint regression/classification loss with interval-aware regression."""

    classification = task_is_classification.bool()
    regression = ~classification
    losses = torch.zeros_like(target)
    if regression.any():
        mean = output["regression_mean"][regression]
        log_variance = output["regression_log_variance"][regression]
        variance = torch.exp(log_variance) + measurement_variance[regression].clamp_min(0)
        point_residual = mean - target[regression]
        has_interval = torch.isfinite(lower[regression]) & torch.isfinite(
            upper[regression]
        )
        signed_interval_residual = F.relu(mean - upper[regression]) - F.relu(
            lower[regression] - mean
        )
        residual = (
            torch.where(has_interval, signed_interval_residual, point_residual)
            if interval_weight > 0
            else point_residual
        )
        gaussian_nll = 0.5 * (residual.square() / variance + torch.log(variance))
        dead_zone = torch.abs(residual)
        losses[regression] = gaussian_nll + interval_weight * dead_zone.square()
    if classification.any():
        losses[classification] = F.binary_cross_entropy_with_logits(
            output["classification_logit"][classification],
            target[classification],
            pos_weight=positive_weight[classification],
            reduction="none",
        )
    denominator = row_weight.sum().clamp_min(torch.finfo(row_weight.dtype).eps)
    return (losses * row_weight).sum() / denominator


def tdi_consistency_loss(
    direct_prediction_z: Tensor,
    tdi_prediction_z: Tensor,
    tdi_label: Tensor,
    *,
    direct_mean: Tensor,
    direct_std: Tensor,
    tdi_mean: Tensor,
    tdi_std: Tensor,
    temperature: float = 6.0,
) -> Tensor:
    """Align predicted pIC50 shift with the declared two-fold TDI class rule."""

    direct = direct_prediction_z * direct_std + direct_mean
    tdi = tdi_prediction_z * tdi_std + tdi_mean
    shift_logit = temperature * (tdi - direct - math.log10(2.0))
    return F.binary_cross_entropy_with_logits(shift_logit, tdi_label)


__all__ = [
    "AssayConditionedCYPModel",
    "ModelConfig",
    "interval_distance",
    "mixed_supervised_loss",
    "tdi_consistency_loss",
]
