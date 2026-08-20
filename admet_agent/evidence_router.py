"""Deterministic, auditable routing over multi-seed validation evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schemas import RunResult, TaskProfile


@dataclass(frozen=True)
class EvidenceRouter:
    """Select a finalist by lower-confidence-bound utility and observed cost."""

    uncertainty_penalty: float = 0.5
    cost_penalty: float = 0.005

    def __post_init__(self) -> None:
        if not math.isfinite(self.uncertainty_penalty) or self.uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty must be a finite non-negative value")
        if not math.isfinite(self.cost_penalty) or self.cost_penalty < 0:
            raise ValueError("cost_penalty must be a finite non-negative value")

    @staticmethod
    def _action(result: RunResult) -> str:
        if result.model == "ensemble":
            return "fuse"
        if "unimol2" in result.representations:
            return "acquire_3d"
        return "stop_2d"

    def select(
        self,
        profile: TaskProfile,
        results: list[RunResult],
    ) -> dict[str, Any]:
        completed = [
            result
            for result in results
            if result.status == "completed"
            and math.isfinite(result.validation_score)
            and math.isfinite(result.elapsed_seconds)
            and result.elapsed_seconds >= 0
        ]
        if not completed:
            raise ValueError("Evidence routing requires at least one completed result")

        max_log_cost = max(math.log1p(result.elapsed_seconds) for result in completed)
        direction = 1.0 if profile.metric != "mae" else -1.0
        candidates = []
        for result in completed:
            uncertainty = float(result.validation_std or 0.0)
            if not math.isfinite(uncertainty) or uncertainty < 0:
                raise ValueError(f"Invalid validation_std for {result.run_id}")
            normalized_cost = (
                math.log1p(result.elapsed_seconds) / max_log_cost
                if max_log_cost > 0
                else 0.0
            )
            oriented_score = direction * result.validation_score
            lower_confidence_bound = (
                oriented_score - self.uncertainty_penalty * uncertainty
            )
            utility = lower_confidence_bound - self.cost_penalty * normalized_cost
            candidates.append(
                {
                    "run_id": result.run_id,
                    "action": self._action(result),
                    "validation_score": result.validation_score,
                    "validation_std": uncertainty,
                    "elapsed_seconds": result.elapsed_seconds,
                    "normalized_log_cost": normalized_cost,
                    "oriented_score": oriented_score,
                    "lower_confidence_bound": lower_confidence_bound,
                    "utility": utility,
                }
            )

        candidates.sort(
            key=lambda item: (-item["utility"], item["elapsed_seconds"], item["run_id"])
        )
        selected = candidates[0]
        cheap = [item for item in candidates if item["action"] == "stop_2d"]
        best_cheap = cheap[0] if cheap else None
        return {
            "policy": "cost_aware_lcb" if self.cost_penalty > 0 else "lcb",
            "uncertainty_penalty": self.uncertainty_penalty,
            "cost_penalty": self.cost_penalty,
            "metric": profile.metric,
            "selected_run_id": selected["run_id"],
            "selected_action": selected["action"],
            "best_cheap_run_id": None if best_cheap is None else best_cheap["run_id"],
            "candidates": candidates,
        }
