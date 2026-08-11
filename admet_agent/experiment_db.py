"""Structured, queryable experiment memory for cross-task learning."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import ExperimentSpec, RunResult, TaskProfile


class ExperimentDatabase:
    """SQLite-backed source of truth; JSONL remains a human-readable audit trail."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    model TEXT NOT NULL,
                    representations TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    validation_score REAL,
                    validation_std REAL,
                    elapsed_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS runs_task_idx ON runs(task);
                CREATE INDEX IF NOT EXISTS runs_metric_idx
                    ON runs(task_type, metric, status);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def record(
        self,
        profile: TaskProfile,
        spec: ExperimentSpec,
        result: RunResult,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, task, task_type, metric, model, representations,
                    hyperparameters, validation_score, validation_std,
                    elapsed_seconds, status, hypothesis, spec_json,
                    result_json, profile_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.task,
                    profile.task_type,
                    result.metric,
                    result.model,
                    json.dumps(list(spec.representations)),
                    json.dumps(spec.hyperparameters, sort_keys=True),
                    None if result.validation_score != result.validation_score else result.validation_score,
                    result.validation_std,
                    result.elapsed_seconds,
                    result.status,
                    result.hypothesis,
                    json.dumps(spec.to_dict(), sort_keys=True),
                    json.dumps(result.to_dict(), sort_keys=True),
                    json.dumps(profile.to_dict(), sort_keys=True),
                ),
            )

    def task_runs(self, task: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE lower(task) = lower(?) ORDER BY created_at, run_id",
                (task,),
            ).fetchall()
        return [dict(row) for row in rows]

    def cross_task_insights(
        self,
        task_type: str,
        metric: str,
        *,
        exclude_task: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Rank recipes within each task, then aggregate ranks across comparable tasks."""
        query = (
            "SELECT task, model, representations, hyperparameters, validation_score "
            "FROM runs WHERE task_type=? AND metric=? AND status='completed' "
            "AND validation_score IS NOT NULL"
        )
        params: list[Any] = [task_type, metric]
        if exclude_task:
            query += " AND lower(task) != lower(?)"
            params.append(exclude_task)
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(query, params).fetchall()]
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(row)
        maximize = metric != "mae"
        recipes: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for task_rows in by_task.values():
            ordered = sorted(
                task_rows,
                key=lambda row: row["validation_score"],
                reverse=maximize,
            )
            denominator = max(1, len(ordered) - 1)
            for rank, row in enumerate(ordered):
                percentile = 1.0 - rank / denominator
                key = (row["model"], row["representations"], row["hyperparameters"])
                recipes[key].append(percentile)
        ranked = sorted(
            recipes.items(),
            key=lambda item: (sum(item[1]) / len(item[1]), len(item[1])),
            reverse=True,
        )
        return [
            {
                "model": key[0],
                "representations": json.loads(key[1]),
                "hyperparameters": json.loads(key[2]),
                "mean_within_task_percentile": sum(values) / len(values),
                "evidence_tasks": len(values),
            }
            for key, values in ranked[:limit]
        ]

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            tasks = connection.execute("SELECT COUNT(DISTINCT task) FROM runs").fetchone()[0]
            completed = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE status='completed'"
            ).fetchone()[0]
        return {"runs": total, "tasks": tasks, "completed": completed}
