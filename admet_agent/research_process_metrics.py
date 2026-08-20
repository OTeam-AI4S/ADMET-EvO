"""Pure extraction of auditable autonomous-research process metrics.

Inputs are JSON-shaped discovery summaries, evidence-ledger records, and
action outcomes.  Test-like keys are rejected recursively before extraction.
Missing evidence stays in conservative denominators and is reported
explicitly; it is never silently converted into success.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

JsonRecord = Mapping[str, Any]
_PAIR_AXES = frozenset({"feature", "model", "data"})
_RESOLVED = frozenset({"supported", "refuted"})
_KNOWN_STATUSES = frozenset({"supported", "refuted", "inconclusive"})
_SAFE_TEST_KEYS = frozenset({"falsification_test", "falsification_tests"})


def _forbidden_test_paths(value: Any, path: str = "input") -> tuple[str, ...]:
    forbidden = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            text = str(key)
            normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text.strip()).lower()
            child_path = f"{path}.{text}"
            tokens = tuple(
                token for token in re.split(r"[^a-z0-9]+", normalized) if token
            )
            safe_negative_test_sentinel = normalized == "test_loaded" and child is False
            if (
                normalized not in _SAFE_TEST_KEYS
                and not safe_negative_test_sentinel
                and any(
                    token in {"test", "testing", "heldout", "holdout"}
                    for token in tokens
                )
            ):
                forbidden.append(child_path)
            forbidden.extend(_forbidden_test_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            forbidden.extend(_forbidden_test_paths(child, f"{path}[{index}]"))
    return tuple(sorted(set(forbidden)))


def _assert_discovery_only(*values: Any) -> None:
    paths = tuple(
        path
        for index, value in enumerate(values)
        for path in _forbidden_test_paths(value, f"input[{index}]")
    )
    if paths:
        raise ValueError(f"Test-like keys are forbidden: {list(paths)}")


def _json_records(
    value: Any,
    *,
    wrapper_keys: tuple[str, ...],
    identifying_keys: frozenset[str],
) -> list[JsonRecord]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = []
        for item in value:
            rows.extend(
                _json_records(
                    item,
                    wrapper_keys=wrapper_keys,
                    identifying_keys=identifying_keys,
                )
            )
        return rows
    if not isinstance(value, Mapping):
        raise TypeError("Metric inputs must be JSON objects or arrays of objects")
    for key in wrapper_keys:
        if key in value:
            return _json_records(
                value[key],
                wrapper_keys=wrapper_keys,
                identifying_keys=identifying_keys,
            )
    if identifying_keys & set(value):
        return [value]
    if value and all(isinstance(item, Mapping) for item in value.values()):
        return [item for item in value.values()]
    return [value]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _task(record: JsonRecord) -> str | None:
    direct = record.get("task")
    if direct:
        return str(direct)
    profile = record.get("profile")
    if isinstance(profile, Mapping) and profile.get("task"):
        return str(profile["task"])
    plan = record.get("research_plan")
    if isinstance(plan, Mapping) and plan.get("task"):
        return str(plan["task"])
    return None


@dataclass(frozen=True)
class RateMetric:
    """A conservative rate plus its observed-only rate and evidence coverage."""

    numerator: int
    denominator: int
    evaluable: int
    missing: int
    rate: float | None
    observed_rate: float | None
    coverage: float | None
    missing_ids: tuple[str, ...]
    definition: str

    @classmethod
    def build(
        cls,
        *,
        numerator: int,
        denominator: int,
        evaluable: int,
        missing_ids: Sequence[str],
        definition: str,
    ) -> RateMetric:
        missing = max(0, denominator - evaluable)
        if missing != len(set(missing_ids)):
            raise ValueError("missing_ids must identify every missing denominator item")
        return cls(
            numerator=numerator,
            denominator=denominator,
            evaluable=evaluable,
            missing=missing,
            rate=(numerator / denominator if denominator else None),
            observed_rate=(numerator / evaluable if evaluable else None),
            coverage=(evaluable / denominator if denominator else None),
            missing_ids=tuple(sorted(set(missing_ids))),
            definition=definition,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeCounts:
    supported: int
    refuted: int
    inconclusive: int
    missing_or_unknown: int
    valid_single_axis: int
    unique_outcomes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WasteAccounting:
    logical_attempts: int
    attempts_with_explicit_identity: int
    duplicate_task_action_attempts: int
    refuted_family_repeat_attempts: int
    overlapping_waste_attempts: int
    total_repeat_waste: int
    waste_rate: float | None
    ordering_coverage: float | None
    missing_order_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceAccounting:
    unique_actions: int
    total_fits: int
    primary_fits: int
    replicate_fits: int
    unclassified_fits: int
    completed_fits: int
    failed_fits: int
    fit_seconds: float
    completed_fit_seconds: float
    failed_fit_seconds: float
    fits_missing_elapsed: int
    racing_summaries: int
    planned_fits_avoided: int
    racing_missing_avoided_count_tasks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchProcessMetrics:
    tasks: tuple[str, ...]
    unique_task_actions: int
    single_axis_contrast_validity: RateMetric
    treatment_control_completeness: RateMetric
    falsification_resolution: RateMetric
    outcome_counts: OutcomeCounts
    negative_result_retention: RateMetric
    self_correction_rate: RateMetric
    preregistered_action_discipline: RateMetric
    repeat_waste: WasteAccounting
    resources: ResourceAccounting
    input_coverage: dict[str, int]
    version: str = "research-process-metrics-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Outcome:
    task: str
    action_id: str
    record: JsonRecord
    source: str
    source_index: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.task, self.action_id

    @property
    def label(self) -> str:
        return f"{self.task}/{self.action_id}"


@dataclass(frozen=True)
class _Attempt:
    task: str
    action_id: str
    family: str
    status: str | None
    attempt_id: str | None
    order: tuple[int, float | str] | None
    parent_action_id: str | None

    @property
    def identity(self) -> tuple[str, str]:
        return self.task, self.attempt_id or self.action_id

    @property
    def label(self) -> str:
        return f"{self.task}/{self.attempt_id or self.action_id}"


@dataclass(frozen=True)
class _Plan:
    task: str
    record: JsonRecord
    source: str
    source_index: int | None = None


def _plan_from_summary(summary: JsonRecord) -> JsonRecord | None:
    plan = summary.get("research_plan")
    return plan if isinstance(plan, Mapping) else None


def _outcome_record(value: JsonRecord) -> JsonRecord:
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping):
        return {**value, **evidence}
    return value


def _outcome_completeness(outcome: _Outcome) -> tuple[int, int, str]:
    fields = (
        "status",
        "axis",
        "single_axis_verified",
        "treatment_run_id",
        "control_run_id",
        "oriented_seed_deltas",
    )
    source_priority = {"ledger": 1, "summary": 2, "outcome": 3}[outcome.source]
    present = sum(field in outcome.record for field in fields)
    return present, source_priority, _canonical(outcome.record)


def _selected_action(plan: JsonRecord) -> tuple[str | None, JsonRecord | None]:
    selected = plan.get("selected_action_id")
    actions = plan.get("actions", [])
    action = next(
        (
            item
            for item in actions
            if isinstance(item, Mapping) and item.get("action_id") == selected
        ),
        None,
    )
    return (str(selected) if selected else None), action


def _extract_plans(
    summaries: Sequence[JsonRecord], ledger: Sequence[JsonRecord]
) -> list[_Plan]:
    plans = []
    for summary in summaries:
        task = _task(summary)
        plan = _plan_from_summary(summary)
        if task and plan is not None:
            plans.append(_Plan(task, plan, "summary"))
    for index, row in enumerate(ledger):
        if row.get("event_type") != "research_plan_created":
            continue
        payload = row.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        plan = payload.get("research_plan")
        task = payload.get("task")
        if not task and isinstance(plan, Mapping):
            task = plan.get("task")
        if task and isinstance(plan, Mapping):
            plans.append(_Plan(str(task), plan, "ledger", index))
    return plans


def _extract_outcomes(
    summaries: Sequence[JsonRecord],
    ledger: Sequence[JsonRecord],
    explicit: Sequence[JsonRecord],
) -> tuple[list[_Outcome], set[tuple[str, str]]]:
    rows = []
    ledger_keys: set[tuple[str, str]] = set()
    for summary in summaries:
        task = _task(summary)
        evidence = summary.get("action_evidence")
        if task and isinstance(evidence, Mapping) and evidence.get("action_id"):
            rows.append(_Outcome(task, str(evidence["action_id"]), evidence, "summary"))
    for index, record in enumerate(ledger):
        if record.get("event_type") != "research_action_evaluated":
            continue
        payload = record.get("payload", {})
        evidence = payload.get("evidence") if isinstance(payload, Mapping) else None
        task = payload.get("task") if isinstance(payload, Mapping) else None
        if task and isinstance(evidence, Mapping) and evidence.get("action_id"):
            outcome = _Outcome(
                str(task), str(evidence["action_id"]), evidence, "ledger", index
            )
            rows.append(outcome)
            ledger_keys.add(outcome.key)
    for value in explicit:
        record = _outcome_record(value)
        task = _task(record)
        action_id = record.get("action_id")
        if task and action_id:
            rows.append(_Outcome(task, str(action_id), record, "outcome"))
    return rows, ledger_keys


def _deduplicate_outcomes(rows: Sequence[_Outcome]) -> dict[tuple[str, str], _Outcome]:
    grouped: dict[tuple[str, str], list[_Outcome]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)
    return {
        key: max(values, key=_outcome_completeness) for key, values in grouped.items()
    }


def _attempt_from_record(task: str, action_id: str, record: JsonRecord) -> _Attempt:
    attempt_id = record.get("attempt_id")
    raw_order = record.get("sequence_index", record.get("attempt_index"))
    if isinstance(raw_order, (int, float)) and math.isfinite(float(raw_order)):
        order: tuple[int, float | str] | None = (0, float(raw_order))
    else:
        timestamp = record.get("timestamp")
        order = (1, str(timestamp)) if timestamp is not None else None
    parent = record.get("corrects_action_id", record.get("parent_action_id"))
    return _Attempt(
        task=task,
        action_id=action_id,
        family=str(record.get("action_family", action_id)),
        status=(str(record["status"]) if record.get("status") else None),
        attempt_id=(str(attempt_id) if attempt_id else None),
        order=order,
        parent_action_id=(str(parent) if parent else None),
    )


def _attempts(plans: Sequence[_Plan], outcomes: Sequence[_Outcome]) -> list[_Attempt]:
    rows = []
    for plan_evidence in plans:
        selected, action = _selected_action(plan_evidence.record)
        if selected:
            rows.append(
                _attempt_from_record(
                    plan_evidence.task,
                    selected,
                    action or plan_evidence.record,
                )
            )
    for outcome in outcomes:
        rows.append(
            _attempt_from_record(outcome.task, outcome.action_id, outcome.record)
        )
    unique: dict[tuple[str, str], _Attempt] = {}
    for row in rows:
        existing = unique.get(row.identity)
        if existing is None or (existing.status is None and row.status is not None):
            unique[row.identity] = row
    return list(unique.values())


def _waste(attempts: Sequence[_Attempt]) -> WasteAccounting:
    by_task_action: dict[tuple[str, str], list[_Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_task_action[(attempt.task, attempt.action_id)].append(attempt)
    duplicate_ids = set()
    for rows in by_task_action.values():
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                row.order is None,
                row.order or (2, ""),
                row.attempt_id or row.action_id,
            ),
        )
        duplicate_ids.update(row.identity for row in ordered_rows[1:])
    ordered = [row for row in attempts if row.order is not None]
    missing_order = [row.label for row in attempts if row.order is None]
    refuted_repeat_ids = set()
    for task in sorted({row.task for row in ordered}):
        prior_refuted: dict[str, set[str]] = defaultdict(set)
        task_rows = sorted(
            (row for row in ordered if row.task == task),
            key=lambda row: (row.order or (2, ""), row.attempt_id or row.action_id),
        )
        for row in task_rows:
            refuted_ids = prior_refuted[row.family]
            if refuted_ids and row.parent_action_id not in refuted_ids:
                refuted_repeat_ids.add(row.identity)
            if row.status == "refuted":
                refuted_ids.add(row.action_id)
    waste_ids = duplicate_ids | refuted_repeat_ids
    return WasteAccounting(
        logical_attempts=len(attempts),
        attempts_with_explicit_identity=sum(
            row.attempt_id is not None for row in attempts
        ),
        duplicate_task_action_attempts=len(duplicate_ids),
        refuted_family_repeat_attempts=len(refuted_repeat_ids),
        overlapping_waste_attempts=len(duplicate_ids & refuted_repeat_ids),
        total_repeat_waste=len(waste_ids),
        waste_rate=(len(waste_ids) / len(attempts) if attempts else None),
        ordering_coverage=(len(ordered) / len(attempts) if attempts else None),
        missing_order_ids=tuple(sorted(set(missing_order))),
    )


def _select_summaries(
    summaries: Sequence[JsonRecord],
) -> tuple[list[JsonRecord], int]:
    grouped: dict[str, list[JsonRecord]] = defaultdict(list)
    anonymous = []
    for summary in summaries:
        task = _task(summary)
        if task:
            grouped[task].append(summary)
        else:
            anonymous.append(summary)
    selected = []
    duplicates = 0
    for rows in grouped.values():
        duplicates += max(0, len(rows) - 1)
        selected.append(
            max(
                rows,
                key=lambda row: (
                    len(row.get("results", []))
                    if isinstance(row.get("results"), Sequence)
                    else 0,
                    _canonical(row),
                ),
            )
        )
    selected.extend(anonymous)
    return selected, duplicates


def _fit_key(result: JsonRecord) -> str:
    if result.get("run_id"):
        return f"run:{result['run_id']}"
    payload = {
        key: result.get(key)
        for key in (
            "task",
            "seed",
            "model",
            "representations",
            "hyperparameters",
            "data_policy",
            "status",
            "elapsed_seconds",
        )
    }
    return "sha:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _resources(
    summaries: Sequence[JsonRecord],
    ledger: Sequence[JsonRecord],
    unique_actions: int,
) -> ResourceAccounting:
    fits: dict[str, JsonRecord] = {}
    primary_seeds: dict[str, int] = {}
    for summary in summaries:
        task = _task(summary)
        seeds = summary.get("validation_seeds")
        if task and isinstance(seeds, Sequence) and seeds:
            primary_seeds[task] = int(seeds[0])
        results = summary.get("results", [])
        if isinstance(results, Sequence):
            for result in results:
                if isinstance(result, Mapping):
                    fits[_fit_key(result)] = result
    for row in ledger:
        if row.get("event_type") not in {"experiment_completed", "experiment_failed"}:
            continue
        payload = row.get("payload", {})
        result = payload.get("result") if isinstance(payload, Mapping) else None
        if isinstance(result, Mapping):
            fits.setdefault(_fit_key(result), result)

    completed = failed = primary = replicate = unclassified = missing_elapsed = 0
    completed_seconds = failed_seconds = 0.0
    for result in fits.values():
        status = str(result.get("status", "completed"))
        if status == "completed":
            completed += 1
        else:
            failed += 1
        elapsed = result.get("elapsed_seconds")
        try:
            elapsed_value = float(elapsed)
        except (TypeError, ValueError):
            elapsed_value = float("nan")
        if not math.isfinite(elapsed_value) or elapsed_value < 0:
            missing_elapsed += 1
        elif status == "completed":
            completed_seconds += elapsed_value
        else:
            failed_seconds += elapsed_value
        task = str(result.get("task", ""))
        if task not in primary_seeds or result.get("seed") is None:
            unclassified += 1
        elif int(result["seed"]) == primary_seeds[task]:
            primary += 1
        else:
            replicate += 1

    racing_count = avoided = 0
    racing_missing = []
    for summary in summaries:
        racing = summary.get("robust_racing")
        if not isinstance(racing, Mapping):
            continue
        racing_count += 1
        value = racing.get("planned_fits_avoided", racing.get("fits_avoided"))
        if value is None:
            decisions = racing.get("decisions", [])
            if isinstance(decisions, Sequence):
                values = [
                    item.get("planned_fits_avoided")
                    for item in decisions
                    if isinstance(item, Mapping) and item.get("decision") == "stop"
                ]
                if values and all(item is not None for item in values):
                    value = sum(int(item) for item in values)
        if value is None:
            racing_missing.append(_task(summary) or "unknown-task")
        else:
            avoided += max(0, int(value))
    return ResourceAccounting(
        unique_actions=unique_actions,
        total_fits=len(fits),
        primary_fits=primary,
        replicate_fits=replicate,
        unclassified_fits=unclassified,
        completed_fits=completed,
        failed_fits=failed,
        fit_seconds=completed_seconds + failed_seconds,
        completed_fit_seconds=completed_seconds,
        failed_fit_seconds=failed_seconds,
        fits_missing_elapsed=missing_elapsed,
        racing_summaries=racing_count,
        planned_fits_avoided=avoided,
        racing_missing_avoided_count_tasks=tuple(sorted(set(racing_missing))),
    )


def extract_research_process_metrics(
    search_summaries: Any = None,
    ledger_records: Any = None,
    action_outcomes: Any = None,
) -> ResearchProcessMetrics:
    """Extract conservative process metrics from validation-only JSON data."""

    _assert_discovery_only(search_summaries, ledger_records, action_outcomes)
    summaries_raw = _json_records(
        search_summaries,
        wrapper_keys=("summaries",),
        identifying_keys=frozenset({"profile", "research_plan", "results"}),
    )
    ledger = _json_records(
        ledger_records,
        wrapper_keys=("records", "events"),
        identifying_keys=frozenset({"event_type"}),
    )
    explicit = _json_records(
        action_outcomes,
        wrapper_keys=("outcomes", "actions"),
        identifying_keys=frozenset({"action_id", "evidence"}),
    )
    summaries, duplicate_summaries = _select_summaries(summaries_raw)
    plans = _extract_plans(summaries, ledger)
    outcome_rows, ledger_keys = _extract_outcomes(summaries, ledger, explicit)
    outcomes = _deduplicate_outcomes(outcome_rows)
    attempts = _attempts(plans, outcome_rows)
    action_keys = set(outcomes)
    plan_map: dict[tuple[str, str], list[tuple[_Plan, JsonRecord | None]]] = (
        defaultdict(list)
    )
    ledger_outcome_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for outcome in outcome_rows:
        if outcome.source == "ledger" and outcome.source_index is not None:
            ledger_outcome_indices[outcome.key].append(outcome.source_index)
    for plan_evidence in plans:
        selected, action = _selected_action(plan_evidence.record)
        if selected:
            key = (plan_evidence.task, selected)
            action_keys.add(key)
            plan_map[key].append((plan_evidence, action))

    outcome_values = list(outcomes.values())
    single_missing = [
        row.label for row in outcome_values if "single_axis_verified" not in row.record
    ]
    single_evaluable = len(outcome_values) - len(single_missing)
    single_valid = sum(
        row.record.get("single_axis_verified") is True for row in outcome_values
    )
    single_axis = RateMetric.build(
        numerator=single_valid,
        denominator=len(outcome_values),
        evaluable=single_evaluable,
        missing_ids=single_missing,
        definition="single_axis_verified true / unique task-action outcomes",
    )

    pair_rows = [
        row for row in outcome_values if str(row.record.get("axis")) in _PAIR_AXES
    ]
    pair_missing = [
        row.label
        for row in pair_rows
        if "treatment_run_id" not in row.record or "control_run_id" not in row.record
    ]
    pair_complete = sum(
        bool(row.record.get("treatment_run_id"))
        and bool(row.record.get("control_run_id"))
        and row.record.get("treatment_run_id") != row.record.get("control_run_id")
        for row in pair_rows
    )
    pair_metric = RateMetric.build(
        numerator=pair_complete,
        denominator=len(pair_rows),
        evaluable=len(pair_rows) - len(pair_missing),
        missing_ids=pair_missing,
        definition="distinct treatment and control IDs / pair-requiring outcomes",
    )

    valid_rows = [
        row for row in outcome_values if row.record.get("single_axis_verified") is True
    ]
    resolution_missing = [
        row.label
        for row in valid_rows
        if row.record.get("status") not in _KNOWN_STATUSES
    ]
    resolved = sum(row.record.get("status") in _RESOLVED for row in valid_rows)
    resolution = RateMetric.build(
        numerator=resolved,
        denominator=len(valid_rows),
        evaluable=len(valid_rows) - len(resolution_missing),
        missing_ids=resolution_missing,
        definition="(supported + refuted) / valid single-axis outcomes",
    )

    status_counts = {
        status: sum(row.record.get("status") == status for row in outcome_values)
        for status in _KNOWN_STATUSES
    }
    outcome_counts = OutcomeCounts(
        supported=status_counts["supported"],
        refuted=status_counts["refuted"],
        inconclusive=status_counts["inconclusive"],
        missing_or_unknown=len(outcome_values) - sum(status_counts.values()),
        valid_single_axis=single_valid,
        unique_outcomes=len(outcome_values),
    )

    refuted_rows = [
        row for row in outcome_values if row.record.get("status") == "refuted"
    ]
    retention_missing = [] if ledger else [row.label for row in refuted_rows]
    retained = sum(row.key in ledger_keys for row in refuted_rows) if ledger else 0
    retention = RateMetric.build(
        numerator=retained,
        denominator=len(refuted_rows),
        evaluable=(len(refuted_rows) if ledger else 0),
        missing_ids=retention_missing,
        definition="refuted task-actions retained in the supplied evidence ledger",
    )

    child_links = {
        (attempt.task, attempt.parent_action_id)
        for attempt in attempts
        if attempt.parent_action_id
    }
    correction_evaluable = corrected = 0
    correction_missing = []
    for row in refuted_rows:
        explicit_value = row.record.get("self_corrected")
        linked = row.key in child_links
        if isinstance(explicit_value, bool):
            correction_evaluable += 1
            corrected += bool(explicit_value) or linked
        elif linked:
            correction_evaluable += 1
            corrected += 1
        else:
            correction_missing.append(row.label)
    self_correction = RateMetric.build(
        numerator=corrected,
        denominator=len(refuted_rows),
        evaluable=correction_evaluable,
        missing_ids=correction_missing,
        definition="refuted task-actions with explicit or linked corrective follow-up",
    )

    discipline_missing = []
    disciplined = evaluable_plans = 0
    for task, action_id in sorted(action_keys):
        evidence = plan_map.get((task, action_id), [])
        if not evidence:
            representative = outcomes.get((task, action_id))
            if representative is not None and isinstance(
                representative.record.get("preregistered"), bool
            ):
                evaluable_plans += 1
                disciplined += bool(representative.record["preregistered"])
            else:
                discipline_missing.append(f"{task}/{action_id}")
            continue
        evaluable_plans += 1
        disciplined += all(
            plan_evidence.record.get("selected_action_id") == action_id
            and action is not None
            and bool(action.get("falsification_test"))
            and (
                plan_evidence.source == "summary"
                or (
                    plan_evidence.source_index is not None
                    and ledger_outcome_indices.get((task, action_id))
                    and plan_evidence.source_index
                    < min(ledger_outcome_indices[(task, action_id)])
                )
            )
            for plan_evidence, action in evidence
        )
    discipline = RateMetric.build(
        numerator=disciplined,
        denominator=len(action_keys),
        evaluable=evaluable_plans,
        missing_ids=discipline_missing,
        definition="attempted actions selected in a prior plan with a falsification test",
    )

    tasks = {task for task, _ in action_keys if task} | {
        task for task in (_task(summary) for summary in summaries) if task
    }
    return ResearchProcessMetrics(
        tasks=tuple(sorted(tasks)),
        unique_task_actions=len(action_keys),
        single_axis_contrast_validity=single_axis,
        treatment_control_completeness=pair_metric,
        falsification_resolution=resolution,
        outcome_counts=outcome_counts,
        negative_result_retention=retention,
        self_correction_rate=self_correction,
        preregistered_action_discipline=discipline,
        repeat_waste=_waste(attempts),
        resources=_resources(summaries, ledger, len(action_keys)),
        input_coverage={
            "search_summaries": len(summaries),
            "duplicate_task_summaries_ignored": duplicate_summaries,
            "ledger_records": len(ledger),
            "explicit_action_outcomes": len(explicit),
            "raw_action_outcome_records": len(outcome_rows),
            "unique_action_outcomes": len(outcomes),
        },
    )
