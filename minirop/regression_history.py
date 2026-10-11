"""Describe recorded regression history, without re-running any analysis.

Run order is supplied by the caller. Gaps are explicit, metrics are recorded
observations (never measurements), and classifications do not change reports.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType

from .regression import (ImprovementReason, RegressionReason, RegressionReport,
                         RegressionStatus, RegressionWarning)
from .snapshot import AnalysisKind, ComparabilityStatus, SnapshotDiff, _freeze, _native


class HistoryPresence(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"
    NEW = "new"
    ABSENT = "absent"


class StabilityStatus(str, Enum):
    INSUFFICIENT_HISTORY = "insufficient_history"
    STABLE_PASS = "stable_pass"
    STABLE_REVIEW = "stable_review"
    NEW_FAILURE = "new_failure"
    PERSISTENT_FAILURE = "persistent_failure"
    RECOVERED = "recovered"
    FLAKY = "flaky"
    MIXED = "mixed"


class TrendStatus(str, Enum):
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    MIXED = "mixed"
    INSUFFICIENT_DATA = "insufficient_data"


class MetricDirection(str, Enum):
    LOWER_IS_CHEAPER = "lower_is_cheaper"


def _require(condition, message):
    if not condition:
        raise ValueError(f"invalid regression history: {message}")


def _run_id(value):
    _require(type(value) is str and bool(value) and value == value.strip(), "run_id")


def _export(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (SnapshotDiff, RegressionReport)):
        return value.to_dict()
    if is_dataclass(value):
        return {field.name: _export(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {_export(key): _export(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_export(item) for item in value]
    return value


class _Record:
    def to_dict(self):
        return _export(self)


@dataclass(frozen=True)
class HistoryRun(_Record):
    run_id: str
    report: RegressionReport
    label: str | None = None
    metadata: Mapping | None = None

    def __post_init__(self):
        _require(isinstance(self.report, RegressionReport), "RegressionReport required")
        # Detach even manually constructed reports containing mutable containers.
        # This restores existing records; it does not evaluate regression rules.
        self.report.validate()
        object.__setattr__(self, "report", RegressionReport.from_dict(self.report.to_dict()))
        metadata = _native({} if self.metadata is None else self.metadata)
        _require(type(metadata) is dict and all(item is None or type(item) in (str, int, float, bool)
                                               for item in metadata.values()), "scalar metadata")
        object.__setattr__(self, "metadata", _freeze(metadata))
        self.validate()

    def validate(self):
        _run_id(self.run_id)
        _require(self.label is None or type(self.label) is str, "label")
        _require(isinstance(self.report, RegressionReport), "report")
        self.report.validate()
        metadata = _native(self.metadata)
        _require(type(metadata) is dict and all(item is None or type(item) in (str, int, float, bool)
                                               for item in metadata.values()), "scalar metadata")
        return True


@dataclass(frozen=True)
class RegressionHistory(_Record):
    runs: tuple[HistoryRun, ...]
    history_schema_version: int = 1

    def __post_init__(self):
        _require(isinstance(self.runs, Sequence) and not isinstance(self.runs, (str, bytes)), "ordered runs required")
        object.__setattr__(self, "runs", tuple(self.runs))
        self.validate()

    def validate(self):
        _require(type(self.history_schema_version) is int and self.history_schema_version == 1, "unsupported history schema")
        _require(type(self.runs) is tuple, "immutable runs")
        seen = set()
        for run in self.runs:
            _require(isinstance(run, HistoryRun), "HistoryRun required")
            run.validate()
            _require(run.run_id not in seen, "duplicate run_id")
            seen.add(run.run_id)
        return True

    @classmethod
    def from_dict(cls, data):
        data = _native(data)
        _require(type(data) is dict and set(data) == {"history_schema_version", "runs"}, "history fields")
        _require(type(data["runs"]) is list, "runs list")
        runs = []
        for item in data["runs"]:
            _require(type(item) is dict and set(item) == {"run_id", "report", "label", "metadata"}, "run fields")
            runs.append(HistoryRun(item["run_id"], RegressionReport.from_dict(item["report"]), item["label"], item["metadata"]))
        return cls(tuple(runs), data["history_schema_version"])


def create_regression_history(runs) -> RegressionHistory:
    """Accept an explicitly ordered Sequence of HistoryRun or (run_id, report)."""
    _require(isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)), "ordered runs required")
    result = []
    for item in runs:
        if isinstance(item, HistoryRun):
            result.append(item)
        else:
            _require(type(item) in (tuple, list) and len(item) == 2, "run pair")
            result.append(HistoryRun(item[0], item[1]))
    return RegressionHistory(tuple(result))


def save_regression_history(history: RegressionHistory, path: str | Path) -> None:
    history.validate()
    Path(path).write_text(json.dumps(history.to_dict(), ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_regression_history(path: str | Path) -> RegressionHistory:
    return RegressionHistory.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CaseObservation(_Record):
    run_id: str
    presence: HistoryPresence
    status: RegressionStatus | None
    comparability: ComparabilityStatus | None
    reasons: tuple[RegressionReason, ...]
    improvements: tuple[ImprovementReason, ...]
    warnings: tuple[RegressionWarning, ...]
    semantic_changed: bool | None
    operational_changed: bool | None
    solver_calls_delta: int | None
    diff: SnapshotDiff | None


@dataclass(frozen=True)
class CaseTransition(_Record):
    from_run: str
    to_run: str
    before: RegressionStatus
    after: RegressionStatus


@dataclass(frozen=True)
class MetricObservation(_Record):
    run_id: str
    value: int | None


@dataclass(frozen=True)
class MetricSeries(_Record):
    case_id: str
    metric: str
    direction: MetricDirection | None
    observations: tuple[MetricObservation, ...]
    numeric_count: int
    first: int | None
    latest: int | None
    minimum: int | None
    maximum: int | None
    net_change: int | None
    has_gaps: bool
    trend: TrendStatus


# Fixed serialization order. Absolute current values require an explicit
# FieldChange.after; absent changes do NOT reveal the unchanged absolute value.
_METRICS = (
    "solver_calls", "solver_calls_delta", "generated_candidates", "generated_candidates_delta",
    "tested_candidates", "tested_candidates_delta", "pruned_candidates", "pruned_candidates_delta",
    "diagnostic_solver_calls", "diagnostic_solver_calls_delta",
)


def _metric_values(observation):
    values = {name: None for name in _METRICS}
    values["solver_calls_delta"] = observation.solver_calls_delta
    diff = observation.diff
    if diff is None:
        return values
    paths = {"search_summary.solver_calls": "solver_calls",
             "search_summary.generated_candidates": "generated_candidates",
             "search_summary.tested_candidates": "tested_candidates",
             "search_summary.pruned_candidates": "pruned_candidates",
             "diagnostic_summary.diagnostic_solver_calls": "diagnostic_solver_calls",
             "outcome.diagnostic_solver_calls": "diagnostic_solver_calls"}
    if diff.analysis_kind is AnalysisKind.SEARCH_DIAGNOSTICS:
        paths["diagnostic_summary.total_analysis_solver_calls"] = "solver_calls"
    for change in diff.changes:
        metric = paths.get(change.field)
        if metric is None:
            continue
        if change.after_present and type(change.after) is int:
            values[metric] = change.after
        if metric != "solver_calls" and change.before_present and change.after_present and \
                type(change.before) is int and type(change.after) is int:
            values[f"{metric}_delta"] = change.after - change.before
    return values


def _metric_series(case_id, name, observations):
    values = [item.value for item in observations if item.value is not None]
    direction = None if name.startswith("pruned_candidates") else MetricDirection.LOWER_IS_CHEAPER
    if len(values) < 2:
        trend = TrendStatus.INSUFFICIENT_DATA
    else:
        increasing = any(b > a for a, b in zip(values, values[1:]))
        decreasing = any(b < a for a, b in zip(values, values[1:]))
        if not increasing and not decreasing:
            trend = TrendStatus.STABLE
        elif direction is None or increasing and decreasing:
            trend = TrendStatus.MIXED
        else:
            trend = TrendStatus.DEGRADING if increasing else TrendStatus.IMPROVING
    return MetricSeries(case_id, name, direction, tuple(observations), len(values),
                        values[0] if values else None, values[-1] if values else None,
                        min(values) if values else None, max(values) if values else None,
                        values[-1] - values[0] if values else None,
                        len(values) < len(observations), trend)


@dataclass(frozen=True)
class CaseHistory(_Record):
    case_id: str
    observations: tuple[CaseObservation, ...]
    transitions: tuple[CaseTransition, ...]
    first_seen_run: str
    last_seen_run: str
    current_status: RegressionStatus
    has_gaps: bool
    continuous_since_first_seen: bool
    stability: StabilityStatus
    current_failure_streak: int
    current_review_streak: int
    current_pass_streak: int
    max_failure_streak: int
    first_failure_run: str | None
    last_failure_run: str | None
    last_recovery_run: str | None
    reason_counts: Mapping
    improvement_counts: Mapping
    warning_counts: Mapping
    recurring_reasons: tuple[RegressionReason, ...]
    current_failure_reasons: tuple[RegressionReason, ...]
    metric_series: tuple[MetricSeries, ...]

    def __post_init__(self):
        for name in ("reason_counts", "improvement_counts", "warning_counts"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class OverallObservation(_Record):
    run_id: str
    status: RegressionStatus


@dataclass(frozen=True)
class HistorySummary(_Record):
    runs: int
    cases: int
    current_pass: int
    current_fail: int
    current_review: int
    new_failures: int
    persistent_failures: int
    persistent_reviews: int
    recovered: int
    flaky: int
    mixed: int
    insufficient_history: int


@dataclass(frozen=True)
class HistoryAnalysis(_Record):
    run_count: int
    case_count: int
    case_histories: tuple[CaseHistory, ...]
    overall_timeline: tuple[OverallObservation, ...]
    overall_transitions: tuple[CaseTransition, ...]
    current_overall_status: RegressionStatus | None
    overall_failure_streak: int
    status_counts: Mapping
    persistent_failures: tuple[str, ...]
    persistent_reviews: tuple[str, ...]
    flaky_cases: tuple[str, ...]
    recovered_cases: tuple[str, ...]
    newly_failed_cases: tuple[str, ...]
    operational_trends: tuple[MetricSeries, ...]
    summary: HistorySummary

    def __post_init__(self):
        object.__setattr__(self, "status_counts", MappingProxyType(dict(self.status_counts)))


def _transitions(observations):
    return tuple(CaseTransition(a.run_id, b.run_id, a.status, b.status)
                 for a, b in zip(observations, observations[1:])
                 if a.status is not None and b.status is not None and a.status is not b.status)


def _streak(observations, status):
    count = 0
    for item in reversed(observations):
        if item.status is not status:
            break
        count += 1
    return count


def _counts(observations, field, enum):
    counts = {item: 0 for item in enum}
    for observation in observations:
        for item in getattr(observation, field):
            counts[item] += 1
    return {item: count for item, count in counts.items() if count}


def _classify(valid, continuous, transitions, failures):
    if len(valid) < 2:
        return StabilityStatus.INSUFFICIENT_HISTORY
    if not continuous:
        return StabilityStatus.MIXED
    statuses = {item.status for item in valid}
    latest = valid[-1].status
    if len(transitions) >= 2 and {RegressionStatus.PASS, RegressionStatus.FAIL} <= statuses:
        return StabilityStatus.FLAKY
    if latest is RegressionStatus.FAIL and failures >= 2:
        return StabilityStatus.PERSISTENT_FAILURE
    if latest is RegressionStatus.FAIL and all(item.status is not RegressionStatus.FAIL for item in valid[:-1]):
        return StabilityStatus.NEW_FAILURE
    if latest is RegressionStatus.PASS and RegressionStatus.FAIL in statuses:
        return StabilityStatus.RECOVERED
    if statuses == {RegressionStatus.PASS}:
        return StabilityStatus.STABLE_PASS
    if statuses == {RegressionStatus.REVIEW}:
        return StabilityStatus.STABLE_REVIEW
    return StabilityStatus.MIXED


def _case_history(case_id, observations):
    valid = tuple(item for item in observations if item.status is not None)
    first_index = next(index for index, item in enumerate(observations) if item.status is not None)
    continuous = all(item.status is not None for item in observations[first_index:])
    transitions = _transitions(observations)
    failure_streak = _streak(observations, RegressionStatus.FAIL)
    failures = [item for item in valid if item.status is RegressionStatus.FAIL]
    current, maximum = 0, 0
    for observation in observations:
        current = current + 1 if observation.status is RegressionStatus.FAIL else 0
        maximum = max(maximum, current)
    recovery_runs = [item.to_run for item in transitions
                     if item.before is RegressionStatus.FAIL and item.after is RegressionStatus.PASS]
    reason_counts = _counts(valid, "reasons", RegressionReason)
    common = set(observations[-1].reasons) if failure_streak else set()
    if failure_streak:
        for item in observations[-failure_streak:]:
            common.intersection_update(item.reasons)
    metric_rows = [_metric_values(item) for item in observations]
    series = tuple(_metric_series(case_id, metric, tuple(MetricObservation(item.run_id, row[metric])
                   for item, row in zip(observations, metric_rows))) for metric in _METRICS)
    return CaseHistory(case_id, tuple(observations), transitions, valid[0].run_id, valid[-1].run_id, valid[-1].status,
        not continuous, continuous, _classify(valid, continuous, transitions, failure_streak),
        failure_streak, _streak(observations, RegressionStatus.REVIEW), _streak(observations, RegressionStatus.PASS),
        maximum, failures[0].run_id if failures else None, failures[-1].run_id if failures else None,
        recovery_runs[-1] if recovery_runs else None, reason_counts,
        _counts(valid, "improvements", ImprovementReason), _counts(valid, "warnings", RegressionWarning),
        tuple(reason for reason in RegressionReason if reason_counts.get(reason, 0) >= 2),
        tuple(reason for reason in RegressionReason if reason in common), series)


def analyze_regression_history(history: RegressionHistory) -> HistoryAnalysis:
    """Describe immutable recorded conclusions; no comparison or policy replay."""
    _require(isinstance(history, RegressionHistory), "RegressionHistory required")
    history.validate()
    reports = [{case.case_id: case for case in run.report.case_results} for run in history.runs]
    case_ids = sorted({case_id for report in reports for case_id in report})
    cases = []
    for case_id in case_ids:
        observations = []
        for run, report in zip(history.runs, reports):
            case = report.get(case_id)
            if case is None:
                observations.append(CaseObservation(run.run_id, HistoryPresence.ABSENT, None, None, (), (), (), None, None, None, None))
            else:
                diff = case.diff
                observations.append(CaseObservation(run.run_id, HistoryPresence(case.presence.value), case.status,
                    case.comparability, case.reasons, case.improvements, case.warnings,
                    diff.semantic_changed if diff is not None else None,
                    diff.operational_changed if diff is not None else None, case.solver_calls_delta, diff))
        cases.append(_case_history(case_id, tuple(observations)))
    cases = tuple(cases)
    overall = tuple(OverallObservation(run.run_id, run.report.status) for run in history.runs)
    status_counts = {status: sum(case.current_status is status for case in cases) for status in RegressionStatus}
    classified = {status: tuple(case.case_id for case in cases if case.stability is status) for status in StabilityStatus}
    reviews = tuple(case.case_id for case in cases if case.current_review_streak >= 2)
    summary = HistorySummary(len(history.runs), len(cases), status_counts[RegressionStatus.PASS],
        status_counts[RegressionStatus.FAIL], status_counts[RegressionStatus.REVIEW],
        len(classified[StabilityStatus.NEW_FAILURE]), len(classified[StabilityStatus.PERSISTENT_FAILURE]), len(reviews),
        len(classified[StabilityStatus.RECOVERED]), len(classified[StabilityStatus.FLAKY]),
        len(classified[StabilityStatus.MIXED]), len(classified[StabilityStatus.INSUFFICIENT_HISTORY]))
    return HistoryAnalysis(len(history.runs), len(cases), cases, overall, _transitions(overall),
        overall[-1].status if overall else None, _streak(overall, RegressionStatus.FAIL), status_counts,
        classified[StabilityStatus.PERSISTENT_FAILURE], reviews, classified[StabilityStatus.FLAKY],
        classified[StabilityStatus.RECOVERED], classified[StabilityStatus.NEW_FAILURE],
        tuple(series for case in cases for series in case.metric_series), summary)
