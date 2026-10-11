"""Deterministic, post-analysis baseline pairing and regression policy.

No analysis is replayed. SnapshotDiff owns all field comparison and input
comparability; this module assigns policy meaning to its recorded changes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType

from .snapshot import (AnalysisKind, AnalysisSnapshot, ComparabilityStatus,
                       SnapshotDiff, compare_analysis_snapshots, _native)


class RegressionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"


class CasePresencePolicy(str, Enum):
    ALLOW = "allow"
    FAIL = "fail"
    REVIEW = "review"


class CasePresence(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"
    NEW = "new"


class RegressionReason(str, Enum):
    MISSING_CASE = "missing_case"
    NEW_CASE = "new_case"
    ANALYSIS_KIND_CHANGED = "analysis_kind_changed"
    INPUT_DIFFERENT = "input_different"
    INPUT_PARTIAL = "input_partial"
    INPUT_UNKNOWN = "input_unknown"
    SAT_UNSAT_FLIP = "sat_unsat_flip"
    STATUS_REGRESSION = "status_regression"
    SEARCH_LOST_FOUND_RESULT = "search_lost_found_result"
    SEARCH_BECAME_INCONCLUSIVE = "search_became_inconclusive"
    FOUND_SEQUENCE_CHANGED = "found_sequence_changed"
    DEPTH_INCREASE = "depth_increase"
    COST_INCREASE = "cost_increase"
    SHORTEST_PROOF_LOST = "shortest_proof_lost"
    COST_PROOF_LOST = "cost_proof_lost"
    NEW_UNKNOWN = "new_unknown"
    REQUIRED_CATEGORY_CHANGED = "required_category_changed"
    CONFLICT_PATTERN_CHANGED = "conflict_pattern_changed"
    DIAGNOSTIC_DETAIL_CHANGED = "diagnostic_detail_changed"
    SOLVER_CALL_BUDGET_EXCEEDED = "solver_call_budget_exceeded"


class ImprovementReason(str, Enum):
    FOUND_RESULT_GAINED = "found_result_gained"
    SEARCH_BECAME_CONCLUSIVE = "search_became_conclusive"
    SOLVE_BECAME_CONCLUSIVE = "solve_became_conclusive"
    DEPTH_DECREASE = "depth_decrease"
    COST_DECREASE = "cost_decrease"
    SHORTEST_PROOF_GAINED = "shortest_proof_gained"
    COST_PROOF_GAINED = "cost_proof_gained"
    UNKNOWN_REDUCED = "unknown_reduced"
    SOLVER_CALLS_DECREASED = "solver_calls_decreased"


class RegressionWarning(str, Enum):
    INPUT_CHANGED = "input_changed"
    PARTIAL_MANIFEST = "partial_manifest"
    MISSING_MANIFEST = "missing_manifest"
    CANDIDATE_COUNT_CHANGED = "candidate_count_changed"
    ANALYSIS_CONFIGURATION_CHANGED = "analysis_configuration_changed"
    COMPARABILITY_OVERRIDE = "comparability_override"
    FOUND_SEQUENCE_CHANGED = "found_sequence_changed"
    FOUND_SEQUENCE_UNAVAILABLE = "found_sequence_unavailable"
    COST_NOT_PROVEN = "cost_not_proven"
    NEW_UNKNOWN = "new_unknown"
    DIAGNOSTICS_CHANGED = "diagnostics_changed"
    SOLVER_CALLS_INCREASED = "solver_calls_increased"
    SEMANTIC_CHANGE_ALLOWED = "semantic_change_allowed"


def _require(condition, message):
    if not condition:
        raise ValueError(f"invalid regression data: {message}")


def _case_id(value):
    _require(type(value) is str and bool(value) and value.strip() == value, "case_id must be nonempty without edge whitespace")


def _schema(data, names):
    _require(isinstance(data, Mapping) and set(data) == set(names), "required/unknown fields")


def _ordered(values, cls):
    values = tuple(values)
    _require(all(isinstance(value, cls) for value in values), f"expected {cls.__name__}")
    return tuple(value for value in cls if value in values)


def _overall(statuses):
    statuses = tuple(statuses)
    if RegressionStatus.FAIL in statuses:
        return RegressionStatus.FAIL
    if RegressionStatus.REVIEW in statuses:
        return RegressionStatus.REVIEW
    return RegressionStatus.PASS


@dataclass(frozen=True)
class BaselineEntry:
    case_id: str
    snapshot: AnalysisSnapshot
    description: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        _require(type(self.tags) in (tuple, list), "tags must be a sequence")
        _require(all(type(tag) is str and bool(tag.strip()) for tag in self.tags), "nonempty string tags")
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))
        self.validate()

    def validate(self):
        _case_id(self.case_id)
        _require(isinstance(self.snapshot, AnalysisSnapshot), "AnalysisSnapshot required")
        self.snapshot.validate()
        _require(self.description is None or type(self.description) is str, "description")
        _require(type(self.tags) is tuple and all(type(tag) is str and bool(tag.strip()) for tag in self.tags), "tags")
        _require(self.tags == tuple(sorted(set(self.tags))), "canonical tags")
        return True

    def to_dict(self):
        return dict(case_id=self.case_id, snapshot=self.snapshot.to_dict(), description=self.description, tags=list(self.tags))

    @classmethod
    def from_dict(cls, data):
        _schema(data, ("case_id", "snapshot", "description", "tags"))
        return cls(data["case_id"], AnalysisSnapshot.from_dict(data["snapshot"]), data["description"], data["tags"])


@dataclass(frozen=True)
class RegressionBaseline:
    entries: tuple[BaselineEntry, ...]
    baseline_schema_version: int = 1

    def __post_init__(self):
        entries = tuple(self.entries)
        _require(all(isinstance(entry, BaselineEntry) for entry in entries), "BaselineEntry required")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.case_id)))
        self.validate()

    def validate(self):
        _require(type(self.baseline_schema_version) is int and self.baseline_schema_version == 1, "unsupported baseline schema")
        _require(type(self.entries) is tuple, "immutable entries")
        ids = []
        for entry in self.entries:
            _require(isinstance(entry, BaselineEntry), "BaselineEntry required")
            entry.validate()
            ids.append(entry.case_id)
        _require(ids == sorted(set(ids)), "duplicate or unordered case_id")
        return True

    def to_dict(self):
        return dict(baseline_schema_version=self.baseline_schema_version, entries=[entry.to_dict() for entry in self.entries])

    @classmethod
    def from_dict(cls, data):
        _schema(data, ("baseline_schema_version", "entries"))
        _require(type(data["entries"]) is list, "entries list")
        return cls(tuple(BaselineEntry.from_dict(entry) for entry in data["entries"]), data["baseline_schema_version"])


def create_regression_baseline(entries) -> RegressionBaseline:
    """Accept a case_id -> snapshot mapping or iterable of BaselineEntry.

    The iterable form detects duplicate IDs before any mapping can overwrite
    them. Description/tags are informational, never matching or rule inputs.
    """
    if isinstance(entries, Mapping):
        entries = tuple(BaselineEntry(key, value) for key, value in entries.items())
    return RegressionBaseline(tuple(entries))


def save_regression_baseline(baseline: RegressionBaseline, path: str | Path) -> None:
    baseline.validate()
    Path(path).write_text(json.dumps(baseline.to_dict(), ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_regression_baseline(path: str | Path) -> RegressionBaseline:
    return RegressionBaseline.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RegressionPolicy:
    policy_version: int = 1
    require_identical_inputs: bool = True
    fail_on_status_regression: bool = True
    fail_on_found_sequence_change: bool = False
    fail_on_depth_increase: bool = True
    fail_on_cost_increase: bool = True
    fail_on_optimality_loss: bool = True
    fail_on_new_unknown: bool = False
    fail_on_required_category_change: bool = False
    fail_on_conflict_pattern_change: bool = False
    max_solver_call_increase: int | None = None
    missing_case: CasePresencePolicy = CasePresencePolicy.FAIL
    new_case: CasePresencePolicy = CasePresencePolicy.REVIEW

    def __post_init__(self):
        self.validate()

    def validate(self):
        _require(type(self.policy_version) is int and self.policy_version == 1, "unsupported policy version")
        for field in fields(self):
            if field.name.startswith("fail_on_") or field.name == "require_identical_inputs":
                _require(type(getattr(self, field.name)) is bool, field.name)
        limit = self.max_solver_call_increase
        _require(limit is None or (type(limit) is int and limit >= 0), "solver call threshold")
        _require(isinstance(self.missing_case, CasePresencePolicy) and isinstance(self.new_case, CasePresencePolicy), "presence policy")
        return True

    def to_dict(self):
        return {field.name: (getattr(self, field.name).value if field.name in ("missing_case", "new_case")
                            else getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, data):
        _schema(data, (field.name for field in fields(cls)))
        values = dict(data)
        for name in ("missing_case", "new_case"):
            _require(type(values[name]) is str, "presence policy string")
            values[name] = CasePresencePolicy(values[name])
        return cls(**values)


@dataclass(frozen=True)
class CaseRegressionResult:
    case_id: str
    presence: CasePresence
    status: RegressionStatus
    comparability: ComparabilityStatus
    diff: SnapshotDiff | None
    reasons: tuple[RegressionReason, ...] = ()
    improvements: tuple[ImprovementReason, ...] = ()
    warnings: tuple[RegressionWarning, ...] = ()
    solver_calls_delta: int | None = None

    def __post_init__(self):
        for name in ("reasons", "improvements", "warnings"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def validate(self):
        _case_id(self.case_id)
        for value, cls in ((self.presence, CasePresence), (self.status, RegressionStatus), (self.comparability, ComparabilityStatus)):
            _require(isinstance(value, cls), cls.__name__)
        for name, cls in (("reasons", RegressionReason), ("improvements", ImprovementReason), ("warnings", RegressionWarning)):
            value = getattr(self, name)
            _require(value == _ordered(value, cls), f"{name} order/duplicates")
        _require(self.solver_calls_delta is None or type(self.solver_calls_delta) is int, "solver delta")
        if self.diff is not None:
            _require(self.presence is CasePresence.MATCHED and isinstance(self.diff, SnapshotDiff), "diff presence")
            self.diff.validate()
            _require(self.comparability is self.diff.comparability and self.solver_calls_delta is not None, "diff comparability/delta")
        else:
            _require(self.comparability is ComparabilityStatus.UNKNOWN and self.solver_calls_delta is None, "unpaired identity")
            expected = {CasePresence.NEW: RegressionReason.NEW_CASE, CasePresence.MISSING: RegressionReason.MISSING_CASE,
                        CasePresence.MATCHED: RegressionReason.ANALYSIS_KIND_CHANGED}[self.presence]
            _require(self.reasons == (expected,) and not self.improvements and not self.warnings, "unpaired reasons")
            if self.presence is CasePresence.MATCHED:
                _require(self.status is RegressionStatus.FAIL, "incompatible kinds must fail")
        return True

    def to_dict(self):
        return dict(case_id=self.case_id, presence=self.presence.value, status=self.status.value,
                    comparability=self.comparability.value, diff=None if self.diff is None else self.diff.to_dict(),
                    reasons=[reason.value for reason in self.reasons], improvements=[item.value for item in self.improvements],
                    warnings=[warning.value for warning in self.warnings], solver_calls_delta=self.solver_calls_delta)


@dataclass(frozen=True)
class RegressionSummary:
    total_cases: int
    matched_cases: int
    passed_cases: int
    failed_cases: int
    review_cases: int
    missing_cases: int
    new_cases: int
    semantic_changes: int
    operational_changes: int
    identical_input_cases: int
    different_input_cases: int
    partial_cases: int
    unknown_comparability_cases: int
    reason_counts: Mapping

    def __post_init__(self):
        object.__setattr__(self, "reason_counts", MappingProxyType(dict(self.reason_counts)))

    def to_dict(self):
        return {field.name: ({reason.value: count for reason, count in self.reason_counts.items()}
                            if field.name == "reason_counts" else getattr(self, field.name)) for field in fields(self)}


def _summarize(cases):
    matched = tuple(case for case in cases if case.presence is CasePresence.MATCHED)
    return RegressionSummary(
        total_cases=len(cases), matched_cases=len(matched),
        passed_cases=sum(case.status is RegressionStatus.PASS for case in cases),
        failed_cases=sum(case.status is RegressionStatus.FAIL for case in cases),
        review_cases=sum(case.status is RegressionStatus.REVIEW for case in cases),
        missing_cases=sum(case.presence is CasePresence.MISSING for case in cases),
        new_cases=sum(case.presence is CasePresence.NEW for case in cases),
        semantic_changes=sum(case.diff is not None and case.diff.semantic_changed for case in cases),
        operational_changes=sum(case.diff is not None and case.diff.operational_changed for case in cases),
        identical_input_cases=sum(case.comparability is ComparabilityStatus.IDENTICAL_INPUTS for case in matched),
        different_input_cases=sum(case.comparability is ComparabilityStatus.DIFFERENT_INPUTS for case in matched),
        partial_cases=sum(case.comparability is ComparabilityStatus.PARTIAL for case in matched),
        unknown_comparability_cases=sum(case.comparability is ComparabilityStatus.UNKNOWN for case in matched),
        reason_counts={reason: count for reason in RegressionReason
                       if (count := sum(reason in case.reasons for case in cases))})


@dataclass(frozen=True)
class RegressionReport:
    baseline_cases: tuple[str, ...]
    current_cases: tuple[str, ...]
    case_results: tuple[CaseRegressionResult, ...]
    policy: RegressionPolicy
    summary: RegressionSummary
    status: RegressionStatus

    def __post_init__(self):
        for name in ("baseline_cases", "current_cases", "case_results"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def matched_cases(self):
        return tuple(case.case_id for case in self.case_results if case.presence is CasePresence.MATCHED)

    @property
    def missing_cases(self):
        return tuple(case.case_id for case in self.case_results if case.presence is CasePresence.MISSING)

    @property
    def new_cases(self):
        return tuple(case.case_id for case in self.case_results if case.presence is CasePresence.NEW)

    def to_dict(self):
        return dict(baseline_cases=list(self.baseline_cases), current_cases=list(self.current_cases),
                    matched_cases=list(self.matched_cases), missing_cases=list(self.missing_cases), new_cases=list(self.new_cases),
                    case_results=[case.to_dict() for case in self.case_results], policy=self.policy.to_dict(),
                    summary=self.summary.to_dict(), status=self.status.value)

    @classmethod
    def from_dict(cls, data):
        """Restore recorded conclusions and validate structure; never evaluate."""
        data = _native(data)
        _schema(data, ("baseline_cases", "current_cases", "matched_cases", "missing_cases", "new_cases",
                       "case_results", "policy", "summary", "status"))
        for key in ("baseline_cases", "current_cases", "matched_cases", "missing_cases", "new_cases", "case_results"):
            _require(type(data[key]) is list, f"{key} list")
        cases = []
        for item in data["case_results"]:
            _schema(item, (field.name for field in fields(CaseRegressionResult)))
            values = dict(item)
            for key, enum in (("presence", CasePresence), ("status", RegressionStatus), ("comparability", ComparabilityStatus)):
                values[key] = enum(values[key])
            for key, enum in (("reasons", RegressionReason), ("improvements", ImprovementReason), ("warnings", RegressionWarning)):
                _require(type(values[key]) is list, f"{key} list")
                values[key] = tuple(enum(value) for value in values[key])
            values["diff"] = None if values["diff"] is None else SnapshotDiff.from_dict(values["diff"])
            cases.append(CaseRegressionResult(**values))
        summary = data["summary"]
        _schema(summary, (field.name for field in fields(RegressionSummary)))
        _require(type(summary["reason_counts"]) is dict, "reason counts")
        summary = RegressionSummary(**{**summary, "reason_counts": {
            RegressionReason(key): value for key, value in summary["reason_counts"].items()}})
        result = cls(tuple(data["baseline_cases"]), tuple(data["current_cases"]), tuple(cases),
                     RegressionPolicy.from_dict(data["policy"]), summary, RegressionStatus(data["status"]))
        result.validate()
        _require(result.to_dict() == data, "inconsistent derived report fields")
        return result

    def validate(self, baseline=None, current=None):
        """Validate structure; optional sources additionally re-evaluate policy.

        Re-evaluation uses only existing snapshots and diffs, never a Solver.
        """
        _require(isinstance(self.policy, RegressionPolicy), "policy")
        self.policy.validate()
        for ids in (self.baseline_cases, self.current_cases):
            for case_id in ids:
                _case_id(case_id)
            _require(list(ids) == sorted(set(ids)), "case ID order/duplicates")
        left, right = set(self.baseline_cases), set(self.current_cases)
        matched_ids = left & right
        _require(all(isinstance(case, CaseRegressionResult) for case in self.case_results), "case result")
        _require([case.case_id for case in self.case_results] == sorted(left | right), "case coverage/order")
        for case in self.case_results:
            case.validate()
            expected = CasePresence.MATCHED if case.case_id in matched_ids else CasePresence.MISSING if case.case_id in left else CasePresence.NEW
            _require(case.presence is expected, "case presence")
            if expected is not CasePresence.MATCHED:
                action = self.policy.missing_case if expected is CasePresence.MISSING else self.policy.new_case
                _require(case.status is _presence_status(action), "case presence policy")
        _require(isinstance(self.summary, RegressionSummary), "summary")
        for field in fields(self.summary):
            if field.name != "reason_counts":
                value = getattr(self.summary, field.name)
                _require(type(value) is int and value >= 0, "summary count")
        _require(all(isinstance(reason, RegressionReason) and type(count) is int and count > 0
                     for reason, count in self.summary.reason_counts.items()), "reason counts")
        _require(self.summary == _summarize(self.case_results), "summary mismatch")
        _require(isinstance(self.status, RegressionStatus) and self.status is _overall(case.status for case in self.case_results), "overall status")
        if baseline is not None or current is not None:
            _require(baseline is not None and current is not None, "both sources required")
            _require(self == evaluate_regression(baseline, current, policy=self.policy), "report source/policy mismatch")
        return True


def _presence_status(action):
    return {CasePresencePolicy.ALLOW: RegressionStatus.PASS, CasePresencePolicy.FAIL: RegressionStatus.FAIL,
            CasePresencePolicy.REVIEW: RegressionStatus.REVIEW}[action]


def _evaluate_diff(case_id, before, diff, policy):
    # ONLY SnapshotDiff decides which fields changed. Baseline reads supply
    # unchanged context (e.g. equal depth or proof flags), not a second diff.
    changes = {change.field: change for change in diff.changes}

    def pair(path, default=None):
        change = changes.get(path)
        if change is not None:
            return change.before, change.after
        section, key = path.split(".", 1)
        value = (getattr(before, section) or {}).get(key, default)
        return value, value

    reasons, improvements, warnings, decisions = [], [], [], []

    def rule(reason, fail=True, review=False):
        reasons.append(reason)
        decisions.append(RegressionStatus.FAIL if fail else RegressionStatus.REVIEW if review else RegressionStatus.PASS)
        if not fail and not review:
            warnings.append(RegressionWarning.SEMANTIC_CHANGE_ALLOWED)

    comparable = diff.comparability is ComparabilityStatus.IDENTICAL_INPUTS
    if not comparable:
        reason, warning = {
            ComparabilityStatus.DIFFERENT_INPUTS: (RegressionReason.INPUT_DIFFERENT, RegressionWarning.INPUT_CHANGED),
            ComparabilityStatus.PARTIAL: (RegressionReason.INPUT_PARTIAL, RegressionWarning.PARTIAL_MANIFEST),
            ComparabilityStatus.UNKNOWN: (RegressionReason.INPUT_UNKNOWN, RegressionWarning.MISSING_MANIFEST),
        }[diff.comparability]
        reasons.append(reason)
        warnings.append(warning)
        if policy.require_identical_inputs:
            decisions.append(RegressionStatus.REVIEW)
        else:
            warnings.append(RegressionWarning.COMPARABILITY_OVERRIDE)
    if any(path in changes for path in ("provenance.candidate_count", "metadata.candidate_count")):
        warnings.append(RegressionWarning.CANDIDATE_COUNT_CHANGED)
    if diff.analysis_configuration_changed:
        warnings.append(RegressionWarning.ANALYSIS_CONFIGURATION_CHANGED)

    strict = comparable or not policy.require_identical_inputs
    if strict:
        old_status, new_status = pair("outcome.status")
        if "outcome.status" in changes:
            if diff.analysis_kind is AnalysisKind.FIXED_SEQUENCE:
                if {old_status, new_status} == {"sat", "unsat"}:
                    rule(RegressionReason.SAT_UNSAT_FLIP, policy.fail_on_status_regression)
                elif new_status == "unknown":
                    rule(RegressionReason.STATUS_REGRESSION, policy.fail_on_status_regression)
                elif old_status == "unknown":
                    improvements.append(ImprovementReason.SOLVE_BECAME_CONCLUSIVE)
            else:
                if old_status == "found" and new_status != "found":
                    rule(RegressionReason.SEARCH_LOST_FOUND_RESULT, policy.fail_on_status_regression)
                if new_status == "inconclusive":
                    rule(RegressionReason.SEARCH_BECAME_INCONCLUSIVE, policy.fail_on_status_regression)
                elif new_status == "found":
                    improvements.append(ImprovementReason.FOUND_RESULT_GAINED)
                if old_status == "inconclusive" and new_status != "inconclusive":
                    improvements.append(ImprovementReason.SEARCH_BECAME_CONCLUSIVE)
        if old_status == new_status == "found" and diff.analysis_kind is AnalysisKind.SEARCH:
            if "outcome.found_sequence_indices" in changes:
                old, new = pair("outcome.found_sequence_indices")
                if old is not None and new is not None:
                    warnings.append(RegressionWarning.FOUND_SEQUENCE_CHANGED)
                    if policy.fail_on_found_sequence_change:
                        rule(RegressionReason.FOUND_SEQUENCE_CHANGED)
                else:
                    warnings.append(RegressionWarning.FOUND_SEQUENCE_UNAVAILABLE)
            old_depth, new_depth = pair("outcome.depth")
            if "outcome.depth" in changes:
                if new_depth > old_depth:
                    rule(RegressionReason.DEPTH_INCREASE, policy.fail_on_depth_increase)
                else:
                    improvements.append(ImprovementReason.DEPTH_DECREASE)
            if "outcome.cost" in changes and old_depth == new_depth:
                old, new = pair("outcome.cost")
                if new > old:
                    proven = pair("outcome.lowest_cost_at_depth_proven") == (True, True)
                    if not proven:
                        warnings.append(RegressionWarning.COST_NOT_PROVEN)
                    rule(RegressionReason.COST_INCREASE, policy.fail_on_cost_increase and proven,
                         review=policy.fail_on_cost_increase and not proven)
                else:
                    improvements.append(ImprovementReason.COST_DECREASE)
            for key, loss, gain in (
                ("shortest_depth_proven", RegressionReason.SHORTEST_PROOF_LOST, ImprovementReason.SHORTEST_PROOF_GAINED),
                ("lowest_cost_at_depth_proven", RegressionReason.COST_PROOF_LOST, ImprovementReason.COST_PROOF_GAINED),
            ):
                if f"outcome.{key}" in changes:
                    old, new = pair(f"outcome.{key}")
                    if old is True and new is False:
                        rule(loss, policy.fail_on_optimality_loss)
                    elif old is False and new is True:
                        improvements.append(gain)
        unknown_paths = ("search_summary.unknown_candidates", "search_summary.unknown_prefix_checks",
                         "diagnostic_summary.unknown_candidates", "diagnostic_summary.unknown_prefix_checks",
                         "diagnostic_summary.unknown_solver_calls")
        for path in unknown_paths:
            if path in changes:
                old, new = pair(path)
                if type(old) is int and type(new) is int:
                    if new > old:
                        warnings.append(RegressionWarning.NEW_UNKNOWN)
                        if policy.fail_on_new_unknown:
                            rule(RegressionReason.NEW_UNKNOWN)
                    elif new < old:
                        improvements.append(ImprovementReason.UNKNOWN_REDUCED)
        required = ("diagnostic_summary.required_categories", "diagnostic_summary.required_category_counts")
        patterns = ("diagnostic_summary.conflict_patterns", "diagnostic_summary.irreducible_category_patterns")
        # When the main conclusion changes, diagnostics can legitimately appear
        # or disappear with it. Do not turn a gained FOUND/conclusive result
        # into REVIEW merely because its former failure report disappeared.
        same_status = old_status == new_status
        if same_status and any(path in changes for path in required):
            rule(RegressionReason.REQUIRED_CATEGORY_CHANGED, policy.fail_on_required_category_change, review=True)
        if same_status and any(path in changes for path in patterns):
            rule(RegressionReason.CONFLICT_PATTERN_CHANGED, policy.fail_on_conflict_pattern_change, review=True)
        diagnostic_details = [change for change in diff.diagnostic_changes if change.semantic and change.field not in required + patterns + unknown_paths]
        if same_status and diagnostic_details:
            rule(RegressionReason.DIAGNOSTIC_DETAIL_CHANGED, False, review=True)
        if diff.diagnostic_changes:
            warnings.append(RegressionWarning.DIAGNOSTICS_CHANGED)

    # Total recorded solver work; no inferred primary calls for missing/new
    # cases. Fixed analyses have one original solve, so only extras can change.
    if diff.analysis_kind is AnalysisKind.FIXED_SEQUENCE:
        call_paths = ("outcome.diagnostic_solver_calls",)
    elif diff.analysis_kind is AnalysisKind.SEARCH:
        call_paths = ("search_summary.solver_calls", "diagnostic_summary.total_analysis_solver_calls")
    else:
        call_paths = ("diagnostic_summary.total_analysis_solver_calls",)
    delta = sum((new or 0) - (old or 0) for old, new in (pair(path, 0) for path in call_paths))
    if delta > 0:
        warnings.append(RegressionWarning.SOLVER_CALLS_INCREASED)
        if strict and policy.max_solver_call_increase is not None and delta > policy.max_solver_call_increase:
            rule(RegressionReason.SOLVER_CALL_BUDGET_EXCEEDED)
    elif delta < 0:
        improvements.append(ImprovementReason.SOLVER_CALLS_DECREASED)
    return CaseRegressionResult(case_id, CasePresence.MATCHED, _overall(decisions), diff.comparability, diff,
                                _ordered(reasons, RegressionReason), _ordered(improvements, ImprovementReason),
                                _ordered(warnings, RegressionWarning), delta)


def evaluate_regression(baseline: RegressionBaseline, current: Mapping[str, AnalysisSnapshot], *,
                        policy: RegressionPolicy | None = None) -> RegressionReport:
    """Pair exact case IDs, compare every case, then apply explicit policy.

    Invalid snapshots raise before evaluation. Matched but incompatible analysis
    kinds produce a FAIL case with no diff, so other cases are still evaluated.
    """
    _require(isinstance(baseline, RegressionBaseline), "RegressionBaseline required")
    baseline.validate()
    policy = RegressionPolicy() if policy is None else policy
    _require(isinstance(policy, RegressionPolicy), "RegressionPolicy required")
    policy.validate()
    _require(isinstance(current, Mapping), "current must map case IDs to snapshots")
    current = dict(current)
    for case_id, snapshot in current.items():
        _case_id(case_id)
        _require(isinstance(snapshot, AnalysisSnapshot), "current AnalysisSnapshot required")
        snapshot.validate()
    previous = {entry.case_id: entry.snapshot for entry in baseline.entries}
    results = []
    for case_id in sorted(previous.keys() | current.keys()):
        if case_id not in current or case_id not in previous:
            missing = case_id not in current
            results.append(CaseRegressionResult(case_id, CasePresence.MISSING if missing else CasePresence.NEW,
                _presence_status(policy.missing_case if missing else policy.new_case), ComparabilityStatus.UNKNOWN, None,
                (RegressionReason.MISSING_CASE if missing else RegressionReason.NEW_CASE,)))
        elif previous[case_id].analysis_kind is not current[case_id].analysis_kind:
            results.append(CaseRegressionResult(case_id, CasePresence.MATCHED, RegressionStatus.FAIL,
                ComparabilityStatus.UNKNOWN, None, (RegressionReason.ANALYSIS_KIND_CHANGED,)))
        else:
            diff = compare_analysis_snapshots(previous[case_id], current[case_id])
            results.append(_evaluate_diff(case_id, previous[case_id], diff, policy))
    results = tuple(results)
    report = RegressionReport(tuple(sorted(previous)), tuple(sorted(current)), results, policy,
                              _summarize(results), _overall(case.status for case in results))
    report.validate()
    return report
