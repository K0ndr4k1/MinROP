"""Versioned compact reports and deterministic diffs. No execution or solving."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType

from .diagnostics import ConstraintCategory
from .search import SearchResult, _sequence_cost
from .search_diagnostics import SearchDiagnostics
from .search_trace import TraceEventKind, TraceSummary
from .solver import SequenceSolveResult
from .manifest import AnalysisManifest


class AnalysisKind(str, Enum):
    FIXED_SEQUENCE = "fixed_sequence"
    SEARCH = "search"
    SEARCH_DIAGNOSTICS = "search_diagnostics"


class ChangeCategory(str, Enum):
    OUTCOME = "outcome"
    SEARCH = "search"
    DIAGNOSTICS = "diagnostics"
    METADATA = "metadata"


class ComparabilityStatus(str, Enum):
    IDENTICAL_INPUTS = "identical_inputs"
    DIFFERENT_INPUTS = "different_inputs"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


# Each v1 section has explicit required fields. Unknown fields are ignored on
# loading, never promoted into executable objects or silently inferred values.
_FIXED = dict(status="solve_status", goals_satisfied="?bool", sequence_length="count",
              constraint_count="count", diagnostics_present="bool", diagnostic_solver_calls="count")
_SEARCH_OUTCOME = dict(status="search_status", found_sequence_indices="?indices", depth="?count",
                       cost="?cost", shortest_depth_proven="?bool", lowest_cost_at_depth_proven="?bool")
_SEARCH = {name: "count" for name in (
    "generated_candidates", "tested_candidates", "pruned_candidates", "prefix_checks", "solver_calls",
    "unknown_candidates", "unknown_prefix_checks", "unknown_solver_calls")}
_SEARCH["shallowest_unknown_depth"] = "?count"
_FIXED_DIAGNOSTIC = dict(categories_present="categories", required_categories="categories",
    conflict_ids="indices", conflict_categories="categories", conflict_size="count",
    irreducible="bool", minimize_requested="bool", partial="bool", unknown_solver_calls="count")
_BULK = {name: "count" for name in (
    "eligible_candidates", "candidates_considered", "candidates_diagnosed", "unknown_candidates",
    "pruned_prefixes", "unknown_prefix_checks", "coverage_numerator", "coverage_denominator",
    "replay_solver_calls", "diagnostic_solver_calls", "total_analysis_solver_calls")}
_BULK.update(max_diagnosed_candidates="?count", minimize_conflicts="bool", truncated="bool",
             category_counts="category_counts", required_category_counts="category_counts",
             unknown_category_counts="category_counts", conflict_patterns="patterns",
             irreducible_category_patterns="patterns", replay_consistent="bool")
_PROVENANCE = dict(candidate_count="?count", max_depth="?count", pruning="?bool", cost_enabled="?bool",
                   trace_recorded="bool", diagnostics_included="bool")
_TRACE = dict(event_count="count", final_status="search_status")
_REPORT_OUTCOME = dict(status="search_status", conclusive="bool")


def _require(condition, message):
    if not condition:
        raise ValueError(f"invalid analysis snapshot: {message}")


def _native(value):
    """Detached JSON-compatible values; reject objects and all nonfinite floats."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        _require(isfinite(value), "nonfinite float")
        return value
    if isinstance(value, Mapping):
        _require(all(type(key) is str for key in value), "object keys must be strings")
        return {key: _native(value[key]) for key in sorted(value)}
    if type(value) in (list, tuple):
        return [_native(item) for item in value]
    raise ValueError("invalid analysis snapshot: unsupported runtime object")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _value(value, spec, name):
    if spec.startswith("?"):
        return None if value is None else _value(value, spec[1:], name)
    if spec == "count":
        _require(type(value) is int and value >= 0, name)
    elif spec == "bool":
        _require(type(value) is bool, name)
    elif spec == "cost":
        _require(type(value) in (int, float) and value >= 0, name)
    elif spec in ("solve_status", "search_status"):
        choices = ("sat", "unsat", "unknown") if spec == "solve_status" else ("found", "not_found", "inconclusive")
        _require(type(value) is str and value in choices, name)
    elif spec in ("indices", "categories"):
        _require(type(value) is list, name)
        for item in value:
            if spec == "indices":
                _value(item, "count", name)
            else:
                _require(type(item) is str and item in {c.value for c in ConstraintCategory}, name)
        if spec == "categories":
            _require(len(set(value)) == len(value), name)
    elif spec == "category_counts":
        _require(type(value) is dict, name)
        _value(list(value), "categories", name)
        for count in value.values():
            _value(count, "count", name)
            _require(count > 0, name)
    elif spec == "patterns":
        _require(type(value) is list, name)
        value = [_section(item, dict(categories="categories", count="count", determined="bool"), name) for item in value]
        keys = []
        for item in value:
            _require(item["categories"] == sorted(item["categories"]), name)
            _require(item["count"] > 0 and item["determined"] is bool(item["categories"]), name)
            keys.append(tuple(item["categories"]))
        _require(len(set(keys)) == len(keys), name)
        _require(value == sorted(value, key=lambda item: (-item["count"], item["categories"])), name)
    return value


def _section(data, spec, name):
    _require(type(data) is dict, name)
    _require(spec.keys() <= data.keys(), f"missing required field in {name}")
    return {key: _value(data[key], field_type, f"{name}.{key}") for key, field_type in spec.items()}


def _normalise(data):
    data = _native(data)
    _require(type(data) is dict, "root")
    required = {"schema_version", "analysis_kind", "producer_version", "outcome", "search_summary",
                "trace_summary", "diagnostic_summary", "provenance", "metadata"}
    _require(required <= data.keys(), "missing required top-level field")
    _require(type(data["schema_version"]) is int and data["schema_version"] in (1, 2), "unsupported schema_version")
    if data["schema_version"] == 2:
        required.update(("manifest", "semantic_fingerprint", "analysis_fingerprint"))
        _require(required <= data.keys(), "missing v2 field")
    _require(type(data["analysis_kind"]) is str and data["analysis_kind"] in {kind.value for kind in AnalysisKind}, "analysis kind")
    _require(data["producer_version"] is None or type(data["producer_version"]) is str, "producer version")
    data = {key: data[key] for key in required}
    kind = data["analysis_kind"]
    fixed = kind == AnalysisKind.FIXED_SEQUENCE.value
    search = kind == AnalysisKind.SEARCH.value
    outcome = data["outcome"] = _section(data["outcome"], _FIXED if fixed else _SEARCH_OUTCOME if search else _REPORT_OUTCOME, "outcome")
    provenance = data["provenance"] = _section(data["provenance"], _PROVENANCE, "provenance")
    metadata = data["metadata"]
    _require(type(metadata) is dict and all(item is None or type(item) in (str, int, float, bool)
                                           for item in metadata.values()), "metadata must contain scalar JSON values")
    stats = data["search_summary"]
    trace = data["trace_summary"]
    diagnostic = data["diagnostic_summary"]
    _require(provenance["diagnostics_included"] is (diagnostic is not None), "diagnostics provenance")
    if trace is not None:
        _require(search and provenance["trace_recorded"], "unexpected trace summary")
        trace = data["trace_summary"] = _section(trace, _TRACE, "trace_summary")
        _require(trace["event_count"] >= 1 and trace["final_status"] == outcome["status"], "trace summary")
    if fixed:
        _require(stats is None and trace is None and not provenance["trace_recorded"], "fixed sequence sections")
        _require(outcome["sequence_length"] > 0, "empty fixed sequence")
        expected = None if outcome["status"] == "unknown" else outcome["status"] == "sat"
        _require(outcome["goals_satisfied"] is expected, "goals_satisfied")
        _require(outcome["diagnostics_present"] is (diagnostic is not None), "diagnostic presence")
        if diagnostic is None:
            _require(outcome["diagnostic_solver_calls"] == 0, "unexpected diagnostic calls")
        else:
            _require(outcome["status"] == "unsat", "diagnostics require UNSAT")
            diagnostic = data["diagnostic_summary"] = _section(diagnostic, _FIXED_DIAGNOSTIC, "diagnostic_summary")
            ids = diagnostic["conflict_ids"]
            _require(ids == sorted(set(ids)) and len(ids) == diagnostic["conflict_size"] > 0 and
                     max(ids) < outcome["constraint_count"], "conflict ids")
            _require(set(diagnostic["required_categories"]) <= set(diagnostic["categories_present"]), "required categories")
            _require(set(diagnostic["conflict_categories"]) <= set(diagnostic["categories_present"]), "conflict categories")
            _require(not diagnostic["irreducible"] or diagnostic["minimize_requested"], "irreducibility")
            _require(diagnostic["unknown_solver_calls"] <= outcome["diagnostic_solver_calls"] and
                     diagnostic["partial"] is (diagnostic["unknown_solver_calls"] > 0), "diagnostic UNKNOWN")
    elif search:
        stats = data["search_summary"] = _section(stats, _SEARCH, "search_summary")
        _require(stats["solver_calls"] == stats["tested_candidates"] + stats["prefix_checks"], "solver calls")
        _require(stats["unknown_solver_calls"] == stats["unknown_candidates"] + stats["unknown_prefix_checks"], "UNKNOWN calls")
        _require(stats["generated_candidates"] >= stats["tested_candidates"] >= stats["unknown_candidates"], "candidate counts")
        _require(stats["unknown_prefix_checks"] <= stats["prefix_checks"] and
                 stats["pruned_candidates"] <= stats["tested_candidates"], "prefix counts")
        shallow = stats["shallowest_unknown_depth"]
        _require((shallow is None) == (stats["unknown_candidates"] == 0), "shallow UNKNOWN")
        _require(shallow is None or shallow > 0, "UNKNOWN depth")
        if shallow is not None and provenance["max_depth"] is not None:
            _require(shallow <= provenance["max_depth"], "UNKNOWN exceeds max_depth")
        if outcome["status"] == "found":
            _require(outcome["depth"] is not None and outcome["depth"] > 0 and outcome["cost"] is not None, "FOUND depth/cost")
            _require(type(outcome["shortest_depth_proven"]) is bool and type(outcome["lowest_cost_at_depth_proven"]) is bool, "FOUND flags")
            _require(stats["tested_candidates"] > stats["unknown_candidates"], "FOUND requires a tested SAT")
            _require(outcome["shortest_depth_proven"] is (shallow is None or shallow >= outcome["depth"]), "shortest flag")
            indices = outcome["found_sequence_indices"]
            _require((indices is not None) is provenance["trace_recorded"], "FOUND identity availability")
            if indices is not None:
                _require(provenance["trace_recorded"] and len(indices) == outcome["depth"], "FOUND identity/depth")
                if provenance["candidate_count"] is not None:
                    _require(all(index < provenance["candidate_count"] for index in indices), "candidate index bound")
            if provenance["max_depth"] is not None:
                _require(outcome["depth"] <= provenance["max_depth"], "FOUND exceeds max_depth")
        else:
            _require(all(outcome[key] is None for key in _SEARCH_OUTCOME if key != "status"), "non-FOUND fields")
            _require(stats["generated_candidates"] == stats["tested_candidates"], "unexamined generated candidates")
        _check_search_status(outcome["status"], stats["unknown_candidates"])
        _require(provenance["trace_recorded"] is (trace is not None), "trace presence")
    else:
        _require(stats is None and trace is None and diagnostic is not None, "report sections")
        _require(outcome["conclusive"] is (outcome["status"] != "inconclusive"), "conclusive")
    if not fixed and diagnostic is not None:
        diagnostic = data["diagnostic_summary"] = _section(diagnostic, _BULK, "diagnostic_summary")
        eligible, considered, diagnosed = (diagnostic[name] for name in ("eligible_candidates", "candidates_considered", "candidates_diagnosed"))
        _require(diagnosed <= considered <= eligible, "diagnostic coverage")
        _require((diagnostic["coverage_numerator"], diagnostic["coverage_denominator"]) == (diagnosed, eligible), "coverage fractions")
        limit = diagnostic["max_diagnosed_candidates"]
        expected = 0 if outcome["status"] == "found" else eligible if limit is None else min(eligible, limit)
        _require(considered == expected, "diagnostic budget")
        _require(diagnostic["truncated"] is (outcome["status"] != "found" and considered < eligible), "truncated")
        _require(diagnostic["replay_solver_calls"] == considered and
                 diagnostic["total_analysis_solver_calls"] == considered + diagnostic["diagnostic_solver_calls"], "analysis calls")
        _require(diagnostic["replay_consistent"] is (considered == diagnosed), "replay consistency")
        _check_search_status(outcome["status"], diagnostic["unknown_candidates"])
        for field in ("category_counts", "required_category_counts", "unknown_category_counts"):
            _require(all(count <= diagnosed for count in diagnostic[field].values()), field)
        for category in set(diagnostic["required_category_counts"]) | set(diagnostic["unknown_category_counts"]):
            _require(diagnostic["required_category_counts"].get(category, 0) + diagnostic["unknown_category_counts"].get(category, 0)
                     <= diagnostic["category_counts"].get(category, 0), "category counts inconsistent")
        _require(sum(pattern["count"] for pattern in diagnostic["conflict_patterns"]) == diagnosed, "pattern total")
        _require(sum(pattern["count"] for pattern in diagnostic["irreducible_category_patterns"]) <= diagnosed, "irreducible pattern total")
        _require(diagnostic["minimize_conflicts"] or not diagnostic["irreducible_category_patterns"], "unrequested reduction")
        if stats is not None:
            _require(diagnostic["unknown_candidates"] == stats["unknown_candidates"] and
                     diagnostic["unknown_prefix_checks"] == stats["unknown_prefix_checks"] and
                     diagnostic["pruned_prefixes"] == stats["pruned_candidates"], "report/search statistics")
            _require(eligible == stats["tested_candidates"] - stats["unknown_candidates"] - int(outcome["status"] == "found"), "eligible count")
    if data["schema_version"] == 2:
        manifest = data["manifest"]
        if manifest is None:
            _require(data["semantic_fingerprint"] is None and data["analysis_fingerprint"] is None,
                     "fingerprints require manifest")
        else:
            manifest = AnalysisManifest.from_dict(manifest)
            _require((data["semantic_fingerprint"], data["analysis_fingerprint"]) ==
                     (manifest.semantic_fingerprint, manifest.analysis_fingerprint), "manifest fingerprint mismatch")
            _check_manifest(data, manifest)
            data["manifest"] = manifest.to_dict()
    return data


def _check_manifest(data, manifest):
    """Check only identities/bounds present in both summaries; never replay."""
    fixed = data["analysis_kind"] == AnalysisKind.FIXED_SEQUENCE.value
    _require(manifest.analysis_kind == ("fixed_sequence" if fixed else "search"), "manifest kind mismatch")
    outcome, provenance = data["outcome"], data["provenance"]
    semantic, analysis = manifest.semantic_options, manifest.analysis_options
    count = len(manifest.candidates)
    diagnostic = data["diagnostic_summary"]
    if fixed:
        _require(outcome["sequence_length"] == count, "manifest sequence length")
        if diagnostic is not None:
            _require(analysis["diagnose"] and analysis["minimize_conflict"] == diagnostic["minimize_requested"],
                     "manifest diagnostic configuration")
    else:
        for key, expected in (("candidate_count", count), ("max_depth", semantic["max_depth"]),
                              ("pruning", analysis["pruning"]), ("cost_enabled", analysis["cost_mode"] != "default")):
            _require(provenance[key] is None or provenance[key] == expected, f"manifest {key} mismatch")
        _require(provenance["trace_recorded"] == analysis["record_trace"], "manifest trace mismatch")
        if outcome.get("depth") is not None:
            _require(count > 0 and outcome["depth"] <= semantic["max_depth"], "manifest depth bound")
        indices = outcome.get("found_sequence_indices")
        if indices is not None:
            _require(all(index < count for index in indices), "manifest candidate index bound")
            _require(_sequence_cost(tuple(indices), semantic["candidate_costs"]) == outcome["cost"],
                     "manifest FOUND cost mismatch")
        shallow = (data["search_summary"] or {}).get("shallowest_unknown_depth")
        if shallow is not None:
            _require(shallow <= semantic["max_depth"], "manifest UNKNOWN depth bound")
        if diagnostic is not None:
            for key in ("max_diagnosed_candidates", "minimize_conflicts"):
                _require(analysis[key] == diagnostic[key], f"manifest {key} mismatch")


def _check_search_status(status, unknown):
    _require(status != "inconclusive" or unknown > 0, "INCONCLUSIVE requires UNKNOWN")
    _require(status != "not_found" or unknown == 0, "NOT_FOUND has UNKNOWN")


@dataclass(frozen=True)
class AnalysisSnapshot:
    schema_version: int
    analysis_kind: AnalysisKind
    producer_version: str | None
    outcome: Mapping
    search_summary: Mapping | None
    trace_summary: Mapping | None
    diagnostic_summary: Mapping | None
    provenance: Mapping
    metadata: Mapping
    manifest: AnalysisManifest | None = None
    semantic_fingerprint: str | None = None
    analysis_fingerprint: str | None = None

    def __post_init__(self):
        for name in ("outcome", "search_summary", "trace_summary", "diagnostic_summary", "provenance", "metadata"):
            object.__setattr__(self, name, _freeze(_native(getattr(self, name))))

    def to_dict(self) -> dict:
        data = {"schema_version": self.schema_version, "analysis_kind": self.analysis_kind.value,
                "producer_version": self.producer_version,
                **{name: _native(getattr(self, name)) for name in (
                    "outcome", "search_summary", "trace_summary", "diagnostic_summary", "provenance", "metadata")}}
        if self.schema_version == 2:
            data.update(manifest=None if self.manifest is None else self.manifest.to_dict(),
                        semantic_fingerprint=self.semantic_fingerprint, analysis_fingerprint=self.analysis_fingerprint)
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "AnalysisSnapshot":
        normal = _normalise(data)
        normal["analysis_kind"] = AnalysisKind(normal["analysis_kind"])
        if normal.get("manifest") is not None:
            normal["manifest"] = AnalysisManifest.from_dict(normal["manifest"])
        return cls(**normal)

    def validate(self) -> bool:
        if self.schema_version == 1:
            _require(self.manifest is None and self.semantic_fingerprint is None and self.analysis_fingerprint is None,
                     "v1 cannot carry a manifest")
        _normalise(self.to_dict())
        return True


def create_analysis_snapshot(*, solve_result: SequenceSolveResult | None = None,
                              search_result: SearchResult | None = None,
                              search_diagnostics: SearchDiagnostics | None = None,
                              metadata: Mapping | None = None,
                              manifest: AnalysisManifest | None = None) -> AnalysisSnapshot:
    """Extract existing summaries only; never traverse the candidate history.

    Trace FOUND identity uses its terminal FOUND event, not object repr. No
    inputs are inferred. An optional validated manifest opts into schema v2;
    the existing manifest-free API retains its v1 representation.
    """
    if solve_result is not None and (search_result is not None or search_diagnostics is not None):
        raise ValueError("fixed sequence and search inputs cannot be mixed")
    if solve_result is None and search_result is None and search_diagnostics is None:
        raise ValueError("an analysis result is required")
    for value, cls in ((solve_result, SequenceSolveResult), (search_result, SearchResult), (search_diagnostics, SearchDiagnostics)):
        if value is not None and not isinstance(value, cls):
            raise TypeError(f"expected {cls.__name__}")
    provenance = dict(candidate_count=None, max_depth=None, pruning=None, cost_enabled=None,
                      trace_recorded=False, diagnostics_included=False)
    try:
        producer_version = version("minirop")
    except PackageNotFoundError:
        producer_version = None
    data = dict(schema_version=1, producer_version=producer_version, metadata={} if metadata is None else metadata,
                provenance=provenance, search_summary=None, trace_summary=None, diagnostic_summary=None)
    if solve_result is not None:
        result = solve_result
        data["analysis_kind"] = AnalysisKind.FIXED_SEQUENCE.value
        data["outcome"] = dict(status=result.status.value, goals_satisfied=result.goals_satisfied,
                               sequence_length=len(result.effect.steps), constraint_count=len(result.constraints),
                               diagnostics_present=result.diagnostics is not None,
                               diagnostic_solver_calls=result.diagnostic_solver_calls)
        diagnostic = result.diagnostics
        if diagnostic is not None:
            data["diagnostic_summary"] = dict(
                categories_present=[category.value for category in diagnostic.categories_present],
                required_categories=[category.value for category in diagnostic.categories_required],
                conflict_ids=list(diagnostic.conflict_ids), conflict_size=len(diagnostic.conflict_ids),
                conflict_categories=[category.value for category in diagnostic.conflict_categories],
                irreducible=diagnostic.irreducible, minimize_requested=diagnostic.minimize_requested,
                partial=diagnostic.partial, unknown_solver_calls=diagnostic.unknown_solver_calls)
    elif search_result is not None:
        result = search_result
        data["analysis_kind"] = AnalysisKind.SEARCH.value
        indices = None
        trace = result.trace
        if trace is not None:
            _require(trace.summary == TraceSummary.from_result(result), "source trace summary mismatch")
            _require(bool(trace.events) and trace.events[-1].kind is TraceEventKind.SEARCH_END and
                     trace.events[-1].summary == trace.summary, "source trace ending")
            provenance.update(candidate_count=trace.candidate_count, max_depth=trace.max_depth,
                              pruning=trace.pruning, cost_enabled=trace.cost_ordering, trace_recorded=True)
            data["trace_summary"] = dict(event_count=len(trace.events), final_status=trace.summary.final_status.value)
            if result.status.value == "found":
                _require(len(trace.events) >= 2 and trace.events[-2].kind is TraceEventKind.FOUND, "missing source FOUND")
                found = trace.events[-2]
                _require((found.depth, found.cost) == (result.depth, result.cost), "source FOUND mismatch")
                indices = list(found.sequence_indices)
        data["outcome"] = dict(status=result.status.value, found_sequence_indices=indices,
                               depth=result.depth, cost=result.cost, shortest_depth_proven=result.shortest_depth_proven,
                               lowest_cost_at_depth_proven=result.lowest_cost_at_depth_proven)
        data["search_summary"] = {key: getattr(result, key) for key in _SEARCH}
    else:
        data["analysis_kind"] = AnalysisKind.SEARCH_DIAGNOSTICS.value
        data["outcome"] = dict(status=search_diagnostics.search_status.value, conclusive=search_diagnostics.conclusive)
    if search_diagnostics is not None:
        report = search_diagnostics
        _require(data["outcome"]["status"] == report.search_status.value, "search/report status mismatch")
        for name in ("candidate_count", "max_depth"):
            _require(provenance[name] is None or provenance[name] == getattr(report, name), "search/report bounds mismatch")
            provenance[name] = getattr(report, name)
        if search_result is None:
            provenance["trace_recorded"] = True  # A Phase 11 report requires a recorded source trace.
        summary = {key: getattr(report, key) for key in _BULK if key not in (
            "category_counts", "required_category_counts", "unknown_category_counts", "conflict_patterns",
            "irreducible_category_patterns", "coverage_numerator", "coverage_denominator", "total_analysis_solver_calls", "replay_consistent")}
        for name in ("category_counts", "required_category_counts", "unknown_category_counts"):
            summary[name] = {category.value: count for category, count in getattr(report, name).items()}
        for name in ("conflict_patterns", "irreducible_category_patterns"):
            summary[name] = [pattern.to_dict() for pattern in getattr(report, name)]
        summary.update(coverage_numerator=report.candidates_diagnosed, coverage_denominator=report.eligible_candidates,
                       total_analysis_solver_calls=report.total_solver_calls,
                       replay_consistent=report.candidates_considered == report.candidates_diagnosed)
        data["diagnostic_summary"] = summary
    provenance["diagnostics_included"] = data["diagnostic_summary"] is not None
    if manifest is not None:
        _require(isinstance(manifest, AnalysisManifest), "AnalysisManifest required")
        manifest.validate()
        data.update(schema_version=2, manifest=manifest.to_dict(),
                    semantic_fingerprint=manifest.semantic_fingerprint,
                    analysis_fingerprint=manifest.analysis_fingerprint)
    return AnalysisSnapshot.from_dict(data)


def save_analysis_snapshot(snapshot: AnalysisSnapshot, path: str | Path) -> None:
    snapshot.validate()
    Path(path).write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_analysis_snapshot(path: str | Path) -> AnalysisSnapshot:
    return AnalysisSnapshot.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class FieldChange:
    field: str
    before: object
    after: object
    category: ChangeCategory
    semantic: bool
    before_present: bool = True
    after_present: bool = True

    def __post_init__(self):
        object.__setattr__(self, "before", _freeze(_native(self.before)))
        object.__setattr__(self, "after", _freeze(_native(self.after)))

    def to_dict(self):
        return dict(field=self.field, before=_native(self.before), after=_native(self.after),
                    category=self.category.value, semantic=self.semantic,
                    before_present=self.before_present, after_present=self.after_present)


def _semantic(section, key):
    if section == "outcome":
        return key not in ("constraint_count", "diagnostic_solver_calls", "diagnostics_present")
    if section == "search_summary":
        return key in ("unknown_candidates", "unknown_prefix_checks", "unknown_solver_calls", "shallowest_unknown_depth")
    if section == "diagnostic_summary":
        return key in ("required_categories", "conflict_ids", "conflict_categories", "conflict_size", "irreducible",
                       "partial", "unknown_solver_calls", "required_category_counts", "unknown_category_counts",
                       "conflict_patterns", "irreducible_category_patterns", "unknown_candidates", "replay_consistent")
    return False


@dataclass(frozen=True)
class SnapshotDiff:
    analysis_kind: AnalysisKind
    changes: tuple[FieldChange, ...]
    comparability_warnings: tuple[str, ...] = ()
    comparability: ComparabilityStatus = ComparabilityStatus.UNKNOWN
    analysis_configuration_changed: bool | None = None

    @property
    def input_changed(self) -> bool | None:
        if self.comparability is ComparabilityStatus.IDENTICAL_INPUTS:
            return False
        if self.comparability is ComparabilityStatus.DIFFERENT_INPUTS:
            return True
        return None

    @property
    def changed(self) -> bool:
        return bool(self.changes) or self.input_changed is True or self.analysis_configuration_changed is True

    @property
    def semantic_changed(self) -> bool:
        return any(change.semantic for change in self.changes)

    @property
    def operational_changed(self) -> bool:
        return any(not change.semantic and change.category is not ChangeCategory.METADATA for change in self.changes)

    def _group(self, category):
        return tuple(change for change in self.changes if change.category is category)

    @property
    def outcome_changes(self):
        return self._group(ChangeCategory.OUTCOME)

    @property
    def search_changes(self):
        return self._group(ChangeCategory.SEARCH)

    @property
    def diagnostic_changes(self):
        return self._group(ChangeCategory.DIAGNOSTICS)

    @property
    def metadata_changes(self):
        return self._group(ChangeCategory.METADATA)

    def to_dict(self):
        return dict(analysis_kind=self.analysis_kind.value, changed=self.changed,
                    semantic_changed=self.semantic_changed, operational_changed=self.operational_changed,
                    comparability=self.comparability.value, input_changed=self.input_changed,
                    analysis_configuration_changed=self.analysis_configuration_changed,
                    changes=[change.to_dict() for change in self.changes],
                    comparability_warnings=list(self.comparability_warnings))

    @classmethod
    def from_dict(cls, data):
        """Restore a recorded diff, without comparing snapshots again."""
        data = _native(data)
        required = {"analysis_kind", "changed", "semantic_changed", "operational_changed", "comparability",
                    "input_changed", "analysis_configuration_changed", "changes", "comparability_warnings"}
        _require(type(data) is dict and set(data) == required, "diff fields")
        for key in ("changed", "semantic_changed", "operational_changed"):
            _require(type(data[key]) is bool, key)
        _require(data["input_changed"] is None or type(data["input_changed"]) is bool, "input_changed")
        _require(type(data["changes"]) is list, "diff changes")
        _require(type(data["comparability_warnings"]) is list and
                 all(type(item) is str for item in data["comparability_warnings"]), "comparability warnings")
        changes = []
        for item in data["changes"]:
            _require(type(item) is dict and set(item) == {
                "field", "before", "after", "category", "semantic", "before_present", "after_present"}, "field change fields")
            changes.append(FieldChange(**{**item, "category": ChangeCategory(item["category"])}))
        result = cls(AnalysisKind(data["analysis_kind"]), tuple(changes), tuple(data["comparability_warnings"]),
                     ComparabilityStatus(data["comparability"]), data["analysis_configuration_changed"])
        result.validate()
        _require(result.to_dict() == data, "inconsistent derived diff flags")
        return result

    def validate(self, before: AnalysisSnapshot | None = None, after: AnalysisSnapshot | None = None) -> bool:
        _require(isinstance(self.analysis_kind, AnalysisKind), "diff kind")
        _require(isinstance(self.comparability, ComparabilityStatus), "comparability")
        _require(self.analysis_configuration_changed is None or type(self.analysis_configuration_changed) is bool,
                 "analysis configuration comparison")
        paths = []
        group_order = {"outcome": 0, "search_summary": 1, "trace_summary": 2,
                       "diagnostic_summary": 3, "provenance": 4, "metadata": 5, "producer_version": 6}
        categories = {"outcome": ChangeCategory.OUTCOME, "search_summary": ChangeCategory.SEARCH,
                      "trace_summary": ChangeCategory.SEARCH, "diagnostic_summary": ChangeCategory.DIAGNOSTICS,
                      "provenance": ChangeCategory.METADATA, "metadata": ChangeCategory.METADATA,
                      "producer_version": ChangeCategory.METADATA}
        ordering = []
        for change in self.changes:
            _require(isinstance(change.category, ChangeCategory) and type(change.semantic) is bool, "diff classification")
            _require(type(change.before_present) is bool and type(change.after_present) is bool, "diff field presence")
            _require((change.before_present or change.before is None) and (change.after_present or change.after is None),
                     "absent diff value must be None")
            _require(change.before != change.after or change.before_present != change.after_present, "unchanged diff field")
            _require(type(change.field) is str and change.field not in paths, "duplicate diff field")
            paths.append(change.field)
            section, _, key = change.field.partition(".")
            _require(section in categories and change.category is categories[section], "diff group")
            ordering.append((group_order[section], key))
            _require(change.semantic is _semantic(section, key), "diff semantic flag")
        _require(ordering == sorted(ordering), "diff field order")
        if before is not None or after is not None:
            _require(before is not None and after is not None, "both diff sources required")
            _require(self == compare_analysis_snapshots(before, after), "diff source mismatch")
        return True


def compare_analysis_snapshots(before: AnalysisSnapshot, after: AnalysisSnapshot) -> SnapshotDiff:
    before.validate()
    after.validate()
    if before.analysis_kind is not after.analysis_kind:
        raise ValueError("snapshots must have the same analysis kind")
    changes = []
    groups = (("outcome", ChangeCategory.OUTCOME), ("search_summary", ChangeCategory.SEARCH),
              ("trace_summary", ChangeCategory.SEARCH), ("diagnostic_summary", ChangeCategory.DIAGNOSTICS),
              ("provenance", ChangeCategory.METADATA), ("metadata", ChangeCategory.METADATA))
    for section, category in groups:
        left, right = getattr(before, section), getattr(after, section)
        # Section presence is operational (or informational), not solver truth.
        if (left is None) != (right is None):
            changes.append(FieldChange(section, left is not None, right is not None, category, False))
        for key in sorted(set(left or ()) | set(right or ())):
            old, new = (left or {}).get(key), (right or {}).get(key)
            old_present, new_present = key in (left or {}), key in (right or {})
            if old != new or old_present != new_present:
                changes.append(FieldChange(f"{section}.{key}", old, new, category, _semantic(section, key), old_present, new_present))
    if before.producer_version != after.producer_version:
        changes.append(FieldChange("producer_version", before.producer_version, after.producer_version, ChangeCategory.METADATA, False))
    warnings = []
    for section in ("provenance", "metadata"):
        old, new = getattr(before, section).get("candidate_count"), getattr(after, section).get("candidate_count")
        if old is not None and new is not None and old != new:
            warnings.append(f"{section}.candidate_count differs; candidate indices may refer to different universes")
    left, right = before.manifest, after.manifest
    comparability = ComparabilityStatus.UNKNOWN
    configuration_changed = None
    if left is not None and right is not None:
        configuration_changed = left.analysis_fingerprint != right.analysis_fingerprint
        if left.semantic_fingerprint != right.semantic_fingerprint:
            comparability = ComparabilityStatus.DIFFERENT_INPUTS
        elif left.complete and right.complete:
            comparability = ComparabilityStatus.IDENTICAL_INPUTS
        else:
            comparability = ComparabilityStatus.PARTIAL
    if comparability is ComparabilityStatus.DIFFERENT_INPUTS:
        warnings.append("analysis inputs differ; result changes do not alone establish a regression")
    elif comparability is ComparabilityStatus.PARTIAL:
        warnings.append("equal fingerprints cover incomplete inputs; full input identity is unknown")
    elif comparability is ComparabilityStatus.UNKNOWN:
        warnings.append("manifest unavailable; input identity is unknown")
    return SnapshotDiff(before.analysis_kind, tuple(changes), tuple(warnings), comparability, configuration_changed)
