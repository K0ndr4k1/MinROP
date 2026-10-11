"""Explicit post-search diagnostics over recorded SOLVED/UNSAT candidates only."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from .constraints import Equality
from .diagnostics import ConstraintCategory, UnsatDiagnostics
from .gadget import Gadget
from .search import SearchResult, SearchStatus
from .search_trace import TraceEventKind
from .solver import FinalGoals, SolveStatus, solve_sequence
from .state import CPUState


@dataclass(frozen=True)
class ConflictPattern:
    categories: tuple[ConstraintCategory, ...]
    count: int
    determined: bool

    def to_dict(self) -> dict:
        return {"categories": [category.value for category in self.categories],
                "count": self.count, "determined": self.determined}


@dataclass(frozen=True)
class CandidateDiagnosticSummary:
    solved_event_id: int
    sequence_indices: tuple[int, ...]
    depth: int
    cost: int | float
    replay_status: SolveStatus
    diagnostics: UnsatDiagnostics | None

    @property
    def constraint_count(self) -> int:
        return len(self.diagnostics.records) if self.diagnostics is not None else 0

    @property
    def categories_present(self) -> tuple[ConstraintCategory, ...]:
        return self.diagnostics.categories_present if self.diagnostics is not None else ()

    @property
    def required_categories(self) -> tuple[ConstraintCategory, ...]:
        return self.diagnostics.categories_required if self.diagnostics is not None else ()

    @property
    def conflict_size(self) -> int:
        return len(self.diagnostics.conflict_ids) if self.diagnostics is not None else 0

    @property
    def irreducible(self) -> bool | None:
        if self.diagnostics is None or not self.diagnostics.minimize_requested:
            return None
        return self.diagnostics.irreducible

    @property
    def diagnostic_solver_calls(self) -> int:
        return self.diagnostics.diagnostic_solver_calls if self.diagnostics is not None else 0

    def to_dict(self) -> dict:
        return {
            "solved_event_id": self.solved_event_id,
            "sequence_indices": list(self.sequence_indices), "depth": self.depth, "cost": self.cost,
            "original_status": SolveStatus.UNSAT.value, "replay_status": self.replay_status.value,
            "constraint_count": self.constraint_count,
            "categories_present": [category.value for category in self.categories_present],
            "required_categories": [category.value for category in self.required_categories],
            "conflict_size": self.conflict_size, "irreducible": self.irreducible,
            "diagnostic_solver_calls": self.diagnostic_solver_calls,
            "diagnostics": None if self.diagnostics is None else self.diagnostics.to_dict(),
        }


def _category_key(categories):
    return tuple(sorted(set(categories), key=lambda category: category.value))


def _patterns(counter):
    ordered = sorted(counter.items(), key=lambda item: (-item[1], tuple(category.value for category in item[0])))
    return tuple(ConflictPattern(key, count, bool(key)) for key, count in ordered)


def _counts(counter):
    return MappingProxyType(dict(sorted(counter.items(), key=lambda item: item[0].value)))


def _aggregate(summaries):
    present, required, unknown, patterns, irreducible = (Counter() for _ in range(5))
    for summary in summaries:
        diagnostic = summary.diagnostics
        if diagnostic is None:
            continue
        present.update(_category_key(diagnostic.categories_present))
        required.update(_category_key(diagnostic.categories_required))
        unknown.update(_category_key(item.category for item in diagnostic.category_diagnostics
                                     if item.required_for_unsat is None))
        patterns[_category_key(diagnostic.categories_required)] += 1
        if diagnostic.minimize_requested and diagnostic.irreducible:
            irreducible[_category_key(diagnostic.conflict_categories)] += 1
    return (_counts(present), _counts(required), _counts(unknown),
            _patterns(patterns), _patterns(irreducible))


def _eligible(result):
    if result.trace is None:
        raise ValueError("search diagnostics require a recorded SearchTrace")
    result.trace.validate(result)
    return tuple(event for event in result.trace.events
                 if event.kind is TraceEventKind.SOLVED and event.solver_status is SolveStatus.UNSAT)


@dataclass(frozen=True)
class SearchDiagnostics:
    search_status: SearchStatus
    candidate_count: int
    max_depth: int
    max_diagnosed_candidates: int | None
    minimize_conflicts: bool
    eligible_candidates: int
    candidates_considered: int
    candidates_diagnosed: int
    unknown_candidates: int
    pruned_prefixes: int
    unknown_prefix_checks: int
    prune_reasons: tuple[tuple[str, int], ...]
    candidate_summaries: tuple[CandidateDiagnosticSummary, ...]
    category_counts: Mapping[ConstraintCategory, int]
    required_category_counts: Mapping[ConstraintCategory, int]
    unknown_category_counts: Mapping[ConstraintCategory, int]
    conflict_patterns: tuple[ConflictPattern, ...]
    irreducible_category_patterns: tuple[ConflictPattern, ...]
    diagnostic_solver_calls: int
    truncated: bool

    @property
    def conclusive(self) -> bool:
        """Original bounded-search conclusion, not a replay consistency claim."""
        return self.search_status is not SearchStatus.INCONCLUSIVE

    @property
    def unsat_candidates(self) -> int:
        return self.eligible_candidates

    @property
    def coverage(self) -> tuple[int, int]:
        return self.candidates_diagnosed, self.eligible_candidates

    @property
    def replay_solver_calls(self) -> int:
        return self.candidates_considered

    @property
    def total_solver_calls(self) -> int:
        return self.replay_solver_calls + self.diagnostic_solver_calls

    @property
    def replay_sat_candidates(self) -> int:
        return sum(item.replay_status is SolveStatus.SAT for item in self.candidate_summaries)

    @property
    def replay_unknown_candidates(self) -> int:
        return sum(item.replay_status is SolveStatus.UNKNOWN for item in self.candidate_summaries)

    @property
    def replay_consistent(self) -> bool:
        """All attempted replays stayed UNSAT; does not certify input identity."""
        return self.replay_sat_candidates == self.replay_unknown_candidates == 0

    def to_dict(self) -> dict:
        return {
            "search_status": self.search_status.value, "conclusive": self.conclusive,
            "candidate_count": self.candidate_count, "max_depth": self.max_depth,
            "max_diagnosed_candidates": self.max_diagnosed_candidates, "minimize_conflicts": self.minimize_conflicts,
            "eligible_candidates": self.eligible_candidates, "unsat_candidates": self.unsat_candidates,
            "candidates_considered": self.candidates_considered, "candidates_diagnosed": self.candidates_diagnosed,
            "coverage": list(self.coverage), "unknown_candidates": self.unknown_candidates,
            "truncated": self.truncated, "pruned_prefixes": self.pruned_prefixes,
            "unknown_prefix_checks": self.unknown_prefix_checks,
            "prune_reasons": [list(item) for item in self.prune_reasons],
            "category_counts": {key.value: value for key, value in self.category_counts.items()},
            "required_category_counts": {key.value: value for key, value in self.required_category_counts.items()},
            "unknown_category_counts": {key.value: value for key, value in self.unknown_category_counts.items()},
            "conflict_patterns": [item.to_dict() for item in self.conflict_patterns],
            "irreducible_category_patterns": [item.to_dict() for item in self.irreducible_category_patterns],
            "diagnostic_solver_calls": self.diagnostic_solver_calls, "replay_solver_calls": self.replay_solver_calls,
            "total_solver_calls": self.total_solver_calls, "replay_consistent": self.replay_consistent,
            "replay_sat_candidates": self.replay_sat_candidates, "replay_unknown_candidates": self.replay_unknown_candidates,
            "candidate_summaries": [item.to_dict() for item in self.candidate_summaries],
        }

    def validate(self, result: SearchResult | None = None) -> bool:
        """Structural checks only; optionally link summaries to the source trace."""
        def require(condition, message):
            if not condition:
                raise ValueError(f"invalid search diagnostics: {message}")
        require(isinstance(self.search_status, SearchStatus), "search status")
        for name in ("candidate_count", "max_depth", "eligible_candidates", "candidates_considered",
                     "candidates_diagnosed", "unknown_candidates", "pruned_prefixes", "unknown_prefix_checks",
                     "diagnostic_solver_calls"):
            value = getattr(self, name)
            require(type(value) is int and value >= 0, name)
        limit = self.max_diagnosed_candidates
        require(limit is None or (type(limit) is int and limit >= 0), "budget")
        require(type(self.minimize_conflicts) is bool and type(self.truncated) is bool, "options")
        if self.search_status is SearchStatus.NOT_FOUND:
            require(self.unknown_candidates == 0, "NOT_FOUND with unknown candidates")
        elif self.search_status is SearchStatus.INCONCLUSIVE:
            require(self.unknown_candidates > 0, "INCONCLUSIVE without unknown candidates")
        expected_attempts = 0 if self.search_status is SearchStatus.FOUND else (
            self.eligible_candidates if limit is None else min(self.eligible_candidates, limit))
        require(self.candidates_considered == expected_attempts == len(self.candidate_summaries), "attempt count/budget")
        require(0 <= self.candidates_diagnosed <= self.candidates_considered <= self.eligible_candidates, "coverage")
        require(self.truncated is (self.search_status is not SearchStatus.FOUND and
                                   self.candidates_considered < self.eligible_candidates), "truncated")
        diagnosed = calls = 0
        last_event_id = -1
        seen = set()
        for item in self.candidate_summaries:
            require(type(item.solved_event_id) is int and item.solved_event_id > last_event_id, "event order")
            last_event_id = item.solved_event_id
            require(type(item.depth) is int and 1 <= item.depth <= self.max_depth, "depth")
            require(len(item.sequence_indices) == item.depth and item.sequence_indices not in seen, "identity")
            seen.add(item.sequence_indices)
            require(all(type(index) is int and 0 <= index < self.candidate_count for index in item.sequence_indices), "index")
            require(type(item.cost) in (int, float) and item.cost >= 0 and
                    (type(item.cost) is int or isfinite(item.cost)), "cost")
            require(isinstance(item.replay_status, SolveStatus), "replay status")
            diagnostic = item.diagnostics
            if item.replay_status is SolveStatus.UNSAT:
                require(diagnostic is not None and diagnostic.status is SolveStatus.UNSAT, "missing UNSAT diagnostics")
                require(diagnostic.minimize_requested is self.minimize_conflicts, "minimize option")
                require(diagnostic.diagnostic_solver_calls == len(diagnostic.category_diagnostics) +
                        len(diagnostic.reduction_checks), "candidate diagnostic calls")
                require(not diagnostic.irreducible or self.minimize_conflicts, "unrequested irreducibility")
                diagnosed += 1
                calls += diagnostic.diagnostic_solver_calls
            else:
                require(diagnostic is None, "non-UNSAT diagnostic")
        require(diagnosed == self.candidates_diagnosed, "diagnosed count")
        require(calls == self.diagnostic_solver_calls, "diagnostic calls")
        aggregates = _aggregate(self.candidate_summaries)
        require(aggregates == (self.category_counts, self.required_category_counts, self.unknown_category_counts,
                               self.conflict_patterns, self.irreducible_category_patterns), "aggregate counts/patterns")
        for counts in (self.category_counts, self.required_category_counts, self.unknown_category_counts):
            require(all(isinstance(key, ConstraintCategory) and type(count) is int and 0 < count <= diagnosed
                        for key, count in counts.items()), "category count")
        if result is not None:
            events = _eligible(result)
            require(self.search_status is result.status and len(events) == self.eligible_candidates, "source search")
            require((self.candidate_count, self.max_depth) == (result.trace.candidate_count, result.trace.max_depth), "source bounds")
            require((self.unknown_candidates, self.pruned_prefixes, self.unknown_prefix_checks, self.prune_reasons) ==
                    (result.unknown_candidates, result.pruned_candidates, result.unknown_prefix_checks, result.prune_reasons), "source statistics")
            for item, event in zip(self.candidate_summaries, events):
                require((item.solved_event_id, item.sequence_indices, item.depth, item.cost) ==
                        (event.event_id, event.sequence_indices, event.depth, event.cost), "source event")
        return True


def diagnose_search_result(result: SearchResult, *, candidates: Sequence[Gadget], goals: FinalGoals | None,
                            initial_state: CPUState | None, max_diagnosed_candidates: int | None = None,
                            minimize_conflicts: bool = False, check_entry: bool = False,
                            constraints: Iterable[Equality] = (), memory_name: str = "memory_0",
                            timeout_ms: int | None = 5000) -> SearchDiagnostics:
    """Diagnose the first eligible trace events, without rerunning search.

    Supply the original ordered candidates, goals, initial state and fixed-solver
    options. Count/index checks cannot prove complete input identity. The budget
    bounds replay attempts even when replay changes to SAT/UNKNOWN. Such attempts
    are retained but excluded from UNSAT aggregation. FOUND never triggers replay.
    """
    if not isinstance(result, SearchResult):
        raise TypeError("result must be a SearchResult")
    if max_diagnosed_candidates is not None and (type(max_diagnosed_candidates) is not int or max_diagnosed_candidates < 0):
        raise ValueError("max_diagnosed_candidates must be None or a nonnegative integer")
    if type(minimize_conflicts) is not bool:
        raise TypeError("minimize_conflicts must be a bool")
    events = _eligible(result)
    if not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a finite ordered sequence")
    pool = tuple(candidates)
    if len(pool) != result.trace.candidate_count:
        raise ValueError("candidate count does not match the recorded trace")
    if not all(isinstance(item, Gadget) for item in pool):
        raise TypeError("all candidates must be Gadget objects")
    selected = () if result.status is SearchStatus.FOUND else events[:max_diagnosed_candidates]
    summaries = []
    if selected:
        saved_goals = (None if goals is None else MappingProxyType(dict(goals))
                       if isinstance(goals, Mapping) else tuple(goals))
        saved_constraints = tuple(constraints)
        initial = None if initial_state is None else initial_state.copy()
        for event in selected:
            sequence = tuple(pool[index] for index in event.sequence_indices)
            replay = solve_sequence(sequence, goals=saved_goals,
                                    initial_state=None if initial is None else initial.copy(),
                                    check_entry=check_entry, constraints=saved_constraints,
                                    memory_name=memory_name, timeout_ms=timeout_ms,
                                    diagnose=True, minimize_conflict=minimize_conflicts)
            if replay.status is SolveStatus.UNSAT and replay.diagnostics is None:
                raise RuntimeError("UNSAT replay did not provide Phase 10 diagnostics")
            if replay.status not in (SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNKNOWN):
                raise RuntimeError(f"unexpected replay status: {replay.status}")
            if replay.status is not SolveStatus.UNSAT and replay.diagnostics is not None:
                raise RuntimeError("non-UNSAT replay unexpectedly provided diagnostics")
            summaries.append(CandidateDiagnosticSummary(event.event_id, event.sequence_indices, event.depth,
                                                          event.cost, replay.status, replay.diagnostics))
    summaries = tuple(summaries)
    present, required, unknown, patterns, irreducible = _aggregate(summaries)
    return SearchDiagnostics(
        search_status=result.status, candidate_count=len(pool), max_depth=result.trace.max_depth,
        max_diagnosed_candidates=max_diagnosed_candidates, minimize_conflicts=minimize_conflicts,
        eligible_candidates=len(events), candidates_considered=len(summaries),
        candidates_diagnosed=sum(item.diagnostics is not None for item in summaries),
        unknown_candidates=result.unknown_candidates, pruned_prefixes=result.pruned_candidates,
        unknown_prefix_checks=result.unknown_prefix_checks, prune_reasons=result.prune_reasons,
        candidate_summaries=summaries, category_counts=present, required_category_counts=required,
        unknown_category_counts=unknown, conflict_patterns=patterns, irreducible_category_patterns=irreducible,
        diagnostic_solver_calls=sum(item.diagnostic_solver_calls for item in summaries),
        truncated=result.status is not SearchStatus.FOUND and len(selected) < len(events),
    )
