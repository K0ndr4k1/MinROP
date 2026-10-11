"""Immutable diagnostic records; validation never invokes a solver."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING

from .solver import SolveStatus

if TYPE_CHECKING:
    from .search import SearchResult, SearchStatus


class TraceEventKind(str, Enum):
    GENERATED = "generated"
    SOLVED = "solved"
    PREFIX_CHECK = "prefix_check"
    PRUNED = "pruned"
    FOUND = "found"
    SEARCH_END = "search_end"


class PruneReason(str, Enum):
    PERSISTENT_CONSTRAINTS_UNSAT = "persistent_constraints_unsat"


@dataclass(frozen=True)
class TraceSummary:
    final_status: SearchStatus
    generated_candidates: int
    tested_candidates: int
    prefix_checks: int
    pruned_candidates: int
    unknown_candidates: int
    unknown_prefix_checks: int
    solver_calls: int
    unknown_solver_calls: int
    shallowest_unknown_depth: int | None
    depth: int | None
    cost: int | float | None
    shortest_depth_proven: bool | None
    lowest_cost_at_depth_proven: bool | None
    prune_reasons: tuple[tuple[str, int], ...]

    @classmethod
    def from_result(cls, result: SearchResult) -> TraceSummary:
        return cls(**{field.name: getattr(result, "status" if field.name == "final_status" else field.name)
                      for field in fields(cls)})


@dataclass(frozen=True)
class SearchTraceEvent:
    event_id: int
    kind: TraceEventKind
    depth: int = 0
    sequence_indices: tuple[int, ...] = ()
    cost: int | float | None = None
    solver_status: SolveStatus | None = None
    goals_satisfied: bool | None = None
    # True on PREFIX_CHECK, or SOLVED when no final goals were present.
    goals_removed: bool = False
    reason: PruneReason | None = None
    proof_event_id: int | None = None
    shortest_depth_proven: bool | None = None
    lowest_cost_at_depth_proven: bool | None = None
    summary: TraceSummary | None = None


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class SearchTrace:
    candidate_count: int
    max_depth: int
    pruning: bool
    cost_ordering: bool
    events: tuple[SearchTraceEvent, ...]
    summary: TraceSummary

    def to_dict(self) -> dict:
        """Fresh JSON-compatible data, not a stable persistence/resume format."""
        return _plain(self)

    def validate(self, result: SearchResult | None = None) -> bool:
        """Check structure, proof links, counters and diagnostic flags.

        This is not a logical proof checker or a search-space completeness
        certificate. Optionally compare the summary with a SearchResult.
        Raise ValueError on inconsistent evidence; never call a solver.
        """
        def require(condition, message):
            if not condition:
                raise ValueError(f"invalid search trace: {message}")

        require(type(self.candidate_count) is int and self.candidate_count >= 0, "candidate count")
        require(type(self.max_depth) is int and self.max_depth >= 0, "max depth")
        require(type(self.pruning) is bool and type(self.cost_ordering) is bool, "configuration flags")
        require(bool(self.events), "missing SEARCH_END")
        generated, solved, prefixes, pruned = {}, {}, {}, {}
        found = None
        last_solve_order = None
        for index, event in enumerate(self.events):
            require(isinstance(event, SearchTraceEvent), "event type")
            require(type(event.event_id) is int and event.event_id == index, "event ids must be consecutive")
            require(isinstance(event.kind, TraceEventKind), "event kind")
            if index:
                previous = self.events[index - 1]
                if previous.kind is TraceEventKind.SOLVED and previous.solver_status is SolveStatus.SAT:
                    require(event.kind is TraceEventKind.FOUND and event.sequence_indices == previous.sequence_indices,
                            "SAT must be followed immediately by FOUND")
            if event.kind is TraceEventKind.SEARCH_END:
                require(index == len(self.events) - 1, "SEARCH_END must be last")
                require(event.depth == 0 and event.sequence_indices == () and event.cost is None, "end identity")
                require(event.summary == self.summary, "end summary mismatch")
                continue
            require(found is None, "candidate event after FOUND")
            require(event.summary is None, "summary belongs only to SEARCH_END")
            key = event.sequence_indices
            require(isinstance(key, tuple), "indices must be a tuple")
            require(type(event.depth) is int and 1 <= event.depth <= self.max_depth, "depth out of bounds")
            require(len(key) == event.depth, "depth/sequence length mismatch")
            require(all(type(i) is int and 0 <= i < self.candidate_count for i in key), "candidate index")
            require(type(event.cost) in (int, float) and event.cost >= 0 and
                    (type(event.cost) is int or isfinite(event.cost)), "invalid cost")
            if not self.cost_ordering:
                require(event.cost == event.depth, "default cost must equal depth")
            require(not any(key[:len(prefix)] == prefix for prefix in pruned), "event below closed prefix")
            if event.kind is TraceEventKind.GENERATED:
                require(key not in generated, "duplicate GENERATED identity")
                generated[key] = event
                require(event.solver_status is None, "GENERATED has solver status")
                continue
            require(key in generated and generated[key].cost == event.cost, "missing/mismatched GENERATED")
            if event.kind in (TraceEventKind.SOLVED, TraceEventKind.PREFIX_CHECK):
                require(isinstance(event.solver_status, SolveStatus), "solver status")
                expected = None if event.solver_status is SolveStatus.UNKNOWN else event.solver_status is SolveStatus.SAT
                require(event.goals_satisfied is expected, "goals_satisfied mismatch")
            if event.kind is TraceEventKind.SOLVED:
                require(key not in solved, "duplicate SOLVED")
                order = (event.depth, event.cost, key)
                require(last_solve_order is None or order > last_solve_order, "unstable solve order")
                last_solve_order = order
                solved[key] = event
            elif event.kind is TraceEventKind.PREFIX_CHECK:
                require(self.pruning and event.depth < self.max_depth, "unexpected prefix check")
                require(key in solved and solved[key].solver_status is SolveStatus.UNSAT, "prefix needs UNSAT candidate")
                require(key not in prefixes and not solved[key].goals_removed, "unnecessary/duplicate prefix check")
                require(event.goals_removed is True, "prefix retained final goals")
                prefixes[key] = event
            elif event.kind is TraceEventKind.PRUNED:
                require(self.pruning and event.depth < self.max_depth, "unexpected pruning")
                proof_id = event.proof_event_id
                require(type(proof_id) is int and 0 <= proof_id < index, "proof event id")
                proof = self.events[proof_id]
                require(proof.kind in (TraceEventKind.PREFIX_CHECK, TraceEventKind.SOLVED), "proof kind")
                require(proof.sequence_indices == key and proof.goals_removed is True, "proof identity/goals")
                require(proof.solver_status is event.solver_status is SolveStatus.UNSAT, "pruning requires UNSAT")
                require(event.reason is PruneReason.PERSISTENT_CONSTRAINTS_UNSAT, "pruning reason")
                pruned[key] = event
            elif event.kind is TraceEventKind.FOUND:
                require(key in solved and solved[key].solver_status is SolveStatus.SAT, "FOUND without SAT")
                require(event.solver_status is SolveStatus.SAT and event.goals_satisfied is True, "FOUND status")
                found = event

        require(self.events[-1].kind is TraceEventKind.SEARCH_END, "missing SEARCH_END")
        summary = self.summary
        unknown = [event for event in solved.values() if event.solver_status is SolveStatus.UNKNOWN]
        unknown_prefix = sum(event.solver_status is SolveStatus.UNKNOWN for event in prefixes.values())
        counts = dict(generated_candidates=len(generated), tested_candidates=len(solved),
                      prefix_checks=len(prefixes), pruned_candidates=len(pruned),
                      unknown_candidates=len(unknown), unknown_prefix_checks=unknown_prefix,
                      solver_calls=len(solved) + len(prefixes), unknown_solver_calls=len(unknown) + unknown_prefix)
        for name, count in counts.items():
            require(type(getattr(summary, name)) is int and getattr(summary, name) == count, f"{name} count mismatch")
        require(summary.shallowest_unknown_depth == min((e.depth for e in unknown), default=None), "unknown depth")
        require(summary.prune_reasons == ((("goal_free_prefix_unsat", len(pruned)),) if pruned else ()), "prune counts")
        expected_status = "found" if found else "inconclusive" if unknown else "not_found"
        require(summary.final_status == expected_status, "final status")
        require(sum(e.solver_status is SolveStatus.SAT for e in solved.values()) == int(found is not None), "unreported SAT")
        if found:
            shortest = not any(e.depth < found.depth for e in unknown)
            cheapest = not any(e.depth == found.depth and e.cost < found.cost for e in unknown)
            require(found.shortest_depth_proven is summary.shortest_depth_proven is shortest, "shortest flag")
            require(found.lowest_cost_at_depth_proven is summary.lowest_cost_at_depth_proven is cheapest, "lowest cost flag")
            require((summary.depth, summary.cost) == (found.depth, found.cost), "FOUND summary")
        else:
            require(all(value is None for value in (summary.depth, summary.cost, summary.shortest_depth_proven,
                                                    summary.lowest_cost_at_depth_proven)), "non-FOUND fields")
            require(generated.keys() == solved.keys(), "unexamined generated candidates at exhaustion")
        if result is not None:
            require(summary == TraceSummary.from_result(result), "SearchResult summary mismatch")
        return True
