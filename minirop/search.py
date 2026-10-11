"""Deterministic bounded enumeration over an explicitly supplied candidate pool.

Optional goal-free UNSAT prefix proofs and stable within-depth cost ordering.
Every candidate and prefix proof uses the existing fixed-sequence solver.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from math import fsum, isfinite
from types import MappingProxyType

from .constraints import Equality
from .gadget import Gadget
from .solver import (FinalGoals, SequenceSolveResult, SolveStatus, solve_sequence,
                     solve_sequence_effect)
from .state import CPUState
from .search_trace import (PruneReason, SearchTrace, SearchTraceEvent, TraceEventKind,
                           TraceSummary)


class SearchStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    sequence: tuple[Gadget, ...] = ()
    solve_result: SequenceSolveResult | None = None
    depth: int | None = None
    tested_candidates: int = 0
    unknown_candidates: int = 0
    shallowest_unknown_depth: int | None = None
    generated_candidates: int = 0
    pruned_candidates: int = 0
    prefix_checks: int = 0
    unknown_prefix_checks: int = 0
    cost: int | float | None = None
    lowest_cost_at_depth_proven: bool | None = None
    prune_reasons: tuple[tuple[str, int], ...] = ()
    trace: SearchTrace | None = None

    @property
    def solver_calls(self) -> int:
        """All actual solver invocations, including goal-free prefix proofs."""
        return self.tested_candidates + self.prefix_checks

    @property
    def unknown_solver_calls(self) -> int:
        return self.unknown_candidates + self.unknown_prefix_checks

    @property
    def shortest_depth_proven(self) -> bool | None:
        """UNKNOWN at a smaller depth prevents a shortest-depth claim."""
        if self.status is not SearchStatus.FOUND:
            return None
        return self.shallowest_unknown_depth is None or self.shallowest_unknown_depth >= self.depth


def _finite_cost(value: int | float) -> int | float:
    # Do not coerce strings, bools, or arbitrary numeric objects silently.
    if type(value) not in (int, float):
        raise TypeError("cost must be an int or float")
    if value < 0 or (type(value) is float and not isfinite(value)):
        raise ValueError("cost must be finite and nonnegative")
    return value


def _sequence_cost(indices: tuple[int, ...], costs: tuple[int | float, ...]) -> int | float:
    try:
        values = [costs[index] for index in indices]
        total = fsum(values) if any(type(value) is float for value in values) else sum(values)
        return _finite_cost(total)
    except OverflowError as error:
        raise ValueError("sequence cost must be finite") from error


def _prefix_proof(result: SequenceSolveResult, *, constraints: tuple[Equality, ...],
                  memory_name: str, timeout_ms: int | None) -> tuple[SolveStatus, int]:
    """Return goal-free feasibility and the number of extra solver calls.

    Input memory bindings, entry/flow equalities, and fixed extra constraints
    persist under extension. Their UNSAT is therefore monotone. Final register
    goals are NOT monotone and must be omitted from this separate proof.
    Never called for a candidate UNKNOWN. The effect is an immutable snapshot;
    no mutable solver or state is shared with another candidate.
    """
    if not result.final_goals:
        return result.status, 0
    proof = solve_sequence_effect(result.effect, constraints=constraints,
                                  memory_name=memory_name, timeout_ms=timeout_ms)
    return proof.status, 1


def _attach_trace(result: SearchResult, events: list[SearchTraceEvent], *,
                  candidate_count: int, max_depth: int, pruning: bool,
                  cost_ordering: bool, found_key: tuple[int, ...] = ()) -> SearchResult:
    summary = TraceSummary.from_result(result)
    if result.status is SearchStatus.FOUND:
        events.append(SearchTraceEvent(
            len(events), TraceEventKind.FOUND, result.depth, found_key, result.cost,
            solver_status=SolveStatus.SAT, goals_satisfied=True,
            shortest_depth_proven=result.shortest_depth_proven,
            lowest_cost_at_depth_proven=result.lowest_cost_at_depth_proven,
        ))
    events.append(SearchTraceEvent(len(events), TraceEventKind.SEARCH_END, summary=summary))
    trace = SearchTrace(candidate_count, max_depth, pruning, cost_ordering, tuple(events), summary)
    return replace(result, trace=trace)


def search_sequences(candidates: Sequence[Gadget], goals: FinalGoals | None = None, *,
                     max_depth: int, initial_state: CPUState | None = None,
                     check_entry: bool = False, constraints: Iterable[Equality] = (),
                     memory_name: str = "memory_0", timeout_ms: int | None = 5000,
                     pruning: bool = False,
                     cost_fn: Callable[[Gadget], int | float] | None = None,
                     record_trace: bool = False,
                     report_execution_errors: bool = False) -> SearchResult:
    """Enumerate depths 1..max_depth and stop at the first observed SAT result.

    candidates must be a finite ordered sequence (e.g. list/tuple), not a set or
    a potentially unbounded iterator. Cartesian products are generated lazily,
    allowing reuse of the same gadget. Defaults preserve Phase 7 order. With
    cost_fn, each depth is materialized and sorted by additive cost then input
    index tuple. The callback is evaluated once per input entry, not per trial.
    It must be pure/deterministic; no sequence/model/state cache is used.

    With pruning, only goal-free UNSAT prefixes stop generating descendants.
    generated_candidates counts formed sequences (including sorted but untested
    ones); pruned_candidates counts closed prefixes, not omitted descendants.
    tested_candidates and unknown_candidates describe full candidate calls;
    prefix_checks and unknown_prefix_checks count additional goal-free calls.

    record_trace saves immutable diagnostic events only when explicitly enabled.
    Events contain input indices and scalar results, never solver/state objects.

    timeout_ms is forwarded per solve, not a whole-search time budget. Analysis
    and input errors propagate: unsupported instructions or missing concrete
    memory must never be relabeled UNSAT/UNKNOWN by the search layer.
    report_execution_errors forwards the fixed solver's optional fault-reporting
    policy. Faulted candidates then remain inconclusive, preserving enumeration.
    """
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if type(pruning) is not bool:
        raise TypeError("pruning must be a bool")
    if type(record_trace) is not bool:
        raise TypeError("record_trace must be a bool")
    if type(report_execution_errors) is not bool:
        raise TypeError("report_execution_errors must be a bool")
    if cost_fn is not None and not callable(cost_fn):
        raise TypeError("cost_fn must be callable")
    if not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a finite ordered sequence such as a list or tuple")
    pool = tuple(candidates)
    if not all(isinstance(candidate, Gadget) for candidate in pool):
        raise TypeError("all candidates must be Gadget objects")
    if not pool or max_depth == 0:
        empty = SearchResult(SearchStatus.NOT_FOUND)
        return (_attach_trace(empty, [], candidate_count=len(pool), max_depth=max_depth,
                              pruning=pruning, cost_ordering=cost_fn is not None)
                if record_trace else empty)
    costs = tuple(_finite_cost(cost_fn(gadget)) for gadget in pool) if cost_fn is not None else (1,) * len(pool)

    # Freeze iterable inputs once, without changing goal meaning or losing
    # duplicate goals. Register/expression validation stays in Phase 6.
    if goals is None:
        saved_goals = None
    elif isinstance(goals, Mapping):
        saved_goals = MappingProxyType(dict(goals))
    else:
        saved_goals = tuple(goals)
    saved_constraints = tuple(constraints)
    initial = None if initial_state is None else initial_state.copy()
    tested = unknown = generated = pruned = prefix_checks = prefix_unknown = 0
    shallowest_unknown = None
    frontier = [()]
    events = [] if record_trace else None

    def emit(kind, key, total_cost, **kwargs):
        event_id = len(events)
        events.append(SearchTraceEvent(event_id, kind, len(key), key, total_cost, **kwargs))
        return event_id

    def finish(status, sequence=(), result=None, depth=None, total_cost=None, cost_proven=None, key=()):
        answer = SearchResult(
            status=status, sequence=sequence, solve_result=result, depth=depth,
            tested_candidates=tested, unknown_candidates=unknown,
            shallowest_unknown_depth=shallowest_unknown,
            generated_candidates=generated, pruned_candidates=pruned,
            prefix_checks=prefix_checks, unknown_prefix_checks=prefix_unknown,
            cost=total_cost, lowest_cost_at_depth_proven=cost_proven,
            prune_reasons=(("goal_free_prefix_unsat", pruned),) if pruned else (),
        )
        return (_attach_trace(answer, events, candidate_count=len(pool), max_depth=max_depth,
                              pruning=pruning, cost_ordering=cost_fn is not None, found_key=key)
                if events is not None else answer)

    for depth in range(1, max_depth + 1):
        indices = ((prefix + (index,) for prefix in frontier for index in range(len(pool)))
                   if pruning else product(range(len(pool)), repeat=depth))
        if cost_fn is not None:
            ordered = [(_sequence_cost(key, costs), key) for key in indices]
            generated += len(ordered)
            if events is not None:
                for total_cost, key in ordered:
                    emit(TraceEventKind.GENERATED, key, total_cost)
            ordered.sort(key=lambda item: (item[0], item[1]))
        else:
            ordered = ((depth, key) for key in indices)
        next_frontier = []
        lowest_unknown_cost = None
        for total_cost, key in ordered:
            if cost_fn is None:
                generated += 1
                if events is not None:
                    emit(TraceEventKind.GENERATED, key, total_cost)
            sequence = tuple(pool[index] for index in key)
            # Each candidate gets an isolated state and a fresh solver context
            # through solve_sequence. There is no model transfer between trials.
            result = solve_sequence(
                sequence, goals=saved_goals,
                initial_state=None if initial is None else initial.copy(),
                check_entry=check_entry, constraints=saved_constraints,
                memory_name=memory_name, timeout_ms=timeout_ms,
                **({"report_execution_errors": True} if report_execution_errors else {}),
            )
            tested += 1
            if events is not None:
                solved_id = emit(TraceEventKind.SOLVED, key, total_cost,
                                 solver_status=result.status, goals_satisfied=result.goals_satisfied,
                                 goals_removed=not result.final_goals)
            if result.status is SolveStatus.SAT:
                if result.goals_satisfied is not True:
                    raise RuntimeError("fixed-sequence solver returned SAT without satisfied goals")
                cost_proven = lowest_unknown_cost is None or lowest_unknown_cost >= total_cost
                return finish(SearchStatus.FOUND, sequence, result, depth, total_cost, cost_proven, key)
            if result.status is SolveStatus.UNKNOWN:
                unknown += 1
                if shallowest_unknown is None:
                    shallowest_unknown = depth
                if lowest_unknown_cost is None:
                    lowest_unknown_cost = total_cost
            elif result.status is not SolveStatus.UNSAT:
                raise RuntimeError(f"unexpected fixed-sequence solver status: {result.status}")
            if pruning and depth < max_depth:
                if result.status is SolveStatus.UNSAT:
                    proof_status, calls = _prefix_proof(
                        result, constraints=saved_constraints,
                        memory_name=memory_name, timeout_ms=timeout_ms,
                    )
                    prefix_checks += calls
                    if events is not None:
                        proof_id = solved_id
                        if calls:
                            proof_id = emit(TraceEventKind.PREFIX_CHECK, key, total_cost,
                                            solver_status=proof_status, goals_removed=True,
                                            goals_satisfied=None if proof_status is SolveStatus.UNKNOWN
                                            else proof_status is SolveStatus.SAT)
                    if proof_status is SolveStatus.UNSAT:
                        pruned += 1
                        if events is not None:
                            emit(TraceEventKind.PRUNED, key, total_cost, solver_status=proof_status,
                                 reason=PruneReason.PERSISTENT_CONSTRAINTS_UNSAT, proof_event_id=proof_id)
                        continue
                    if proof_status is SolveStatus.UNKNOWN:
                        prefix_unknown += calls
                    elif proof_status is not SolveStatus.SAT:
                        raise RuntimeError(f"unexpected prefix solver status: {proof_status}")
                next_frontier.append(key)
        if pruning:
            frontier = next_frontier
            if not frontier:
                break

    status = SearchStatus.INCONCLUSIVE if unknown else SearchStatus.NOT_FOUND
    return finish(status)


@dataclass(frozen=True)
class VerificationResult:
    verified: bool | None
    original_status: SearchStatus
    replay_status: SolveStatus | None
    goals_satisfied: bool | None
    replay_result: SequenceSolveResult | None = None


def verify_search_result(result: SearchResult, goals: FinalGoals | None, *,
                         initial_state: CPUState | None, check_entry: bool = False,
                         constraints: Iterable[Equality] = (), memory_name: str = "memory_0",
                         timeout_ms: int | None = 5000) -> VerificationResult:
    """Re-solve only the FOUND sequence, without search, pruning or cost logic.

    Caller must supply the same goals, initial state (explicit None for the
    default), and other fixed-solver options. Trace deliberately stores none of
    these mutable inputs. Changed inputs verify a different problem; this API
    does not certify input identity, shortestness or search completeness.
    Non-FOUND returns verified=None without calling the solver. Analysis errors
    propagate; SAT/UNSAT/UNKNOWN replay results are preserved in full.
    """
    if not isinstance(result, SearchResult):
        raise TypeError("result must be a SearchResult")
    if result.status is not SearchStatus.FOUND:
        return VerificationResult(None, result.status, None, None)
    replay = solve_sequence(result.sequence, goals=goals, initial_state=initial_state,
                            check_entry=check_entry, constraints=constraints,
                            memory_name=memory_name, timeout_ms=timeout_ms)
    verified = replay.status is SolveStatus.SAT and replay.goals_satisfied is True
    return VerificationResult(verified, result.status, replay.status, replay.goals_satisfied, replay)
