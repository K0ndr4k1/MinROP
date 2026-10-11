import json
from dataclasses import FrozenInstanceError, fields, replace

import pytest

import minirop.search as search_module
from minirop.constraints import Equality, RegisterGoal
from minirop.search import (SearchResult, SearchStatus, search_sequences,
                           verify_search_result)
from minirop.search_trace import PruneReason, SearchTrace, TraceEventKind as Kind
from minirop.solver import SolveResult, SolveStatus, SolverContext, solve_sequence
from minirop.state import make_symbolic_initial_state
from test_sequence import gadget


@pytest.fixture
def setup():
    a = gadget("48 c7 c0 01 00 00 00 c3", 0x4000)
    b = gadget("48 c7 c3 02 00 00 00 c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    initial.set_register("rbx", 0)
    return a, b, initial


def comparable(result):
    return {field.name: getattr(result, field.name) for field in fields(result)
            if field.name not in ("solve_result", "trace")}


def assert_counts(result):
    trace = result.trace
    assert isinstance(trace, SearchTrace)
    assert trace.validate(result) is True
    for kind, field in [(Kind.GENERATED, "generated_candidates"), (Kind.SOLVED, "tested_candidates"),
                        (Kind.PREFIX_CHECK, "prefix_checks"), (Kind.PRUNED, "pruned_candidates")]:
        assert sum(event.kind is kind for event in trace.events) == getattr(result, field)
    assert sum(e.kind is Kind.SOLVED and e.solver_status is SolveStatus.UNKNOWN for e in trace.events) == result.unknown_candidates
    assert sum(e.kind in (Kind.SOLVED, Kind.PREFIX_CHECK) and e.solver_status is SolveStatus.UNKNOWN
               for e in trace.events) == result.unknown_solver_calls
    assert trace.events[-1].kind is Kind.SEARCH_END
    assert trace.events[-1].summary == trace.summary


@pytest.mark.parametrize("pruning", [False, True])
@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("goal", [1, 99])
def test_trace_observes_without_changing_found_or_not_found(setup, pruning, cost, goal):
    a, b, initial = setup
    kwargs = dict(goals={"rax": goal, "rbx": 2}, max_depth=3, initial_state=initial,
                  pruning=pruning, cost_fn={a: 3, b: 1}.__getitem__ if cost else None)
    plain = search_sequences([a, b], **kwargs)
    recorded = search_sequences([a, b], record_trace=True, **kwargs)
    assert plain.trace is None
    assert comparable(plain) == comparable(recorded)
    assert_counts(recorded)
    if goal == 1:
        assert recorded.solve_result.constraints == plain.solve_result.constraints


def test_disabled_trace_allocates_no_event_objects(setup, monkeypatch):
    a, _, _ = setup
    def unexpected(*args, **kwargs):
        pytest.fail("trace disabled")
    monkeypatch.setattr(search_module, "SearchTraceEvent", unexpected)
    assert search_sequences([a], max_depth=1).trace is None


def test_exact_lazy_event_order_and_stable_indices(setup):
    a, b, initial = setup
    result = search_sequences([a, b], {"rax": 1, "rbx": 2}, max_depth=3,
                              initial_state=initial, record_trace=True)
    events = result.trace.events
    assert [e.kind for e in events] == [Kind.GENERATED, Kind.SOLVED] * 4 + [Kind.FOUND, Kind.SEARCH_END]
    assert [e.sequence_indices for e in events if e.kind is Kind.GENERATED] == [(0,), (1,), (0, 0), (0, 1)]
    assert [e.event_id for e in events] == list(range(10))
    assert_counts(result)


@pytest.mark.parametrize("pruning", [False, True])
def test_deterministic_json_trace_and_recorded_costs(setup, pruning):
    a, b, initial = setup
    costs = {a: 10, b: 0.5}
    kwargs = dict(goals={"rax": 1, "rbx": 2}, max_depth=2, initial_state=initial,
                  pruning=pruning, cost_fn=costs.__getitem__, record_trace=True)
    r1 = search_sequences([a, b], **kwargs)
    r2 = search_sequences([a, b], **kwargs)
    assert r1.trace.to_dict() == r2.trace.to_dict()
    data = r1.trace.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    for event in r1.trace.events[:-1]:
        assert event.depth == len(event.sequence_indices) <= 2
        assert event.cost == sum(costs[[a, b][i]] for i in event.sequence_indices)
    found = r1.trace.events[-2]
    assert found.kind is Kind.FOUND and found.cost == r1.cost
    assert found.sequence_indices == (0, 1)
    # Cost sorting generates all of depth 2 before solving its first entry.
    generated_at_two = [e.event_id for e in r1.trace.events if e.kind is Kind.GENERATED and e.depth == 2]
    first_solved_at_two = next(e.event_id for e in r1.trace.events if e.kind is Kind.SOLVED and e.depth == 2)
    assert max(generated_at_two) < first_solved_at_two
    assert_counts(r1)


def test_repeated_indices_and_duplicate_input_entries(setup):
    a, b, initial = setup
    result = search_sequences([a, b, a], {"rax": 99}, max_depth=3, initial_state=initial, record_trace=True)
    keys = [e.sequence_indices for e in result.trace.events if e.kind is Kind.GENERATED]
    assert (0, 1, 0) in keys and (0, 0) in keys and (2, 2) in keys
    assert len(keys) == 39
    assert_counts(result)


@pytest.mark.parametrize("goals", [None, {"rax": 99}])
def test_pruned_proof_links_and_counts_with_or_without_final_goals(setup, goals):
    a, b, initial = setup
    result = search_sequences([a, b], goals, max_depth=3, initial_state=initial,
                              constraints=[Equality(1, 2)], pruning=True, record_trace=True)
    assert result.generated_candidates == result.tested_candidates == result.pruned_candidates == 2
    assert result.prefix_checks == (0 if goals is None else 2)
    for event in result.trace.events:
        if event.kind is Kind.PRUNED:
            proof = result.trace.events[event.proof_event_id]
            assert proof.kind is (Kind.SOLVED if goals is None else Kind.PREFIX_CHECK)
            assert proof.goals_removed is True
            assert proof.solver_status is SolveStatus.UNSAT
            assert proof.sequence_indices == event.sequence_indices
            assert event.reason is PruneReason.PERSISTENT_CONSTRAINTS_UNSAT
    assert_counts(result)


def test_control_flow_pruning_and_statistics_unchanged_with_trace():
    a, b = gadget("c3", 0x4000), gadget("c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    initial.set_register("rip", a.address)
    for index in range(4):
        initial.set_stack_slot(index, a.address)
    kwargs = dict(goals={"rax": 1}, max_depth=4, initial_state=initial, pruning=True, check_entry=True)
    plain = search_sequences([a, b], **kwargs)
    traced = search_sequences([a, b], record_trace=True, **kwargs)
    assert comparable(plain) == comparable(traced)
    assert traced.solver_calls == 14 and traced.pruned_candidates == 3
    assert_counts(traced)


@pytest.mark.parametrize("pruning", [False, True])
def test_inconclusive_trace_and_unknown_counts(setup, monkeypatch, pruning):
    a, b, _ = setup
    c = gadget("c3", 0x4200)
    baseline = solve_sequence([a])
    def simulated(sequence, **kwargs):
        status = SolveStatus.UNKNOWN if sequence == (b,) else SolveStatus.UNSAT
        return replace(baseline, solution=SolveResult(status))
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    plain = search_sequences([a, b, c], max_depth=1, pruning=pruning)
    recorded = search_sequences([a, b, c], max_depth=1, pruning=pruning, record_trace=True)
    assert comparable(plain) == comparable(recorded)
    assert recorded.status is SearchStatus.INCONCLUSIVE
    solves = [e for e in recorded.trace.events if e.kind is Kind.SOLVED]
    assert [e.solver_status for e in solves] == [SolveStatus.UNSAT, SolveStatus.UNKNOWN, SolveStatus.UNSAT]
    assert solves[1].goals_satisfied is None
    assert_counts(recorded)


def test_both_kinds_of_unknown_recorded_separately(setup, monkeypatch):
    a, b, initial = setup
    baseline = solve_sequence([a])
    original = solve_sequence
    def simulated(sequence, **kwargs):
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)) if sequence == (b,) else original(sequence, **kwargs)
    def unknown_proof(*args, **kwargs):
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN))
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    monkeypatch.setattr(search_module, "solve_sequence_effect", unknown_proof)
    result = search_sequences([a, b], {"rax": 99}, max_depth=2, initial_state=initial,
                              pruning=True, record_trace=True)
    assert result.status is SearchStatus.INCONCLUSIVE
    assert result.unknown_candidates == result.unknown_prefix_checks == 1
    assert result.unknown_solver_calls == 2 and result.pruned_candidates == 0
    assert_counts(result)


@pytest.mark.parametrize("shallower", [False, True])
def test_unknown_explains_optimality_flags(setup, monkeypatch, shallower):
    a, b, _ = setup
    baseline = solve_sequence([a])
    def simulated(sequence, **kwargs):
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)) if sequence == (a,) else baseline
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences([a] if shallower else [a, b], max_depth=2, pruning=True,
                              cost_fn={a: 1, b: 2}.__getitem__, record_trace=True)
    found = result.trace.events[-2]
    unknown = next(e for e in result.trace.events if e.kind is Kind.SOLVED and e.solver_status is SolveStatus.UNKNOWN)
    assert found.shortest_depth_proven is (not shallower)
    assert found.lowest_cost_at_depth_proven is shallower
    assert unknown.depth < found.depth if shallower else unknown.cost < found.cost
    assert_counts(result)


@pytest.mark.parametrize("empty,depth", [(True, 3), (False, 0), (True, 0)])
def test_empty_trace_has_only_search_end(setup, empty, depth):
    a, _, _ = setup
    result = search_sequences([] if empty else [a], max_depth=depth, record_trace=True)
    assert len(result.trace.events) == 1 and result.trace.events[0].kind is Kind.SEARCH_END
    assert result.trace.summary.final_status is SearchStatus.NOT_FOUND
    assert_counts(result)


def test_trace_immutable_detached_from_inputs_and_later_search(setup, tmp_path, monkeypatch):
    a, b, initial = setup
    pool = [a, b]
    goals = [RegisterGoal("rax", 1), RegisterGoal("rbx", 2), RegisterGoal("rax", 1)]
    costs = {a: 1, b: 2}
    before = (pool.copy(), goals.copy(), costs.copy(), initial.register_bindings(), initial.memory_bindings())
    monkeypatch.chdir(tmp_path)
    result = search_sequences(pool, goals, max_depth=2, initial_state=initial,
                              cost_fn=costs.__getitem__, pruning=True, record_trace=True)
    saved = result.trace.to_dict()
    assert before == (pool, goals, costs, initial.register_bindings(), initial.memory_bindings())
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FrozenInstanceError):
        result.trace.events[0].cost = 999
    with pytest.raises(FrozenInstanceError):
        result.trace.summary.depth = 99
    exported = result.trace.to_dict()
    exported["events"][0]["sequence_indices"].append(99)
    initial.set_register("rax", 99)
    pool.reverse()
    goals.clear()
    costs[a] = 10
    search_sequences(pool, goals, max_depth=1, initial_state=initial, record_trace=True)
    assert result.trace.to_dict() == saved


def test_verification_really_calls_fixed_solver_and_preserves_inputs(setup, monkeypatch):
    a, b, initial = setup
    goals = [RegisterGoal("rax", 1), RegisterGoal("rbx", 2)]
    result = search_sequences([a, b], goals, max_depth=2, initial_state=initial, record_trace=True)
    before = (initial.register_bindings(), initial.memory_bindings(), goals.copy(), result.trace.to_dict())
    calls = []
    original = SolverContext.solve
    def counted(self):
        calls.append(self)
        return original(self)
    monkeypatch.setattr(SolverContext, "solve", counted)
    verification = verify_search_result(result, goals, initial_state=initial)
    assert len(calls) == 1
    assert verification.verified is True and verification.replay_status is SolveStatus.SAT
    assert verification.goals_satisfied is True and verification.original_status is SearchStatus.FOUND
    assert verification.replay_result is not result.solve_result
    assert verification.replay_result.constraints == result.solve_result.constraints
    assert before == (initial.register_bindings(), initial.memory_bindings(), goals, result.trace.to_dict())


@pytest.mark.parametrize("status", [SolveStatus.UNSAT, SolveStatus.UNKNOWN])
def test_failed_verification_retains_actual_replay_result(setup, monkeypatch, status):
    a, _, _ = setup
    result = search_sequences([a], max_depth=1)
    replay = replace(result.solve_result, solution=SolveResult(status, reason_unknown="simulated"))
    calls = []
    def simulated(sequence, **kwargs):
        calls.append(sequence)
        return replay
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    verification = verify_search_result(result, None, initial_state=None)
    assert calls == [result.sequence]
    assert verification.verified is False
    assert verification.replay_status is status and verification.replay_result is replay
    assert verification.goals_satisfied is (False if status is SolveStatus.UNSAT else None)


def test_changed_replay_conditions_are_actually_solved(setup):
    a, _, _ = setup
    result = search_sequences([a], {"rax": 1}, max_depth=1)
    verification = verify_search_result(result, {"rax": 2}, initial_state=None)
    assert verification.verified is False and verification.replay_status is SolveStatus.UNSAT


def test_verification_forwards_all_fixed_solver_options(setup, monkeypatch):
    a, _, initial = setup
    result = search_sequences([a], max_depth=1)
    extra = [Equality(1, 1)]
    calls = []
    def traced(sequence, **kwargs):
        calls.append(kwargs)
        return solve_sequence(sequence, **kwargs)
    monkeypatch.setattr(search_module, "solve_sequence", traced)
    verified = verify_search_result(result, {"rax": 1}, initial_state=initial, check_entry=True,
                                    constraints=extra, memory_name="custom", timeout_ms=2000)
    assert verified.verified is True
    assert calls == [dict(goals={"rax": 1}, initial_state=initial, check_entry=True,
                          constraints=extra, memory_name="custom", timeout_ms=2000)]


@pytest.mark.parametrize("status", [SearchStatus.NOT_FOUND, SearchStatus.INCONCLUSIVE])
def test_non_found_verification_does_not_solve(status, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("non-FOUND verification must not solve")
    monkeypatch.setattr(search_module, "solve_sequence", unexpected)
    verification = verify_search_result(SearchResult(status), None, initial_state=None)
    assert verification.verified is verification.replay_status is verification.replay_result is None
    assert verification.original_status is status


def test_validate_never_calls_solver_and_checks_result(setup, monkeypatch):
    a, _, _ = setup
    result = search_sequences([a], max_depth=1, record_trace=True)
    def unexpected(*args, **kwargs):
        pytest.fail("validation must not solve")
    monkeypatch.setattr(SolverContext, "solve", unexpected)
    assert result.trace.validate(result)
    with pytest.raises(ValueError, match="SearchResult"):
        result.trace.validate(replace(result, tested_candidates=99))


@pytest.mark.parametrize("damage", ["id", "depth", "index", "cost", "count", "end", "after_found", "flag", "status"])
def test_validate_rejects_damaged_trace(setup, damage):
    a, _, _ = setup
    result = search_sequences([a], max_depth=1, record_trace=True)
    trace = result.trace
    events = list(trace.events)
    if damage == "id":
        events[0] = replace(events[0], event_id=9)
    elif damage == "depth":
        events[0] = replace(events[0], depth=2)
    elif damage == "index":
        events[0] = replace(events[0], sequence_indices=(1,))
    elif damage == "cost":
        events[1] = replace(events[1], cost=99)
    elif damage == "count":
        summary = replace(trace.summary, tested_candidates=99)
        trace = replace(trace, summary=summary)
        events[-1] = replace(events[-1], summary=summary)
    elif damage == "end":
        events.pop()
    elif damage == "after_found":
        events.insert(-1, replace(events[0], event_id=3))
        events[-1] = replace(events[-1], event_id=4)
    elif damage == "flag":
        events[-2] = replace(events[-2], shortest_depth_proven=False)
    elif damage == "status":
        events[1] = replace(events[1], solver_status=SolveStatus.UNKNOWN)
    with pytest.raises(ValueError, match="invalid search trace"):
        replace(trace, events=tuple(events)).validate()


@pytest.mark.parametrize("damage", ["missing", "unknown", "goals", "identity", "reason"])
def test_validate_rejects_invalid_pruning_proof(setup, damage):
    a, _, _ = setup
    result = search_sequences([a], {"rax": 99}, max_depth=2, pruning=True,
                              constraints=[Equality(1, 2)], record_trace=True)
    events = list(result.trace.events)
    pruning_index = next(i for i, event in enumerate(events) if event.kind is Kind.PRUNED)
    proof_index = events[pruning_index].proof_event_id
    if damage == "missing":
        events[pruning_index] = replace(events[pruning_index], proof_event_id=None)
    elif damage == "unknown":
        events[proof_index] = replace(events[proof_index], solver_status=SolveStatus.UNKNOWN, goals_satisfied=None)
    elif damage == "goals":
        events[proof_index] = replace(events[proof_index], goals_removed=False)
    elif damage == "identity":
        events[proof_index] = replace(events[proof_index], sequence_indices=(0, 0), depth=2)
    elif damage == "reason":
        events[pruning_index] = replace(events[pruning_index], reason=None)
    with pytest.raises(ValueError):
        replace(result.trace, events=tuple(events)).validate()


def test_invalid_record_trace_option():
    with pytest.raises(TypeError, match="record_trace"):
        search_sequences([], max_depth=1, record_trace=1)
