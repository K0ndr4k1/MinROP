from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest

import minirop.search as search_module
from minirop import GadgetScanner
from minirop.constraints import Equality, RegisterGoal
from minirop.search import SearchStatus, search_sequences
from minirop.semantics import UnsupportedInstruction
from minirop.solver import SolveResult, SolveStatus, solve_sequence
from minirop.state import MemoryAccessError, make_symbolic_initial_state
from minirop.symbolic import Symbol
from test_sequence import gadget


@pytest.fixture
def setup_search():
    a = gadget("48 c7 c0 01 00 00 00 c3", 0x4000)
    b = gadget("48 c7 c3 02 00 00 00 c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    initial.set_register("rbx", 0)
    return a, b, initial


def test_depth_one_success_and_full_result(setup_search):
    a, b, initial = setup_search
    result = search_sequences([a, b], {"RAX": 1}, max_depth=3, initial_state=initial)
    assert result.status is SearchStatus.FOUND
    assert result.sequence == (a,)
    assert result.depth == result.tested_candidates == 1
    assert result.unknown_candidates == 0
    assert result.shortest_depth_proven is True
    assert result.solve_result.status is SolveStatus.SAT
    assert result.solve_result.goals_satisfied is True
    assert result.solve_result.final_goals == (RegisterGoal("RAX", 1),)
    with pytest.raises(FrozenInstanceError):
        result.depth = 9


@pytest.mark.parametrize("reverse", [False, True])
def test_depth_two_and_stable_tie_order(setup_search, reverse):
    a, b, initial = setup_search
    pool = [b, a] if reverse else [a, b]
    goals = [RegisterGoal("rax", 1), RegisterGoal("rbx", 2)]
    shallow = search_sequences(pool, goals, max_depth=1, initial_state=initial)
    assert shallow.status is SearchStatus.NOT_FOUND
    assert shallow.tested_candidates == 2
    assert shallow.sequence == () and shallow.solve_result is None and shallow.depth is None
    assert shallow.shortest_depth_proven is None
    for _ in range(2):
        result = search_sequences(pool, goals, max_depth=2, initial_state=initial)
        assert result.status is SearchStatus.FOUND
        assert result.sequence == tuple(pool)
        assert result.depth == 2 and result.tested_candidates == 4
        assert result.shortest_depth_proven is True


def test_shorter_candidate_wins_over_earlier_longer_combination(setup_search):
    a, b, initial = setup_search
    c = gadget("48 c7 c0 01 00 00 00 48 c7 c3 02 00 00 00 c3", 0x4200)
    result = search_sequences([a, b, c], {"rax": 1, "rbx": 2}, max_depth=2, initial_state=initial)
    assert result.sequence == (c,) and result.depth == 1
    assert result.tested_candidates == 3


def test_unsat_does_not_end_search(setup_search):
    a, b, initial = setup_search
    result = search_sequences([a, b], {"rbx": 2}, max_depth=1, initial_state=initial)
    assert result.sequence == (b,) and result.tested_candidates == 2


def test_exhaustive_order_and_duplicate_goal_generator(setup_search, monkeypatch):
    a, b, initial = setup_search
    goals = (RegisterGoal("rax", n) for n in [1, 2])
    calls = []

    def traced(sequence, **kwargs):
        calls.append(sequence)
        result = solve_sequence(sequence, **kwargs)
        assert len(result.final_goals) == 2
        assert result.status is SolveStatus.UNSAT
        return result

    monkeypatch.setattr(search_module, "solve_sequence", traced)
    result = search_sequences([a, b], goals, max_depth=3, initial_state=initial)
    assert calls == [seq for depth in range(1, 4) for seq in product([a, b], repeat=depth)]
    assert result.status is SearchStatus.NOT_FOUND
    assert result.tested_candidates == 14 and result.unknown_candidates == 0


def test_same_gadget_can_be_used_twice():
    a = gadget("48 83 c0 01 c3", 0x4000)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    result = search_sequences([a], {"rax": 2}, max_depth=2, initial_state=initial)
    assert result.sequence == (a, a) and result.depth == 2
    assert result.tested_candidates == 2


def test_duplicate_pool_entries_are_not_deduplicated(setup_search):
    a, b, initial = setup_search
    result = search_sequences([a, a, b], {"rbx": 2}, max_depth=1, initial_state=initial)
    assert result.sequence == (b,) and result.tested_candidates == 3


@pytest.mark.parametrize("statuses,expected,unknown,shortest", [
    ([SolveStatus.UNKNOWN, SolveStatus.SAT], SearchStatus.FOUND, 1, True),
    ([SolveStatus.UNKNOWN, SolveStatus.UNSAT], SearchStatus.INCONCLUSIVE, 1, None),
    ([SolveStatus.UNSAT, SolveStatus.UNKNOWN], SearchStatus.INCONCLUSIVE, 1, None),
    ([SolveStatus.UNKNOWN, SolveStatus.UNKNOWN], SearchStatus.INCONCLUSIVE, 2, None),
    ([SolveStatus.UNSAT, SolveStatus.UNSAT], SearchStatus.NOT_FOUND, 0, None),
])
def test_three_value_aggregation(setup_search, monkeypatch, statuses, expected, unknown, shortest):
    a, b, _ = setup_search
    baseline = solve_sequence([a])
    outcomes = iter(statuses)
    returned = []

    def simulated(sequence, **kwargs):
        status = next(outcomes)
        result = replace(baseline, solution=SolveResult(status, reason_unknown="simulated" if status is SolveStatus.UNKNOWN else None))
        returned.append(result)
        return result

    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences([a, b], max_depth=1)
    assert result.status is expected
    assert result.tested_candidates == 2 and result.unknown_candidates == unknown
    assert result.shortest_depth_proven is shortest
    if expected is SearchStatus.FOUND:
        assert result.solve_result is returned[-1]
        assert result.sequence == (b,)
    else:
        assert result.solve_result is None and result.sequence == () and result.depth is None


def test_unknown_at_smaller_depth_prevents_shortest_claim(setup_search, monkeypatch):
    a, _, _ = setup_search
    baseline = solve_sequence([a])

    def simulated(sequence, **kwargs):
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)) if len(sequence) == 1 else baseline

    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences([a], max_depth=2)
    assert result.status is SearchStatus.FOUND and result.depth == 2
    assert result.unknown_candidates == 1 and result.shallowest_unknown_depth == 1
    assert result.shortest_depth_proven is False


@pytest.mark.parametrize("empty,depth", [(True, 3), (False, 0), (True, 0)])
def test_empty_search_never_calls_solver(setup_search, monkeypatch, empty, depth):
    a, _, _ = setup_search
    def unexpected(*args, **kwargs):
        pytest.fail("empty search called solver")
    monkeypatch.setattr(search_module, "solve_sequence", unexpected)
    result = search_sequences([] if empty else [a], max_depth=depth)
    assert result.status is SearchStatus.NOT_FOUND
    assert result.tested_candidates == result.unknown_candidates == 0


@pytest.mark.parametrize("depth", [-1, 1.5, True, "2", None])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="nonnegative integer"):
        search_sequences([], max_depth=depth)


@pytest.mark.parametrize("pool", [set(), iter([]), [object()], "abc"])
def test_invalid_candidates(pool):
    with pytest.raises(TypeError):
        search_sequences(pool, max_depth=1)


@pytest.mark.parametrize("goals", [None, {}, [], ()])
def test_empty_goals_preserve_fixed_api_behavior(setup_search, goals):
    a, _, _ = setup_search
    result = search_sequences([a], goals, max_depth=1)
    assert result.status is SearchStatus.FOUND
    assert result.solve_result.final_goals == ()
    assert result.solve_result.goals_satisfied is True


def test_inputs_unchanged_and_fixed_solver_parity(setup_search):
    a, b, initial = setup_search
    candidates = [a, b]
    goals = [RegisterGoal("rax", 1), RegisterGoal("rbx", 2), RegisterGoal("RAX", 1)]
    before = (initial.register_bindings(), initial.memory_bindings(), candidates.copy(), goals.copy())
    result = search_sequences(candidates, goals, max_depth=2, initial_state=initial)
    assert before == (initial.register_bindings(), initial.memory_bindings(), candidates, goals)
    assert result.solve_result.final_goals == tuple(goals)
    direct = solve_sequence(result.sequence, goals=goals, initial_state=initial)
    assert direct.status is result.solve_result.status is SolveStatus.SAT
    assert direct.goals_satisfied is result.solve_result.goals_satisfied is True
    assert direct.constraints == result.solve_result.constraints
    assert direct.effect.final_state == result.solve_result.effect.final_state


def test_candidate_state_copies_and_option_forwarding(setup_search, monkeypatch):
    a, b, initial = setup_search
    before = (initial.register_bindings(), initial.memory_bindings())
    seen = []
    extra = Equality(Symbol("custom"), 7)
    baseline = solve_sequence([a])
    def simulated(sequence, **kwargs):
        trial = kwargs["initial_state"]
        assert (trial.register_bindings(), trial.memory_bindings()) == before
        assert kwargs["constraints"] == (extra,)
        assert kwargs["check_entry"] is True
        assert kwargs["memory_name"] == "custom_memory" and kwargs["timeout_ms"] == 123
        trial.set_register("rax", 99)
        trial.set_stack_slot(0, 123)
        trial.write_memory(0x1000, 999)
        seen.append(trial)
        return replace(baseline, solution=SolveResult(SolveStatus.UNSAT))
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences([a, b], max_depth=2, initial_state=initial, check_entry=True,
                              constraints=iter([extra]), memory_name="custom_memory", timeout_ms=123)
    assert result.tested_candidates == 6
    assert len({id(state) for state in seen}) == 6
    assert (initial.register_bindings(), initial.memory_bindings()) == before


def test_control_flow_constraints_are_used(setup_search):
    a, b, initial = setup_search
    initial.set_stack_slot(0, a.address)
    # A->B violates the concrete first return; B->A is compatible.
    result = search_sequences([a, b], {"rax": 1, "rbx": 2}, max_depth=2, initial_state=initial)
    assert result.sequence == (b, a) and result.tested_candidates == 5


def test_entry_and_extra_constraints_are_used(setup_search):
    a, b, initial = setup_search
    initial.set_register("rip", b.address)
    result = search_sequences([a, b], max_depth=1, initial_state=initial, check_entry=True)
    assert result.sequence == (b,)
    conflict = search_sequences([a], max_depth=1, constraints=[Equality(1, 2)])
    assert conflict.status is SearchStatus.NOT_FOUND


def test_search_does_not_scan(setup_search, monkeypatch):
    a, _, _ = setup_search
    def unexpected(*args, **kwargs):
        pytest.fail("search must not scan")
    monkeypatch.setattr(GadgetScanner, "scan", unexpected)
    assert search_sequences([a], max_depth=1).status is SearchStatus.FOUND


def test_analysis_errors_propagate(setup_search):
    a, _, initial = setup_search
    initial.set_register("rsp", 0x1000)
    with pytest.raises(MemoryAccessError):
        search_sequences([a], max_depth=1, initial_state=initial)
    unsupported = gadget("0f a2 c3", 0x4200)
    with pytest.raises(UnsupportedInstruction):
        search_sequences([unsupported], max_depth=1)
