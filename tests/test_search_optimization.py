from dataclasses import replace
from itertools import product

import pytest

import minirop.search as search_module
from minirop.constraints import Equality, RegisterGoal
from minirop.search import SearchStatus, search_sequences
from minirop.solver import SolveResult, SolveStatus, SolverContext, solve_sequence
from minirop.state import make_symbolic_initial_state
from minirop.symbolic import Symbol
from test_sequence import gadget


@pytest.fixture
def pieces():
    a = gadget("48 c7 c0 01 00 00 00 c3", 0x4000)
    b = gadget("48 c7 c3 02 00 00 00 c3", 0x4100)
    c = gadget("48 c7 c3 02 00 00 00 c3", 0x4200)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    initial.set_register("rbx", 0)
    return a, b, c, initial


def test_default_compatibility_and_statistics(pieces):
    a, b, _, initial = pieces
    result = search_sequences([a, b], {"rax": 1, "rbx": 2}, max_depth=2, initial_state=initial)
    explicit = search_sequences([a, b], {"rax": 1, "rbx": 2}, max_depth=2,
                                initial_state=initial, pruning=False, cost_fn=None)
    assert result.status is explicit.status is SearchStatus.FOUND
    assert result.sequence == explicit.sequence == (a, b)
    assert result.depth == explicit.depth == result.cost == 2
    assert result.tested_candidates == result.generated_candidates == result.solver_calls == 4
    assert result.pruned_candidates == result.prefix_checks == result.unknown_solver_calls == 0
    assert result.lowest_cost_at_depth_proven is True


@pytest.mark.parametrize("pruning", [False, True])
def test_cost_prefers_ab_over_ac_despite_input_order(pieces, pruning):
    a, b, c, initial = pieces
    costs = {a: 1, b: 2, c: 10}
    result = search_sequences([a, c, b], {"rax": 1, "rbx": 2}, max_depth=2,
                              initial_state=initial, pruning=pruning, cost_fn=costs.__getitem__)
    assert result.sequence == (a, b) and result.cost == 3
    assert result.depth == 2 and result.tested_candidates == 5
    assert result.generated_candidates == 12
    assert result.shortest_depth_proven is result.lowest_cost_at_depth_proven is True


@pytest.mark.parametrize("pruning", [False, True])
def test_cost_ties_keep_original_index_order_after_sorted_frontier(pieces, pruning):
    a, b, _, initial = pieces
    costs = {a: 10, b: 1}
    result = search_sequences([a, b], {"rax": 1, "rbx": 2}, max_depth=2,
                              initial_state=initial, pruning=pruning, cost_fn=costs.__getitem__)
    # The depth-one frontier is B,A after cost sorting, but AB precedes BA.
    assert result.sequence == (a, b) and result.cost == 11
    assert result.tested_candidates == 4


def test_depth_beats_cost(pieces):
    a, b, _, initial = pieces
    both = gadget("48 c7 c0 01 00 00 00 48 c7 c3 02 00 00 00 c3", 0x4300)
    costs = {both: 100, a: 1, b: 1}
    result = search_sequences([both, a, b], {"rax": 1, "rbx": 2}, max_depth=3,
                              initial_state=initial, cost_fn=costs.__getitem__)
    assert result.sequence == (both,) and result.cost == 100 and result.depth == 1
    assert result.tested_candidates == result.generated_candidates == 3


@pytest.mark.parametrize("cost", [0, 0.5, 2, 10**400])
def test_repetition_zero_fractional_and_large_integer_costs(cost):
    a = gadget("48 83 c0 01 c3", 0x4000)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    result = search_sequences([a], {"rax": 3}, max_depth=3, initial_state=initial,
                              pruning=True, cost_fn=lambda _: cost)
    assert result.sequence == (a, a, a) and result.cost == 3 * cost
    assert result.tested_candidates == result.generated_candidates == 3
    assert result.prefix_checks == 2 and result.pruned_candidates == 0


@pytest.mark.parametrize("cost,exception", [
    (-1, ValueError), (float("nan"), ValueError), (float("inf"), ValueError),
    (-float("inf"), ValueError), ("1", TypeError), (True, TypeError), (None, TypeError),
])
def test_invalid_cost_rejected_before_any_solving(pieces, monkeypatch, cost, exception):
    a, b, _, _ = pieces
    def unexpected(*args, **kwargs):
        pytest.fail("costs must be validated before solving")
    monkeypatch.setattr(search_module, "solve_sequence", unexpected)
    with pytest.raises(exception, match="cost"):
        search_sequences([a, b], max_depth=2, cost_fn=lambda item: 0 if item is a else cost)


def test_nonfinite_sequence_total_rejected(pieces):
    a, _, _, initial = pieces
    with pytest.raises(ValueError, match="cost.*finite"):
        search_sequences([a], {"rax": 99}, max_depth=2, initial_state=initial, cost_fn=lambda _: 1e308)


def test_float_cost_sum_is_independent_of_permutation():
    costs = (1e16, 1.0, 1.0)
    assert search_module._sequence_cost((0, 1, 2), costs) == search_module._sequence_cost((1, 2, 0), costs)


@pytest.mark.parametrize("kwargs", [{"cost_fn": {}}, {"pruning": 1}])
def test_invalid_options(kwargs):
    with pytest.raises(TypeError):
        search_sequences([], max_depth=1, **kwargs)


@pytest.mark.parametrize("pruning", [False, True])
@pytest.mark.parametrize("costs,expected_cost_proven", [((1, 2), False), ((1, 1), True)])
def test_unknown_before_sat_at_same_depth(pieces, monkeypatch, pruning, costs, expected_cost_proven):
    a, b, _, _ = pieces
    baseline = solve_sequence([a])
    calls = []
    def simulated(sequence, **kwargs):
        calls.append(sequence)
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)) if sequence == (a,) else baseline
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences([a, b], max_depth=2, pruning=pruning,
                              cost_fn={a: costs[0], b: costs[1]}.__getitem__)
    assert calls == [(a,), (b,)]
    assert result.sequence == (b,) and result.unknown_candidates == 1
    assert result.shortest_depth_proven is True
    assert result.lowest_cost_at_depth_proven is expected_cost_proven
    assert result.prefix_checks == result.pruned_candidates == 0


def test_shallower_unknown_keeps_extensions_and_only_affects_shortest(pieces, monkeypatch):
    a, _, _, _ = pieces
    baseline = solve_sequence([a])
    calls = []
    def simulated(sequence, **kwargs):
        calls.append(sequence)
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN)) if len(sequence) == 1 else baseline
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences([a], max_depth=2, pruning=True, cost_fn=lambda _: 1)
    assert calls == [(a,), (a, a)]
    assert result.shortest_depth_proven is False
    assert result.lowest_cost_at_depth_proven is True
    assert result.unknown_candidates == 1 and result.pruned_candidates == result.prefix_checks == 0


def test_higher_cost_unknown_is_never_checked(pieces, monkeypatch):
    a, b, _, _ = pieces
    baseline = solve_sequence([a])
    def simulated(sequence, **kwargs):
        assert sequence == (a,)
        return baseline
    monkeypatch.setattr(search_module, "solve_sequence", simulated)
    result = search_sequences([b, a], max_depth=2, cost_fn={a: 1, b: 2}.__getitem__)
    assert result.sequence == (a,) and result.lowest_cost_at_depth_proven is True
    assert result.unknown_candidates == 0
    assert result.generated_candidates == 2 and result.tested_candidates == 1


def test_pruning_saves_total_solver_calls_with_exact_counts(monkeypatch):
    a, b = gadget("c3", 0x4000), gadget("c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    initial.set_register("rip", a.address)
    for index in range(4):
        initial.set_stack_slot(index, a.address)
    actual_calls = []
    original = SolverContext.solve
    def counted(self):
        actual_calls.append(self)
        return original(self)
    monkeypatch.setattr(SolverContext, "solve", counted)
    kwargs = dict(goals={"rax": 1}, max_depth=4, initial_state=initial, check_entry=True)
    plain = search_sequences([a, b], **kwargs)
    assert len(actual_calls) == plain.solver_calls == plain.tested_candidates == 30
    actual_calls.clear()
    pruned = search_sequences([a, b], pruning=True, **kwargs)
    assert plain.status is pruned.status is SearchStatus.NOT_FOUND
    assert len(actual_calls) == pruned.solver_calls == 14
    assert pruned.generated_candidates == pruned.tested_candidates == 8
    assert pruned.prefix_checks == 6 and pruned.pruned_candidates == 3
    assert pruned.prune_reasons == (("goal_free_prefix_unsat", 3),)
    assert pruned.unknown_solver_calls == 0
    assert pruned.cost is pruned.lowest_cost_at_depth_proven is None


def test_goal_free_unsat_reuses_proof_without_extra_call(pieces):
    a, b, _, _ = pieces
    result = search_sequences([a, b], max_depth=3, pruning=True, constraints=[Equality(1, 2)])
    assert result.status is SearchStatus.NOT_FOUND
    assert result.generated_candidates == result.tested_candidates == result.pruned_candidates == 2
    assert result.prefix_checks == 0 and result.solver_calls == 2


def test_goal_mismatch_is_repairable_and_does_not_prune(pieces):
    a, b, _, initial = pieces
    result = search_sequences([a, b], {"rax": 1, "rbx": 2}, max_depth=2,
                              initial_state=initial, pruning=True)
    assert result.sequence == (a, b)
    assert result.prefix_checks == 2 and result.pruned_candidates == 0


def test_prefix_unknown_does_not_prune_or_make_definite_exhaustion_inconclusive(pieces, monkeypatch):
    a, _, _, initial = pieces
    baseline = solve_sequence([a])
    def unknown(*args, **kwargs):
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN))
    monkeypatch.setattr(search_module, "solve_sequence_effect", unknown)
    result = search_sequences([a], {"rax": 99}, max_depth=3, initial_state=initial, pruning=True)
    assert result.status is SearchStatus.NOT_FOUND
    assert result.tested_candidates == result.generated_candidates == 3
    assert result.prefix_checks == result.unknown_prefix_checks == result.unknown_solver_calls == 2
    assert result.pruned_candidates == result.unknown_candidates == 0


@pytest.mark.parametrize("pruning", [False, True])
def test_all_candidate_unknown_remains_inconclusive(pieces, monkeypatch, pruning):
    a, b, _, _ = pieces
    baseline = solve_sequence([a])
    def unknown(*args, **kwargs):
        return replace(baseline, solution=SolveResult(SolveStatus.UNKNOWN))
    monkeypatch.setattr(search_module, "solve_sequence", unknown)
    result = search_sequences([a, b], max_depth=2, pruning=pruning, cost_fn=lambda _: 0)
    assert result.status is SearchStatus.INCONCLUSIVE
    assert result.generated_candidates == result.tested_candidates == result.unknown_candidates == 6
    assert result.prefix_checks == result.pruned_candidates == 0


@pytest.mark.parametrize("target", [0, 1, 2, 3])
@pytest.mark.parametrize("entry", [False, True])
@pytest.mark.parametrize("use_cost", [False, True])
def test_small_spaces_match_exhaustive_oracle(target, entry, use_cost):
    a = gadget("48 83 c0 01 c3", 0x4000)
    b = gadget("48 c7 c0 00 00 00 00 c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0)
    initial.set_register("rip", a.address)
    initial.set_stack_slot(0, a.address)
    costs = {a: 3, b: 0}
    kwargs = dict(goals={"rax": target}, max_depth=3, initial_state=initial,
                  check_entry=entry, cost_fn=costs.__getitem__ if use_cost else None)
    plain = search_sequences([a, b], **kwargs)
    pruned = search_sequences([a, b], pruning=True, **kwargs)
    assert plain.status is pruned.status
    assert plain.sequence == pruned.sequence and plain.depth == pruned.depth
    assert plain.cost == pruned.cost
    assert plain.shortest_depth_proven is pruned.shortest_depth_proven
    assert plain.lowest_cost_at_depth_proven is pruned.lowest_cost_at_depth_proven
    if pruned.status is SearchStatus.FOUND:
        direct = solve_sequence(pruned.sequence, kwargs["goals"], initial_state=initial, check_entry=entry)
        assert direct.status is SolveStatus.SAT and direct.goals_satisfied is True


def test_sorted_full_space_order_is_depth_cost_then_input_indices(pieces, monkeypatch):
    a, b, c, initial = pieces
    pool = [a, b, c]
    costs = {a: 3, b: 0, c: 1.5}
    calls = []
    original = search_module.solve_sequence
    def traced(sequence, **kwargs):
        calls.append(sequence)
        return original(sequence, **kwargs)
    monkeypatch.setattr(search_module, "solve_sequence", traced)
    result = search_sequences(pool, {"rax": 99}, max_depth=3, initial_state=initial,
                              pruning=True, cost_fn=costs.__getitem__)
    expected = []
    for depth in range(1, 4):
        indices = sorted(product(range(3), repeat=depth), key=lambda key: (sum(costs[pool[i]] for i in key), key))
        expected.extend(tuple(pool[i] for i in key) for key in indices)
    assert calls == expected
    assert result.tested_candidates == result.generated_candidates == 39
    assert result.pruned_candidates == 0 and result.prefix_checks == 12


def test_all_inputs_preserved_cost_evaluated_once_and_duplicate_goals_retained(pieces):
    a, b, _, initial = pieces
    pool = [a, a, b]
    goals = [RegisterGoal("rax", 1), RegisterGoal("rbx", 2), RegisterGoal("RAX", 1)]
    costs = {a: 2, b: 1}
    before = (pool.copy(), goals.copy(), costs.copy(), initial.register_bindings(), initial.memory_bindings())
    calls = []
    def cost_fn(item):
        calls.append(item)
        return costs[item]
    result = search_sequences(pool, iter(goals), max_depth=2, initial_state=initial,
                              pruning=True, cost_fn=cost_fn)
    assert calls == pool
    assert before == (pool, goals, costs, initial.register_bindings(), initial.memory_bindings())
    assert result.solve_result.final_goals == tuple(goals)
    assert result.sequence == (a, b) and result.cost == 3


def test_prefix_proof_receives_only_persistent_constraints_and_options(pieces, monkeypatch):
    a, _, _, initial = pieces
    extra = Equality(Symbol("x"), 3)
    captured = []
    original = search_module.solve_sequence_effect
    def traced(effect, **kwargs):
        captured.append(kwargs)
        assert kwargs == {"constraints": (extra,), "memory_name": "custom_memory", "timeout_ms": 2000}
        return original(effect, **kwargs)
    monkeypatch.setattr(search_module, "solve_sequence_effect", traced)
    result = search_sequences([a], [RegisterGoal("rax", 2)], max_depth=2, initial_state=initial,
                              pruning=True, constraints=iter([extra]), memory_name="custom_memory", timeout_ms=2000)
    assert len(captured) == result.prefix_checks == 1
    assert result.tested_candidates == 2 and result.pruned_candidates == 0


@pytest.mark.parametrize("empty,depth", [(True, 3), (False, 0)])
def test_empty_search_has_zero_statistics_and_does_not_call_cost(pieces, empty, depth):
    a, _, _, _ = pieces
    def unexpected(_):
        pytest.fail("empty search must not evaluate costs")
    result = search_sequences([] if empty else [a], max_depth=depth, pruning=True, cost_fn=unexpected)
    assert result.status is SearchStatus.NOT_FOUND
    assert result.generated_candidates == result.tested_candidates == result.pruned_candidates == 0
    assert result.solver_calls == result.unknown_solver_calls == 0
    assert result.cost is result.lowest_cost_at_depth_proven is None
