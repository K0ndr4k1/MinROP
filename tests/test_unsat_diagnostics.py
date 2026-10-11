import json
from dataclasses import FrozenInstanceError

import pytest

import minirop.diagnostics as diagnostics_module
from minirop.constraints import Equality, RegisterGoal
from minirop.diagnostics import ConstraintCategory as Category
from minirop.search import search_sequences, verify_search_result
from minirop.solver import SolveResult, SolveStatus, SolverContext, solve_sequence, solve_sequence_effect
from minirop.state import make_symbolic_initial_state
from minirop.symbolic import MemoryRead, Symbol
from test_sequence import gadget


@pytest.fixture
def simple():
    return [gadget("c3", 0x4000)], make_symbolic_initial_state(0), [RegisterGoal("RAX", 1), RegisterGoal("RAX", 2)]


def solve_simple(simple, **kwargs):
    sequence, initial, goals = simple
    return solve_sequence(sequence, goals, initial_state=initial, **kwargs)


def category_map(diagnostics):
    return {item.category: item for item in diagnostics.category_diagnostics}


@pytest.mark.parametrize("diagnose", [False, True])
@pytest.mark.parametrize("sat", [False, True])
def test_default_and_sat_paths_do_not_add_calls(simple, monkeypatch, diagnose, sat):
    sequence, initial, goals = simple
    if sat:
        goals = goals[:1]
    calls = []
    original = SolverContext.solve
    def counted(self):
        calls.append(self)
        return original(self)
    monkeypatch.setattr(SolverContext, "solve", counted)
    result = solve_sequence(sequence, goals, initial_state=initial, diagnose=diagnose)
    assert result.status is (SolveStatus.SAT if sat else SolveStatus.UNSAT)
    if not diagnose or sat:
        assert len(calls) == 1
        assert result.diagnostics is None and result.diagnostic_solver_calls == 0
    else:
        assert len(calls) == 2
        assert result.diagnostic_solver_calls == 1


@pytest.mark.parametrize("minimize", [False, True])
def test_unknown_main_result_has_no_diagnostics(simple, monkeypatch, minimize):
    calls = []
    def unknown(self):
        calls.append(self)
        return SolveResult(SolveStatus.UNKNOWN, reason_unknown="simulated")
    monkeypatch.setattr(SolverContext, "solve", unknown)
    result = solve_simple(simple, diagnose=True, minimize_conflict=minimize)
    assert len(calls) == 1
    assert result.status is SolveStatus.UNKNOWN and result.goals_satisfied is None
    assert result.diagnostics is None and result.diagnostic_solver_calls == 0


def test_conflicting_goals_have_distinct_records_and_sources(simple):
    result = solve_simple(simple, diagnose=True)
    diagnostic = result.diagnostics
    assert diagnostic.status is SolveStatus.UNSAT
    assert diagnostic.conflict_ids == (0, 1)
    assert diagnostic.conflict_categories == diagnostic.categories_required == (Category.FINAL_GOAL,)
    assert diagnostic.irreducible is False and diagnostic.minimize_requested is False
    assert [record.id for record in diagnostic.records] == [0, 1]
    assert tuple(record.constraint for record in diagnostic.records) == result.constraints
    for index, record in enumerate(diagnostic.records):
        assert record.category is Category.FINAL_GOAL
        assert record.source.goal_index == index and record.source.register == "RAX"
        assert record.source.sequence_index == 0


def test_equal_goals_keep_distinct_identity(simple):
    sequence, initial, _ = simple
    goals = [RegisterGoal("rax", 1), RegisterGoal("rax", 1)]
    result = solve_sequence(sequence, goals, initial_state=initial, diagnose=True, minimize_conflict=True,
                            constraints=[Equality(Symbol("rax_0"), 2)])
    records = result.diagnostics.records
    assert records[0].constraint == records[1].constraint
    assert records[0].id != records[1].id and records[0].source.goal_index == 0 and records[1].source.goal_index == 1
    assert result.diagnostics.conflict_ids == (1, 2)
    assert result.diagnostics.irreducible is True


def test_reduction_removes_irrelevant_constraint_with_exact_calls(simple, monkeypatch):
    contexts = []
    original = SolverContext.solve
    def counted(self):
        contexts.append(self)
        return original(self)
    monkeypatch.setattr(SolverContext, "solve", counted)
    result = solve_simple(simple, diagnose=True, minimize_conflict=True,
                          constraints=[Equality(Symbol("unrelated"), 7)])
    diagnostic = result.diagnostics
    assert diagnostic.conflict_ids == (0, 1)
    assert diagnostic.conflict_categories == (Category.FINAL_GOAL,)
    assert diagnostic.irreducible is True and diagnostic.partial is False
    assert diagnostic.diagnostic_solver_calls == diagnostic.solver_calls == result.diagnostic_solver_calls == 5
    assert len(contexts) == len({id(context) for context in contexts}) == 6
    categories = category_map(diagnostic)
    assert categories[Category.FINAL_GOAL].required_for_unsat is True
    assert categories[Category.AUXILIARY].required_for_unsat is False
    assert [item.removed_id for item in diagnostic.reduction_checks] == [0, 1, 2]
    assert [item.status for item in diagnostic.reduction_checks] == [SolveStatus.SAT, SolveStatus.SAT, SolveStatus.UNSAT]


def test_reduced_conflict_is_unsat_and_every_deletion_is_sat(simple):
    result = solve_simple(simple, diagnose=True, minimize_conflict=True,
                          constraints=[Equality(Symbol("irrelevant"), 7)])
    diagnostic = result.diagnostics
    core = [record.constraint for record in diagnostic.records if record.id in diagnostic.conflict_ids]
    for removed in [None, *range(len(core))]:
        problem = SolverContext()
        for index, equality in enumerate(core):
            if index != removed:
                problem.add(equality)
        assert problem.solve().status is (SolveStatus.UNSAT if removed is None else SolveStatus.SAT)


@pytest.mark.parametrize("minimize", [False, True])
def test_category_unknown_stays_unknown(simple, monkeypatch, minimize):
    original = diagnostics_module._solve_subset
    calls = []
    def simulated(records, **kwargs):
        calls.append(records)
        return SolveStatus.UNKNOWN if len(calls) == 1 else original(records, **kwargs)
    monkeypatch.setattr(diagnostics_module, "_solve_subset", simulated)
    diagnostic = solve_simple(simple, diagnose=True, minimize_conflict=minimize).diagnostics
    item = diagnostic.category_diagnostics[0]
    assert item.without_category_status is SolveStatus.UNKNOWN and item.required_for_unsat is None
    assert diagnostic.partial is True and diagnostic.unknown_solver_calls == 1
    assert diagnostic.irreducible is minimize  # Category uncertainty is separate from deletion proofs.
    assert diagnostic.conflict_ids == (0, 1)


def test_deletion_unknown_keeps_constraint_and_prevents_irreducibility(simple, monkeypatch):
    original = diagnostics_module._solve_subset
    calls = []
    def simulated(records, **kwargs):
        calls.append(tuple(record.id for record in records))
        # Two category checks, then deletion checks 0,1,2. Last removes an irrelevant equality.
        return SolveStatus.UNKNOWN if len(calls) == 5 else original(records, **kwargs)
    monkeypatch.setattr(diagnostics_module, "_solve_subset", simulated)
    diagnostic = solve_simple(simple, diagnose=True, minimize_conflict=True,
                              constraints=[Equality(Symbol("irrelevant"), 7)]).diagnostics
    assert diagnostic.conflict_ids == (0, 1, 2)
    assert diagnostic.irreducible is False and diagnostic.partial is True
    assert diagnostic.reduction_checks[-1].status is SolveStatus.UNKNOWN
    assert diagnostic.unknown_solver_calls == 1
    assert diagnostic.diagnostic_solver_calls == len(calls) == 5


def test_all_explicit_constraints_recorded_with_correct_sources():
    sequence = [gadget("c3", 0x4000), gadget("c3", 0x4100)]
    initial = make_symbolic_initial_state(2)
    result = solve_sequence(sequence, {"RAX": 2}, initial_state=initial, check_entry=True,
                            checkpoints={0: {"rax": 1}}, constraints=[Equality(Symbol("extra"), 3)],
                            diagnose=True, minimize_conflict=True)
    records = result.diagnostics.records
    assert [record.id for record in records] == list(range(7))
    assert tuple(record.constraint for record in records) == result.constraints
    assert all(record.constraint is equality for record, equality in zip(records, result.constraints))
    assert [record.category for record in records] == [Category.MEMORY, Category.MEMORY, Category.INITIAL,
                                                      Category.CONTROL_FLOW, Category.CHECKPOINT,
                                                      Category.FINAL_GOAL, Category.AUXILIARY]
    assert [r.source.memory_binding_index for r in records[:2]] == [0, 1]
    assert records[2].source.sequence_index == 0 and records[2].source.source_sequence_index is None
    assert records[3].source.sequence_index == 1 and records[3].source.source_sequence_index == 0
    assert records[3].source.register == "rip"
    assert records[4].source.checkpoint_index == records[4].source.sequence_index == 0
    assert records[4].source.register == "rax"
    assert records[5].source.sequence_index == 1 and records[5].source.goal_index == 0
    assert records[6].source.auxiliary_index == 0
    assert result.diagnostics.conflict_ids == (4, 5)


def test_memory_removal_does_not_silently_restore_initial_bindings():
    initial = make_symbolic_initial_state(1)
    initial.set_stack_slot(0, 1)
    auxiliary = Equality(MemoryRead(Symbol("rsp_0")), 2)
    result = solve_sequence([gadget("c3", 0x4000)], initial_state=initial,
                            constraints=[auxiliary], diagnose=True, minimize_conflict=True)
    diagnostic = result.diagnostics
    assert diagnostic.conflict_categories == (Category.MEMORY, Category.AUXILIARY)
    assert diagnostic.categories_required == (Category.MEMORY, Category.AUXILIARY)
    assert all(item.required_for_unsat is True for item in diagnostic.category_diagnostics)
    assert diagnostic.conflict_ids == (0, 1) and diagnostic.irreducible is True


def test_concrete_and_symbolic_memory_bindings_are_all_visible():
    initial = make_symbolic_initial_state(1)
    initial.write_memory(0x1000, 5)
    result = solve_sequence([gadget("c3", 0x4000)], initial_state=initial,
                            constraints=[Equality(MemoryRead(Symbol("address")), 6), Equality(Symbol("address"), 0x1000)],
                            diagnose=True, minimize_conflict=True)
    assert len([r for r in result.diagnostics.records if r.category is Category.MEMORY]) == 2
    assert result.diagnostics.conflict_ids == (0, 2, 3)
    assert result.diagnostics.irreducible is True


@pytest.mark.parametrize("entry", [False, True])
def test_entry_and_control_flow_diagnostics(entry):
    initial = make_symbolic_initial_state(1)
    initial.set_register("rip", 0xDEAD)
    initial.set_stack_slot(0, 0xDEAD)
    sequence = [gadget("c3", 0x4000)]
    if not entry:
        sequence.append(gadget("c3", 0x4100))
        initial.set_stack_slot(1, 0xDEAD)
    result = solve_sequence(sequence, initial_state=initial, check_entry=entry,
                            diagnose=True, minimize_conflict=True)
    category = Category.INITIAL if entry else Category.CONTROL_FLOW
    assert result.diagnostics.conflict_categories == (category,)
    assert category_map(result.diagnostics)[category].required_for_unsat is True


def test_stable_records_reduction_and_json(simple):
    r1 = solve_simple(simple, diagnose=True, minimize_conflict=True)
    r2 = solve_simple(simple, diagnose=True, minimize_conflict=True)
    assert r1.diagnostics == r2.diagnostics
    data = r1.diagnostics.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data == r2.diagnostics.to_dict()
    assert "object at" not in json.dumps(data) and "0x" not in json.dumps(data)
    assert all("constraint" not in record for record in data["records"])
    data["records"][0]["source"]["goal_index"] = 99
    assert r1.diagnostics.records[0].source.goal_index == 0
    with pytest.raises(FrozenInstanceError):
        r1.diagnostics.records[0].id = 99
    with pytest.raises(FrozenInstanceError):
        r1.diagnostics.irreducible = False


def test_inputs_and_original_results_are_unchanged(simple):
    sequence, initial, goals = simple
    auxiliary = [Equality(Symbol("extra"), 7)]
    checkpoints = {0: {"rbx": 3}}
    before = (sequence.copy(), goals.copy(), auxiliary.copy(), initial.register_bindings(), initial.memory_bindings())
    plain = solve_sequence(sequence, goals, initial_state=initial, constraints=auxiliary, checkpoints=checkpoints)
    diagnosed = solve_sequence(sequence, goals, initial_state=initial, constraints=auxiliary, checkpoints=checkpoints,
                                diagnose=True, minimize_conflict=True)
    assert before == (sequence, goals, auxiliary, initial.register_bindings(), initial.memory_bindings())
    assert checkpoints == {0: {"rbx": 3}}
    assert plain.status is diagnosed.status is SolveStatus.UNSAT
    assert plain.goals_satisfied is diagnosed.goals_satisfied is False
    assert plain.effect == diagnosed.effect
    assert plain.constraints == diagnosed.constraints
    assert plain.sequence_constraints == diagnosed.sequence_constraints
    assert plain.final_goal_constraints == diagnosed.final_goal_constraints
    assert plain.diagnostics is None


def test_effect_api_and_generator_inputs_match_sequence_api(simple):
    sequence, initial, goals = simple
    original = solve_sequence(sequence, goals, initial_state=initial)
    diagnosed = solve_sequence_effect(original.effect, iter(goals), constraints=iter([Equality(Symbol("x"), 3)]),
                                       diagnose=True, minimize_conflict=True)
    direct = solve_sequence(sequence, goals, initial_state=initial, constraints=[Equality(Symbol("x"), 3)],
                            diagnose=True, minimize_conflict=True)
    assert diagnosed.effect is original.effect
    assert diagnosed.diagnostics == direct.diagnostics


def test_diagnostic_contexts_forward_options_and_have_no_hidden_state(simple, monkeypatch):
    original = diagnostics_module.SolverContext
    seen = []
    def traced(*args, **kwargs):
        assert not args
        assert kwargs == {"memory_name": "diagnostic_memory", "timeout_ms": 2000}
        context = original(**kwargs)
        assert context.constraints == ()
        seen.append(context)
        return context
    monkeypatch.setattr(diagnostics_module, "SolverContext", traced)
    result = solve_simple(simple, diagnose=True, minimize_conflict=True, memory_name="diagnostic_memory", timeout_ms=2000)
    assert len(seen) == result.diagnostic_solver_calls == 3


def test_empty_context_is_sat_and_empty_unsat_is_rejected(simple, monkeypatch):
    assert SolverContext().solve().status is SolveStatus.SAT
    def unsat(self):
        return SolveResult(SolveStatus.UNSAT)
    monkeypatch.setattr(SolverContext, "solve", unsat)
    with pytest.raises(RuntimeError, match="empty diagnostic constraint set"):
        solve_simple(simple, diagnose=True)


def test_empty_original_unsat_is_rejected(simple, monkeypatch):
    sequence, initial, _ = simple
    monkeypatch.setattr(SolverContext, "solve", lambda self: SolveResult(SolveStatus.UNSAT))
    with pytest.raises(RuntimeError, match="empty original constraint set"):
        solve_sequence(sequence, initial_state=initial, diagnose=True)


@pytest.mark.parametrize("kwargs,error", [
    ({"minimize_conflict": True}, ValueError), ({"diagnose": 1}, TypeError),
    ({"diagnose": True, "minimize_conflict": "yes"}, TypeError),
])
def test_invalid_options(simple, kwargs, error):
    with pytest.raises(error):
        solve_simple(simple, **kwargs)


def test_search_trace_and_verification_do_not_start_diagnostics(simple, monkeypatch):
    sequence, initial, goals = simple
    found = search_sequences(sequence, goals[:1], max_depth=1, initial_state=initial, record_trace=True)
    saved = found.trace.to_dict()
    counts = (found.solver_calls, found.tested_candidates, found.prefix_checks)
    diagnosed = solve_sequence(found.sequence, goals, initial_state=initial, diagnose=True, minimize_conflict=True)
    assert diagnosed.status is SolveStatus.UNSAT
    assert found.trace.to_dict() == saved
    assert (found.solver_calls, found.tested_candidates, found.prefix_checks) == counts
    assert found.solve_result.diagnostics is None
    def unexpected(*args, **kwargs):
        pytest.fail("search/verification must not start diagnostics")
    monkeypatch.setattr(diagnostics_module, "_diagnose_unsat", unexpected)
    replay = verify_search_result(found, goals, initial_state=initial)
    assert replay.verified is False and replay.replay_result.diagnostics is None
    failed = search_sequences(sequence, goals, max_depth=2, initial_state=initial, pruning=True, record_trace=True)
    assert failed.trace.validate(failed)
