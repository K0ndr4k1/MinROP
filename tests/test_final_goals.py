from dataclasses import FrozenInstanceError

import pytest

from minirop import analyze_sequence
from minirop.constraints import Equality, RegisterGoal
from minirop.solver import (RegisterGoal as SolverRegisterGoal, SolveResult, SolveStatus,
                           SolverContext, solve_sequence, solve_sequence_effect)
from minirop.state import CPUState, InvalidRegister, make_symbolic_initial_state
from minirop.symbolic import Constant, NonConcreteValue, Symbol, WORD_MASK, add_values, evaluate
from test_sequence import gadget, pieces


def test_single_goal_and_reported_constraints():
    effect = analyze_sequence(pieces())
    result = solve_sequence_effect(effect, goals=[RegisterGoal("RAX", 19)])
    assert result.status is SolveStatus.SAT and result.goals_satisfied is True
    assert result.effect is effect
    assert result.final_goals == (RegisterGoal("RAX", 19),)
    assert result.final_goal_constraints == (Equality(effect.final_state.get_register("rax"), 19),)
    assert result.solution.assignments["stack_0"] == 12
    assert len(result.constraints) == len(result.sequence_constraints) + len(result.final_goal_constraints)


def test_multiple_goals_and_relative_initial_rsp():
    initial = make_symbolic_initial_state(4)
    effect = analyze_sequence(pieces(), initial)
    goals = [RegisterGoal("rax", 19), RegisterGoal("rbx", 77), RegisterGoal("rip", 0x5000),
             RegisterGoal("rsp", add_values(effect.initial_state.get_register("rsp"), 0x20))]
    result = solve_sequence_effect(effect, goals)
    assert result.goals_satisfied is True
    for equality in result.final_goal_constraints:
        assert evaluate(equality.left, result.solution.assignments, memory=result.solution.memory_values) == evaluate(
            equality.right, result.solution.assignments, memory=result.solution.memory_values)
    assert result.effect.final_state.get_register("rbx") == Symbol("rbx_0")


def test_absolute_final_rsp_goal():
    result = solve_sequence(pieces(), goals={"rsp": 0x1020})
    assert result.goals_satisfied is True
    assert result.solution.assignments["rsp_0"] == 0x1000


@pytest.mark.parametrize("custom_rsp", [Symbol("caller_stack"), Constant(0x3000), 0x3000])
def test_relative_goal_uses_actual_initial_state_value(custom_rsp):
    if isinstance(custom_rsp, Symbol):
        initial = CPUState(stack_base=custom_rsp)
        initial.set_stack_slot(0, Symbol("return_0"))
        initial.set_stack_slot(1, Symbol("return_1"))
    else:
        initial = CPUState()
        initial.write_memory(0x3000, 0x4100)
        initial.write_memory(0x3008, 0x5000)
    initial.set_register("rsp", custom_rsp)
    effect = analyze_sequence([gadget("c3", 0x4000), gadget("c3", 0x4100)], initial)
    target = add_values(effect.initial_state.get_register("rsp"), 16)
    result = solve_sequence_effect(effect, {"rsp": target})
    assert result.goals_satisfied is True
    assert result.final_goals == (RegisterGoal("rsp", target),)


def test_wrong_relative_rsp_is_unsat_for_every_initial_value():
    effect = analyze_sequence(pieces())
    result = solve_sequence_effect(effect, {"rsp": add_values(effect.initial_state.get_register("rsp"), 0x18)})
    assert result.status is SolveStatus.UNSAT
    assert result.goals_satisfied is False


def test_direct_final_state_conflict_does_not_assign():
    effect = analyze_sequence([gadget("48 c7 c0 01 00 00 00 c3", 0x4000)])
    before = effect.final_state
    result = solve_sequence_effect(effect, [RegisterGoal("rax", 2)])
    assert result.status is SolveStatus.UNSAT and result.goals_satisfied is False
    assert result.effect.final_state is before
    assert result.effect.final_state.get_register("rax") == 1
    assert result.final_goal_constraints == (Equality(1, 2),)
    assert result.solution.assignments == {}


@pytest.mark.parametrize("second_name", ["rax", "RAX"])
def test_conflicting_goals_for_same_register_are_not_overwritten(second_name):
    result = solve_sequence(pieces(), [RegisterGoal("rax", 1), RegisterGoal(second_name, 2)])
    assert result.status is SolveStatus.UNSAT
    assert len(result.final_goals) == len(result.final_goal_constraints) == 2
    assert result.final_goal_constraints[0].left == result.final_goal_constraints[1].left
    assert [constraint.right for constraint in result.final_goal_constraints] == [1, 2]


def test_compatible_duplicate_goals_are_retained():
    goals = [RegisterGoal("rax", 19), RegisterGoal("rax", Constant(19))]
    result = solve_sequence(pieces(), goals)
    assert result.goals_satisfied is True
    assert result.final_goals == tuple(goals)
    assert len(result.final_goal_constraints) == 2


def test_cross_step_checkpoint_conflict():
    result = solve_sequence(pieces(), [RegisterGoal("rax", 19)], checkpoints={0: {"rdi": 13}})
    assert result.status is SolveStatus.UNSAT
    assert Equality(Symbol("stack_0"), 13) in result.sequence_constraints
    assert result.final_goal_constraints[0].right == 19


def test_final_goal_and_return_link_conflict_share_solver():
    sequence = [gadget("c3", 0x4000), gadget("c3", 0x4100)]
    initial = make_symbolic_initial_state(2)
    initial.set_register("rax", Symbol("stack_0"))
    result = solve_sequence(sequence, [RegisterGoal("rax", 0x4200)], initial_state=initial)
    assert result.status is SolveStatus.UNSAT
    assert Equality(Symbol("stack_0"), 0x4100) in result.sequence_constraints
    assert result.final_goal_constraints == (Equality(Symbol("stack_0"), 0x4200),)


def test_final_goal_with_concrete_memory_binding():
    initial = make_symbolic_initial_state(4)
    initial.write_memory(0x1000, 12)
    fixed = [Equality(Symbol("rsp_0"), 0x1000)]
    compatible = solve_sequence(pieces(), {"rax": 19}, initial_state=initial, constraints=fixed)
    conflicting = solve_sequence(pieces(), {"rax": 20}, initial_state=initial, constraints=fixed)
    assert compatible.goals_satisfied is True
    assert conflicting.goals_satisfied is False


@pytest.mark.parametrize("pivot, delta", [("5c c3", 24), ("c9 c3", 32)])
def test_pivot_relative_final_rsp_goal(pivot, delta):
    sequence = [gadget(pivot, 0x4000), gadget("5f c3", 0x4100)]
    effect = analyze_sequence(sequence)
    target = add_values(effect.initial_state.get_register("rsp"), 0x20)
    result = solve_sequence_effect(effect, [RegisterGoal("rsp", target), RegisterGoal("rdi", 7)])
    assert result.goals_satisfied is True
    assert evaluate(effect.final_state.get_register("rsp"), result.solution.assignments,
                    memory=result.solution.memory_values) == evaluate(target, result.solution.assignments)
    # This is a relation between the pivot and initial RSP, not a fixed stack delta.
    base = "stack_0" if pivot == "5c c3" else "rbp_0"
    assert result.solution.assignments[base] == (result.solution.assignments["rsp_0"] + 0x20 - delta) & WORD_MASK


@pytest.mark.parametrize("empty", [None, {}, [], ()])
def test_no_goals_preserves_phase5_constraint_pipeline(empty):
    effect = analyze_sequence(pieces())
    baseline = solve_sequence_effect(effect, checkpoints={0: {"rdi": 12}})
    result = solve_sequence_effect(effect, empty, checkpoints={0: {"rdi": 12}})
    assert result.status == baseline.status == SolveStatus.SAT
    assert result.constraints == baseline.constraints == result.sequence_constraints
    assert result.final_goals == result.final_goal_constraints == ()
    assert result.effect is effect
    assert result.goals_satisfied is True


def test_goal_generator_is_consumed_once_and_input_list_is_detached():
    original = [RegisterGoal("rax", 19), RegisterGoal("rip", 0x5000)]
    result = solve_sequence(pieces(), (goal for goal in original))
    original.clear()
    assert result.goals_satisfied is True
    assert len(result.final_goals) == len(result.final_goal_constraints) == 2
    with pytest.raises(FrozenInstanceError):
        result.final_goals[0].expected = 0


def test_same_equality_from_checkpoint_goal_and_extra_keeps_provenance():
    sequence = [gadget("c3", 0x4000), gadget("c3", 0x4100)]
    equality = Equality(Symbol("rax_0"), 1)
    result = solve_sequence(sequence, {"rax": 1}, checkpoints={0: {"rax": 1}}, constraints=[equality])
    assert result.goals_satisfied is True
    assert result.constraints.count(equality) == 3
    assert result.sequence_constraints.count(equality) == 2
    assert result.final_goal_constraints == (equality,)


def test_single_context_single_check_contains_final_goals(monkeypatch):
    counts = {"contexts": 0, "checks": 0}
    old_init, old_solve = SolverContext.__init__, SolverContext.solve
    def counted_init(self, *args, **kwargs):
        counts["contexts"] += 1
        old_init(self, *args, **kwargs)
    def counted_solve(self):
        counts["checks"] += 1
        # Contradictory final equalities must both be present BEFORE the check.
        assert Equality(Symbol("rbx_0"), 1) in self.constraints
        assert Equality(Symbol("rbx_0"), 2) in self.constraints
        return old_solve(self)
    monkeypatch.setattr(SolverContext, "__init__", counted_init)
    monkeypatch.setattr(SolverContext, "solve", counted_solve)
    result = solve_sequence(pieces(), [RegisterGoal("rbx", 1), RegisterGoal("rbx", 2)])
    assert result.status is SolveStatus.UNSAT
    assert counts == {"contexts": 1, "checks": 1}


def test_unknown_is_not_reported_as_goal_failure_or_success(monkeypatch):
    monkeypatch.setattr(SolverContext, "solve", lambda self: SolveResult(SolveStatus.UNKNOWN, reason_unknown="timeout"))
    result = solve_sequence(pieces(), {"rax": 19})
    assert result.goals_satisfied is None
    assert result.status is SolveStatus.UNKNOWN
    assert result.final_goal_constraints
    assert result.solution.reason_unknown == "timeout"


def test_unsat_base_does_not_diagnose_individual_goal():
    initial = make_symbolic_initial_state(4)
    initial.set_stack_slot(1, 0xdead)
    effect = analyze_sequence(pieces(), initial)
    result = solve_sequence_effect(effect, {"rsp": effect.final_state.get_register("rsp")})
    assert result.goals_satisfied is False  # The goal is tautological, but the sequence is impossible.
    assert result.status is SolveStatus.UNSAT


@pytest.mark.parametrize("goals", [[("rax", 1)], [Equality(Symbol("x"), 1)], "rax", 0])
def test_invalid_goal_container(goals):
    with pytest.raises(TypeError):
        solve_sequence(pieces(), goals)


@pytest.mark.parametrize("name", ["eip", "invalid"])
def test_invalid_goal_register(name):
    with pytest.raises(InvalidRegister):
        solve_sequence(pieces(), [RegisterGoal(name, 1)])


@pytest.mark.parametrize("value", [None, object(), "symbol", True])
def test_invalid_expected_value(value):
    with pytest.raises(NonConcreteValue):
        solve_sequence(pieces(), [RegisterGoal("rax", value)])


def test_public_goal_import_and_nonstring_register():
    assert SolverRegisterGoal is RegisterGoal
    with pytest.raises(TypeError, match="register"):
        solve_sequence(pieces(), [RegisterGoal(1, 0)])
