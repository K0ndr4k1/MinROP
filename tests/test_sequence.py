from dataclasses import FrozenInstanceError, replace

import pytest

from minirop import CodeRegion, ELFImage, GadgetScanner, analyze_sequence, execute_gadget
from minirop.constraints import Equality
from minirop.sequence import StateSnapshot
from minirop.semantics import UnsupportedInstruction
from minirop.solver import (
    Equality as ExistingEquality, SolveResult, SolveStatus, SolverContext,
    solve_gadget, solve_sequence, solve_sequence_effect,
)
from minirop.state import CPUState, InvalidRegister, MemoryAccessError, REGISTER_NAMES, UninitializedRegister, make_symbolic_initial_state
from minirop.symbolic import Add, Constant, MemoryRead, Sub, Symbol, WORD_MASK, evaluate


def gadget(code, address):
    data = bytes.fromhex(code)
    return next(g for g in GadgetScanner(max_gadget_bytes=64, max_instructions=12).scan(
        [CodeRegion("test", address, 0, data)]) if g.address == address and g.bytes == data)


def pieces():
    return [gadget("5f c3", 0x4000), gadget("48 89 f8 48 83 c0 08 c3", 0x4100),
            gadget("48 83 e8 01 c3", 0x4200)]


def test_state_flows_through_all_steps_without_resetting():
    initial = make_symbolic_initial_state(4)
    effect = analyze_sequence(pieces(), initial)
    assert len(effect.steps) == 3
    assert effect.steps[0].before is effect.initial_state
    assert effect.steps[1].before is effect.steps[0].after
    assert effect.steps[2].before is effect.steps[1].after
    assert effect.steps[1].before.get_register("rdi") == Symbol("stack_0")
    assert effect.steps[1].before.get_register("rsp") == Add(Symbol("rsp_0"), Constant(16))
    # Control-flow is a condition; the incoming RIP is never replaced by 0x4100.
    assert effect.steps[1].before.get_register("rip") == Symbol("stack_1")
    assert effect.final_state.get_register("rax") == Sub(Add(Symbol("stack_0"), Constant(8)), Constant(1))
    assert effect.final_state.get_register("rip") == Symbol("stack_3")
    assert effect.final_state.get_register("rsp") == Add(Symbol("rsp_0"), Constant(32))
    assert effect.final_state.get_register("rbx") == Symbol("rbx_0")
    assert effect.constraints == (Equality(Symbol("stack_1"), 0x4100), Equality(Symbol("stack_2"), 0x4200))
    assert initial.get_register("rdi") == Symbol("rdi_0")
    assert initial.get_register("rsp") == Symbol("rsp_0")


def test_concrete_values_and_stack_are_carried_forward():
    initial = CPUState()
    initial.set_register("rsp", 0x1000)
    initial.set_register("rbx", 77)
    for index, value in enumerate([12, 0x4100, 0x4200, 0x5000]):
        initial.write_memory(0x1000 + index * 8, value)
    result = solve_sequence(pieces(), {"rax": 19, "rip": 0x5000}, initial_state=initial)
    assert result.status is SolveStatus.SAT
    final = result.effect.final_state
    assert final.get_register("rsp") == 0x1020
    assert final.get_register("rdi") == 12
    assert final.get_register("rbx") == 77
    assert final.memory_bindings() == initial.memory_bindings()
    assert initial.get_register("rsp") == 0x1000


def test_unified_goals_and_complete_constraint_summary():
    initial = make_symbolic_initial_state(4)
    extra = Equality(Symbol("rsp_0"), 0x1000)
    result = solve_sequence(pieces(), {"rax": 19, "rip": 0x5000}, initial_state=initial,
                            checkpoints={0: {"rdi": 12}, 1: {"rax": 20}}, constraints=[extra])
    assert result.status is SolveStatus.SAT
    assert result.solution.assignments["stack_0"] == 12
    assert result.solution.assignments["stack_1"] == 0x4100
    assert result.solution.assignments["stack_2"] == 0x4200
    assert result.solution.assignments["stack_3"] == 0x5000
    # 4 input-memory bindings + 2 links + 2 checkpoints + 2 final goals + 1 input condition.
    assert len(result.constraints) == 11
    assert extra in result.constraints
    for equality in result.constraints:
        assert evaluate(equality.left, result.solution.assignments, memory=result.solution.memory_values) == evaluate(
            equality.right, result.solution.assignments, memory=result.solution.memory_values)


def test_individually_sat_steps_can_be_jointly_unsat():
    a, b = gadget("c3", 0x4000), gadget("c3", 0x4100)
    assert solve_gadget(a, {"rdi": 1}).status is SolveStatus.SAT
    assert solve_gadget(b, {"rdi": 2}).status is SolveStatus.SAT
    result = solve_sequence([a, b], checkpoints={0: {"rdi": 1}, 1: {"rdi": 2}})
    assert result.status is SolveStatus.UNSAT
    assert result.solution.assignments == {}


def test_incorrect_return_is_not_silently_overwritten():
    initial = make_symbolic_initial_state(3)
    initial.set_stack_slot(1, 0xdead)
    result = solve_sequence(pieces()[:2], initial_state=initial)
    assert result.status is SolveStatus.UNSAT
    assert result.effect.steps[1].before.get_register("rip") == 0xdead
    assert result.effect.constraints == (Equality(0xdead, 0x4100),)


def test_concrete_control_flow_mismatch():
    initial = CPUState()
    initial.set_register("rsp", 0x1000)
    initial.write_memory(0x1000, 0xdead)
    initial.write_memory(0x1008, 0x5000)
    result = solve_sequence([gadget("c3", 0x4000), gadget("c3", 0x4100)], initial_state=initial)
    assert result.status is SolveStatus.UNSAT


def test_reversing_order_is_not_auto_corrected():
    a, b = gadget("c3", 0x4000), gadget("c3", 0x4100)
    initial = make_symbolic_initial_state(2)
    initial.set_stack_slot(0, b.address)
    assert solve_sequence([a, b], initial_state=initial).status is SolveStatus.SAT
    result = solve_sequence([b, a], initial_state=initial)
    assert result.status is SolveStatus.UNSAT
    assert [step.gadget for step in result.effect.steps] == [b, a]


def test_single_step_keeps_phase4_behavior_and_final_target_is_free():
    target = pieces()[0]
    effect = analyze_sequence([target])
    assert effect.constraints == ()
    assert effect.final_state.get_register("rip") == Symbol("stack_1")
    result = solve_sequence([target], {"rdi": 12})
    assert result.status is SolveStatus.SAT
    assert result.solution.assignments["stack_0"] == 12


def test_optional_initial_rip_check():
    initial = make_symbolic_initial_state(2)
    initial.set_register("rip", 0xdead)
    target = pieces()[0]
    assert solve_sequence([target], initial_state=initial).status is SolveStatus.SAT
    result = solve_sequence([target], initial_state=initial, check_entry=True)
    assert result.status is SolveStatus.UNSAT
    assert result.effect.control_flow[0].source_index is None
    symbolic = solve_sequence([target], check_entry=True)
    assert symbolic.solution.assignments["rip_0"] == target.address


def test_entry_check_requires_initialized_rip():
    state = CPUState()
    with pytest.raises(UninitializedRegister):
        analyze_sequence([pieces()[0]], state, check_entry=True)


def test_pop_rsp_carries_new_base_to_next_piece():
    a, b = gadget("5c c3", 0x4000), gadget("5f c3", 0x4100)
    effect = analyze_sequence([a, b])
    assert effect.steps[1].before.get_register("rsp") == Add(Symbol("stack_0"), Constant(8))
    assert effect.final_state.get_register("rdi") == MemoryRead(Add(Symbol("stack_0"), Constant(8)))
    assert effect.final_state.get_register("rsp") == Add(Symbol("stack_0"), Constant(24))
    assert effect.constraints == (Equality(MemoryRead(Symbol("stack_0")), b.address),)


def test_leave_carries_frame_and_original_memory():
    effect = analyze_sequence([gadget("c9 c3", 0x4000), gadget("5f c3", 0x4100)])
    assert effect.steps[1].before.get_register("rbp") == MemoryRead(Symbol("rbp_0"))
    assert effect.final_state.get_register("rdi") == MemoryRead(Add(Symbol("rbp_0"), Constant(16)))
    assert effect.final_state.get_register("rsp") == Add(Symbol("rbp_0"), Constant(32))
    assert effect.final_state.memory_bindings() == effect.initial_state.memory_bindings()


def test_cross_step_memory_alias_constraints_share_one_snapshot():
    a = gadget("5f c3", 0x4000)
    b = gadget("48 89 ec 5e c3", 0x4100)  # mov rsp, rbp ; pop rsi ; ret
    result = solve_sequence([a, b], {"rsi": 2}, checkpoints={0: {"rdi": 1}},
                            constraints=[Equality(Symbol("rbp_0"), Symbol("rsp_0"))])
    assert result.status is SolveStatus.UNSAT


def test_earlier_observations_survive_later_overwrite():
    result = solve_sequence([gadget("c9 c3", 0x4000), gadget("5d c3", 0x4100)],
                            constraints=[Equality(Symbol("rbp_0"), 0x2000)])
    assert result.status is SolveStatus.SAT
    assert 0x2000 in result.solution.memory_values  # Old RBP read, overwritten at step 1.
    assert result.solution.memory_values[0x2008] == 0x4100


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("variant", ["ordinary", "pop_rsp", "leave"])
def test_sequence_model_replay_matches_every_step(seed, variant):
    if variant == "ordinary":
        sequence = pieces()
        goals = {"rax": [0, 1, WORD_MASK, 1 << 63][seed], "rip": 0x5000}
        fixed = []
    elif variant == "pop_rsp":
        sequence = [gadget("5c c3", 0x4000), gadget("5f c3", 0x4100)]
        goals = {"rdi": [0, 1, WORD_MASK, 1 << 63][seed], "rip": 0x5000}
        fixed = [Equality(Symbol("stack_0"), 0x2000)]
    else:
        sequence = [gadget("c9 c3", 0x4000), gadget("5f c3", 0x4100)]
        goals = {"rdi": [0, 1, WORD_MASK, 1 << 63][seed], "rip": 0x5000}
        fixed = [Equality(Symbol("rbp_0"), 0x2000)]
    result = solve_sequence(sequence, goals, check_entry=True,
                            constraints=[Equality(Symbol("rsp_0"), 0x1000), *fixed])
    assert result.status is SolveStatus.SAT
    model = result.solution
    state = CPUState()
    for name in REGISTER_NAMES:
        state.set_register(name, model.assignments.get(f"{name}_0", 0))
    for address, value in model.memory_values.items():
        state.write_memory(address, value)
    for index, step in enumerate(result.effect.steps):
        assert state.get_register("rip") == step.gadget.address
        state = execute_gadget(step.gadget, state)
        for name in REGISTER_NAMES:
            assert state.get_register(name) == evaluate(step.after.get_register(name), model.assignments,
                                                       memory=model.memory_values)
        if index + 1 < len(sequence):
            assert state.get_register("rip") == sequence[index + 1].address
    # Phase 6: the same concrete replay must also satisfy EVERY final goal.
    assert result.goals_satisfied is True
    for goal in result.final_goals:
        assert state.get_register(goal.register) == evaluate(goal.expected, model.assignments,
                                                             memory=model.memory_values)
    for equality in result.final_goal_constraints:
        assert evaluate(equality.left, model.assignments, memory=model.memory_values) == evaluate(
            equality.right, model.assignments, memory=model.memory_values)


def test_analysis_snapshots_are_immutable_and_detached():
    state = make_symbolic_initial_state(4)
    effect = analyze_sequence(pieces(), state)
    state.set_register("rsp", 0xdead)
    state.set_stack_slot(0, 999)
    assert effect.initial_state.get_register("rsp") == Symbol("rsp_0")
    assert effect.initial_state.memory_bindings()[0] == (Symbol("rsp_0"), Symbol("stack_0"))
    with pytest.raises(FrozenInstanceError):
        effect.final_state.registers = ()
    with pytest.raises(InvalidRegister):
        effect.final_state.get_register("eip")
    with pytest.raises(UninitializedRegister):
        StateSnapshot.capture(CPUState()).get_register("rax")
    assert solve_sequence_effect(effect, {"rax": 19}).status is SolveStatus.SAT


def test_one_solver_context_and_one_check(monkeypatch):
    counts = {"contexts": 0, "checks": 0}
    old_init, old_solve = SolverContext.__init__, SolverContext.solve
    def counted_init(self, *args, **kwargs):
        counts["contexts"] += 1
        old_init(self, *args, **kwargs)
    def counted_solve(self):
        counts["checks"] += 1
        return old_solve(self)
    monkeypatch.setattr(SolverContext, "__init__", counted_init)
    monkeypatch.setattr(SolverContext, "solve", counted_solve)
    assert solve_sequence(pieces(), {"rax": 19}).status is SolveStatus.SAT
    assert counts == {"contexts": 1, "checks": 1}


def test_unknown_propagation_does_not_claim_sequence_validity(monkeypatch):
    monkeypatch.setattr(SolverContext, "solve", lambda self: SolveResult(SolveStatus.UNKNOWN, reason_unknown="timeout"))
    result = solve_sequence(pieces())
    assert result.status is SolveStatus.UNKNOWN
    assert result.solution.reason_unknown == "timeout"
    assert result.solution.assignments == {}
    assert result.effect.constraints


def test_repeated_piece_and_matching_code_overlap_are_allowed():
    full = gadget("5f c3", 0x4000)
    suffix = gadget("c3", 0x4001)
    result = solve_sequence([full, suffix, suffix])
    assert result.status is SolveStatus.SAT
    assert result.solution.assignments["stack_1"] == suffix.address
    assert result.solution.assignments["stack_2"] == suffix.address


def test_conflicting_code_images_are_rejected():
    with pytest.raises(ValueError, match="conflicting code"):
        analyze_sequence([gadget("5f c3", 0x4000), gadget("5e c3", 0x4000)])


@pytest.mark.parametrize("value", [[], [object()]])
def test_invalid_sequence_input(value):
    with pytest.raises((ValueError, TypeError)):
        analyze_sequence(value)


@pytest.mark.parametrize("index", [-1, 3, True, "0"])
def test_invalid_checkpoint_index(index):
    with pytest.raises(ValueError, match="checkpoint"):
        solve_sequence(pieces(), checkpoints={index: {"rax": 0}})


def test_bad_code_addresses_and_bad_goal():
    target = pieces()[0]
    for address in (-1, WORD_MASK, WORD_MASK + 1):
        with pytest.raises(ValueError, match="address space"):
            analyze_sequence([replace(target, address=address)])
    with pytest.raises(InvalidRegister):
        solve_sequence(pieces(), {"eip": 1})


def test_later_execution_error_preserves_input():
    initial = make_symbolic_initial_state()
    before = initial.register_bindings(), initial.memory_bindings()
    with pytest.raises(UnsupportedInstruction):
        analyze_sequence([pieces()[0], gadget("0f a2 c3", 0x4100)], initial)
    assert (initial.register_bindings(), initial.memory_bindings()) == before
    concrete = CPUState()
    concrete.set_register("rsp", 0x1000)
    concrete.write_memory(0x1000, 0x4100)
    with pytest.raises(MemoryAccessError):
        analyze_sequence([gadget("c3", 0x4000), gadget("c3", 0x4100)], concrete)
    assert concrete.get_register("rsp") == 0x1000


def test_constraint_api_remains_compatible_and_reports_bindings():
    assert ExistingEquality is Equality
    problem = SolverContext(make_symbolic_initial_state(1))
    captured = problem.constraints
    assert captured == (Equality(MemoryRead(Symbol("rsp_0")), Symbol("stack_0")),)
    problem.add_equal(Symbol("x"), 1)
    assert len(captured) == 1 and len(problem.constraints) == 2


def test_elf_discovered_sequence_and_generator_input(elf_path):
    found = GadgetScanner().scan(ELFImage.load(elf_path).executable_segments)
    first = next(g for g in found if g.bytes == b"\x5f\xc3" and g.address == 0x401000)
    second = next(g for g in found if g.bytes == b"\x5e\x41\x5f\xc3" and g.address == 0x401003)
    result = solve_sequence((g for g in [first, second]), {"rdi": 1, "rsi": 2, "r15": 3})
    assert result.status is SolveStatus.SAT
    assert result.solution.assignments["stack_1"] == second.address
    assert result.effect.final_state.get_register("rsp") == Add(Symbol("rsp_0"), Constant(40))
