import random

import pytest
import z3

from minirop import CodeRegion, ELFImage, GadgetScanner, execute_gadget
from minirop.solver import Equality, SolveStatus, SolverContext, Z3Translator, solve_effect, solve_gadget
from minirop.state import CPUState, InvalidRegister, make_symbolic_initial_state
from minirop.symbolic import (
    Add, Constant, MemoryRead, NonConcreteValue, Sub, Symbol, WORD_MASK, Xor,
    add_values, evaluate,
)
from minirop.semantics import UnsupportedInstruction


def gadget(code):
    data = bytes.fromhex(code)
    return next(g for g in GadgetScanner(max_gadget_bytes=64, max_instructions=12).scan(
        [CodeRegion("test", 0x4000, 0, data)]) if g.address == 0x4000 and g.bytes == data)


def test_symbol_identity_sort_and_context_isolation():
    translator = Z3Translator("input_snapshot")
    first = translator.translate(Symbol("x"))
    assert translator.translate(Symbol("x")) is first
    assert first.size() == 64
    assert z3.is_bv(first)
    assert translator.memory.domain().size() == 64
    assert translator.memory.range().size() == 8
    assert str(translator.memory.decl().name()) == "input_snapshot"
    other = Z3Translator()
    assert other.context is not translator.context
    with pytest.raises(TypeError):
        translator.symbols["x"] = other.translate(Symbol("x"))


@pytest.mark.parametrize("expr", [
    Constant(-1), Symbol("x"), Add(Symbol("x"), Constant(1)),
    Sub(Symbol("x"), Symbol("y")), Xor(Symbol("x"), Symbol("y")),
    Add(Sub(Symbol("x"), Constant(8)), Xor(Symbol("y"), Constant(255))),
    MemoryRead(Add(Symbol("base"), Constant(8))),
])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_translation_matches_independent_evaluation(expr, seed):
    rng = random.Random(seed)
    bindings = {"x": [0, WORD_MASK, 1 << 63, rng.getrandbits(64)][seed],
                "y": rng.getrandbits(64), "base": 0x1000}
    memory = {0x1008: rng.getrandbits(64)}
    translator = Z3Translator()
    term = translator.translate(expr)
    solver = z3.Solver(ctx=translator.context)
    for name, value in bindings.items():
        solver.add(translator.translate(Symbol(name)) == translator.translate(value))
    for address, value in memory.items():
        solver.add(translator.translate(MemoryRead(Constant(address))) == translator.translate(value))
    solver.add(term != translator.translate(evaluate(expr, bindings, memory=memory)))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("expression, goal, expected", [
    (Add(Symbol("x"), Constant(8)), 0x1337, 0x132f),
    (Add(Symbol("x"), Constant(1)), 0, WORD_MASK),
    (Sub(Symbol("x"), Constant(1)), WORD_MASK, 0),
    (Xor(Symbol("x"), Constant(0xff)), 0x1234, 0x12cb),
])
def test_solve_arithmetic(expression, goal, expected):
    problem = SolverContext()
    problem.add_equal(expression, goal)
    result = problem.solve()
    assert result.status is SolveStatus.SAT
    assert result.assignments["x"] == expected
    assert type(result.assignments["x"]) is int
    assert evaluate(expression, result.assignments) == goal


def test_expression_equality_and_contradiction():
    problem = SolverContext()
    problem.add(Equality(Add(Symbol("x"), Constant(8)), Symbol("y")))
    problem.add_equal(Symbol("y"), 20)
    sat = problem.solve()
    assert sat.assignments == {"x": 12, "y": 20}
    problem.add_equal(Symbol("x"), 13)
    unsat = problem.solve()
    assert unsat.status is SolveStatus.UNSAT
    assert unsat.assignments == unsat.memory_values == {}
    assert unsat.raw_model is None
    assert unsat.reason_unknown is None
    assert sat.assignments == {"x": 12, "y": 20}  # Earlier result is stable.
    with pytest.raises(TypeError):
        sat.assignments["x"] = 0


def test_memory_same_address_same_value():
    problem = SolverContext()
    problem.add_equal(MemoryRead(Symbol("x")), 1)
    problem.add_equal(MemoryRead(Symbol("x")), 2)
    assert problem.solve().status is SolveStatus.UNSAT


def test_solver_proven_address_alias_is_consistent():
    problem = SolverContext()
    problem.add_equal(Add(Symbol("x"), Constant(8)), Symbol("y"))
    problem.add_equal(MemoryRead(Add(Symbol("x"), Constant(8))), 1)
    problem.add_equal(MemoryRead(Symbol("y")), 2)
    assert problem.solve().status is SolveStatus.UNSAT


def test_distinct_addresses_can_hold_distinct_values():
    problem = SolverContext()
    problem.add_equal(Symbol("x"), 0x1000)
    problem.add_equal(Symbol("y"), 0x2000)
    problem.add_equal(MemoryRead(Symbol("x")), 1)
    problem.add_equal(MemoryRead(Symbol("y")), 2)
    result = problem.solve()
    assert result.status is SolveStatus.SAT
    assert result.memory_values == {0x1000: 1, 0x2000: 2}


def test_nested_memory_read_model_can_be_evaluated():
    problem = SolverContext()
    expression = MemoryRead(MemoryRead(Symbol("x")))
    problem.add_equal(Symbol("x"), 0x1000)
    problem.add_equal(MemoryRead(Symbol("x")), 0x2000)
    problem.add_equal(expression, 0x404000)
    result = problem.solve()
    assert result.status is SolveStatus.SAT
    assert evaluate(expression, result.assignments, memory=result.memory_values) == 0x404000


@pytest.mark.parametrize("index", [0, 1, 2, -1])
def test_named_stack_slot_cannot_conflict_with_memory(index):
    state = make_symbolic_initial_state(0)
    state.set_stack_slot(index, Symbol("custom_slot"))
    problem = SolverContext(state)
    problem.add_equal(Symbol("custom_slot"), 1)
    problem.add_equal(MemoryRead(add_values(Symbol("rsp_0"), index * 8)), 2)
    assert problem.solve().status is SolveStatus.UNSAT


def test_bindings_use_actual_base_and_values_not_naming_convention():
    state = CPUState(stack_base=Symbol("custom_base"))
    state.set_stack_slot(0, Add(Symbol("a"), Constant(1)))
    state.set_stack_slot(1, 55)
    state.set_register("rsp", Symbol("pivoted_rsp"))
    problem = SolverContext(state)
    problem.add_equal(Symbol("custom_base"), 0x1000)
    problem.add_equal(Symbol("a"), 9)
    result = problem.solve()
    assert result.memory_values == {0x1000: 10, 0x1008: 55}
    assert "pivoted_rsp" not in result.assignments


def test_concrete_words_and_stack_slots_share_snapshot():
    state = make_symbolic_initial_state(1)
    state.write_memory(0x1000, 42)
    problem = SolverContext(state)
    problem.add_equal(Symbol("rsp_0"), 0x1000)
    problem.add_equal(Symbol("stack_0"), 43)
    assert problem.solve().status is SolveStatus.UNSAT


def test_state_binding_snapshot_is_detached_and_immutable():
    state = make_symbolic_initial_state(1)
    state.set_stack_slot(0, 42)
    captured = state.memory_bindings()
    problem = SolverContext(state)
    state.set_stack_slot(0, 99)
    assert captured == ((Symbol("rsp_0"), 42),)
    problem.add_equal(MemoryRead(Symbol("rsp_0")), 42)
    assert problem.solve().status is SolveStatus.SAT


def test_independent_problems_do_not_share_memory():
    a, b = SolverContext(memory_name="snapshot_a"), SolverContext(memory_name="snapshot_b")
    a.add_equal(MemoryRead(Constant(0x1000)), 1)
    b.add_equal(MemoryRead(Constant(0x1000)), 2)
    assert a.solve().memory_values[0x1000] == 1
    assert b.solve().memory_values[0x1000] == 2


def test_goal_example_and_concrete_replay():
    target = gadget("5f 48 83 c7 08 c3")
    result = solve_gadget(target, {"rdi": 0x1337, "rip": 0x401000},
                          constraints=[Equality(Symbol("rsp_0"), 0x1000)])
    assert result.status is SolveStatus.SAT
    assert result.assignments["stack_0"] == 0x132f
    assert result.assignments["stack_1"] == 0x401000
    concrete = CPUState()
    concrete.set_register("rsp", result.assignments["rsp_0"])
    for address, value in result.memory_values.items():
        concrete.write_memory(address, value)
    replay = execute_gadget(target, concrete)
    assert replay.get_register("rdi") == 0x1337
    assert replay.get_register("rip") == 0x401000
    assert replay.get_register("rsp") == 0x1010


@pytest.mark.parametrize("code, goals", [
    ("48 31 c0 c3", {"rax": 1}),
    ("48 89 f8 c3", {"rax": 1, "rdi": 2}),
])
def test_impossible_gadget_goals(code, goals):
    assert solve_gadget(gadget(code), goals).status is SolveStatus.UNSAT


def test_solve_precomputed_effect_with_expression_goal():
    initial = make_symbolic_initial_state(2)
    target = gadget("5f 48 83 c7 08 c3")
    effect = execute_gadget(target, initial)
    result = solve_effect(effect, {"RDI": Symbol("wanted"), "rip": 0x401000},
                          initial_state=initial, constraints=[Equality(Symbol("wanted"), 0x1337)])
    assert result.assignments["stack_0"] == 0x132f
    assert initial.get_register("rsp") == Symbol("rsp_0")


def test_pop_rsp_aliases_original_stack():
    result = solve_gadget(gadget("5c c3"), {"rip": 99}, constraints=[
        Equality(Symbol("rsp_0"), 0x1000), Equality(Symbol("stack_0"), 0x1008),
        Equality(Symbol("stack_1"), 98),
    ])
    assert result.status is SolveStatus.UNSAT


def test_leave_aliases_original_stack():
    result = solve_gadget(gadget("c9 c3"), {"rbp": 77, "rip": 99}, constraints=[
        Equality(Symbol("rbp_0"), Symbol("rsp_0")), Equality(Symbol("stack_0"), 78),
    ])
    assert result.status is SolveStatus.UNSAT


@pytest.mark.parametrize("code, fixed", [
    ("5c c3", [Equality(Symbol("stack_0"), 0x2000)]),
    ("c9 c3", [Equality(Symbol("rbp_0"), 0x2000)]),
])
def test_pivot_models_replay_in_concrete_executor(code, fixed):
    target = gadget(code)
    result = solve_gadget(target, {"rip": 0x401000}, constraints=[
        Equality(Symbol("rsp_0"), 0x1000), *fixed,
    ])
    assert result.status is SolveStatus.SAT
    initial = CPUState()
    initial.set_register("rsp", result.assignments["rsp_0"])
    if code == "c9 c3":
        initial.set_register("rbp", result.assignments["rbp_0"])
    for address, value in result.memory_values.items():
        initial.write_memory(address, value)
    assert execute_gadget(target, initial).get_register("rip") == 0x401000


def test_unknown_is_not_unsat_and_does_not_extract_model(monkeypatch):
    # Deterministically test the UNKNOWN branch, without a flaky timing race.
    problem = SolverContext()
    monkeypatch.setattr(problem._solver, "check", lambda: z3.unknown)
    monkeypatch.setattr(problem._solver, "reason_unknown", lambda: "timeout")
    def no_model():
        pytest.fail("model must not be requested after UNKNOWN")
    monkeypatch.setattr(problem._solver, "model", no_model)
    result = problem.solve()
    assert result.status is SolveStatus.UNKNOWN
    assert result.reason_unknown == "timeout"
    assert result.assignments == result.memory_values == {}
    assert result.raw_model is None


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        SolverContext(timeout_ms=timeout)


def test_empty_problem_and_underconstrained_assignment():
    assert SolverContext(timeout_ms=None).solve().status is SolveStatus.SAT
    problem = SolverContext()
    problem.add_equal(Symbol("free"), Symbol("free"))
    result = problem.solve()
    assert type(result.assignments["free"]) is int
    assert 0 <= result.assignments["free"] <= WORD_MASK


def test_invalid_inputs_and_instruction_scope():
    with pytest.raises(ValueError):
        Z3Translator("")
    with pytest.raises(NonConcreteValue):
        Z3Translator().translate(object())
    with pytest.raises(TypeError):
        SolverContext().add(True)
    with pytest.raises(InvalidRegister):
        solve_gadget(gadget("c3"), {"eip": 1})
    with pytest.raises(UnsupportedInstruction):
        solve_gadget(gadget("0f a2 c3"), {"rax": 0})


def test_elf_discovery_to_solver(elf_path):
    target = next(g for g in GadgetScanner().scan(ELFImage.load(elf_path).executable_segments)
                  if g.address == 0x401000 and g.bytes == b"\x5f\xc3")
    result = solve_gadget(target, {"rdi": 0x1337, "rip": 0x401000})
    assert result.status is SolveStatus.SAT
    assert result.assignments["stack_0"] == 0x1337
    assert result.assignments["stack_1"] == 0x401000
