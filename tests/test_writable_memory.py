"""Functional checks for byte memory transitions and ordinary stack operations."""
import pytest

from minirop.constraints import Equality
from minirop.disasm import Disassembler
from minirop.executor import execute, execute_gadget
from minirop.manifest import create_analysis_manifest
from minirop.search import SearchStatus, search_sequences
from minirop.semantics import UnsupportedInstruction, lift_instruction
from minirop.solver import SolveStatus, SolverContext, solve_gadget, solve_sequence
from minirop.state import CPUState, MemoryAccessError, make_symbolic_initial_state
from minirop.symbolic import (Constant, InitialMemory, MemoryRead, MemoryStore,
                              Symbol, WORD_MASK, add_values, evaluate, read64, write64)
from test_sequence import gadget


def ops(code):
    return tuple(lift_instruction(i) for i in Disassembler().disassemble(bytes.fromhex(code)))


def fixed(**registers):
    state = CPUState()
    for name, value in registers.items():
        state.set_register(name, value)
    return state


def prove_equal(left, right, *, initial=None, constraints=()):
    context = SolverContext(initial)
    for equality in constraints:
        context.add(equality)
    context._solver.add(context.translator.translate(left) != context.translator.translate(right))
    assert context.solve().status is SolveStatus.UNSAT


def test_mov_store_then_load_all_basic_forms():
    # Base, positive displacement, negative displacement, and sign-extended imm32.
    for code, registers, expected in [
        ("48 89 18 48 8b 08", {"rax": 0x2000, "rbx": 0x123}, 0x123),
        ("48 89 58 08 48 8b 48 08", {"rax": 0x2000, "rbx": 0x456}, 0x456),
        ("48 89 4d f8 48 8b 55 f8", {"rbp": 0x2000, "rcx": 0x789}, 0x789),
        ("48 c7 07 00 00 00 00 48 8b 0f", {"rdi": 0x2000}, 0),
        ("48 c7 46 10 01 00 00 00 48 8b 4e 10", {"rsi": 0x2000}, 1),
        ("48 c7 00 ff ff ff ff 48 8b 08", {"rax": 0x2000}, WORD_MASK),
    ]:
        initial = fixed(**registers)
        result = execute(ops(code), initial)
        target = "rdx" if "rbp" in registers else "rcx"
        assert result.get_register(target) == expected
        assert all(result.get_register(k) == v for k, v in registers.items())
        assert isinstance(initial.memory_state, InitialMemory)
        symbolic = execute(ops(code), make_symbolic_initial_state())
        source = Symbol("rcx_0") if target == "rdx" else Symbol("rbx_0")
        prove_equal(symbolic.get_register(target), source if "rbx" in registers or target == "rdx" else expected)


def test_consecutive_stores_latest_value_wins():
    code = "48 89 18 48 89 08 48 8b 10"
    result = execute(ops(code), fixed(rax=0x2000, rbx=1, rcx=2))
    assert result.get_register("rdx") == 2
    result = execute(ops(code), make_symbolic_initial_state())
    prove_equal(result.get_register("rdx"), Symbol("rcx_0"))


def test_concrete_unaligned_partial_overlaps_are_little_endian():
    state = CPUState()
    state.write_memory(0x1001, 0x0807060504030201)
    state.write_memory(0x1009, 0x100f0e0d0c0b0a09)
    assert state.read_memory(0x1002) == 0x0908070605040302
    state.store_memory(0x1002, 0xffeeddccbbaa9988)
    assert state.read_memory(0x1001) == 0xeeddccbbaa998801
    assert state.read_memory(0x1003) == 0x0affeeddccbbaa99
    with pytest.raises(MemoryAccessError):
        state.read_memory(0x3000)


def test_read_after_write_identity_is_an_array_theorem():
    a, v = Symbol("A"), Symbol("V")
    memory = write64(InitialMemory(), a, v)
    assert read64(memory, a) == v
    # Bypass the structural simplifier to test the actual eight Store/Select terms.
    prove_equal(MemoryRead(a, memory), v)


def test_symbolic_equal_addresses_alias_even_with_different_expressions():
    initial = make_symbolic_initial_state()
    result = execute(ops("48 89 18 48 8b 0a"), initial)
    prove_equal(result.get_register("rcx"), Symbol("rbx_0"), initial=initial,
                constraints=[Equality(Symbol("rax_0"), Symbol("rdx_0"))])
    a = Symbol("A")
    memory = write64(InitialMemory(), add_values(a, 8), 91)
    prove_equal(MemoryRead(add_values(add_values(a, 4), 4), memory), 91)


def test_disjoint_symbolic_addresses_preserve_old_memory():
    a, b = Symbol("A"), Symbol("B")
    memory = write64(InitialMemory(), a, 5)
    initial_read, current_read = MemoryRead(b), MemoryRead(b, memory)
    prove_equal(current_read, initial_read, constraints=[Equality(b, add_values(a, 16))])


def test_unresolved_alias_is_neither_assumed_equal_nor_disjoint():
    a, b = Symbol("A"), Symbol("B")
    value = read64(write64(InitialMemory(), a, 5), b)
    assert isinstance(value, MemoryRead)
    context = SolverContext()
    context.add_equal(a, 0x1000)
    context.add_equal(b, 0x2000)
    context.add_equal(MemoryRead(b), 7)
    context.add_equal(value, 7)
    assert context.solve().status is SolveStatus.SAT
    context = SolverContext()
    context.add_equal(a, b)
    context.add_equal(value, 7)
    assert context.solve().status is SolveStatus.UNSAT


def test_symbolic_partial_overlap_including_unequal_addresses():
    a = Symbol("A")
    memory = write64(InitialMemory(), a, 0x0807060504030201)
    value = MemoryRead(add_values(a, 1), memory)
    prove_equal(value, 0x0908070605040302,
                constraints=[Equality(MemoryRead(add_values(a, 8)), 9)])
    # A+1 differs from A, but seven bytes still alias.
    assert evaluate(value, {"A": 0x1001}, memory={0x1009: 9}) == 0x0908070605040302


def test_address_wraparound_is_shared_by_concrete_evaluation_and_smt():
    start = WORD_MASK - 3
    state = CPUState()
    state.write_memory(start, 0x0807060504030201)
    assert state.read_memory(start) == 0x0807060504030201
    state.store_memory(0, 0xffeeddccbbaa9988)
    assert state.read_memory(start) == 0xbbaa998804030201
    value = MemoryRead(Constant(start), write64(InitialMemory(), 0, 0xffeeddccbbaa9988))
    prove_equal(value, 0xbbaa998804030201, initial=state)
    assert evaluate(value, {}, memory={start: 0x0807060504030201}) == 0xbbaa998804030201
    pushed = execute(ops("50 5b"), fixed(rsp=4, rax=19))
    assert pushed.get_register("rbx") == 19 and pushed.get_register("rsp") == 4


def test_initial_constraints_survive_runtime_overwrite():
    initial = fixed(rax=0x2000, rbx=22)
    initial.write_memory(0x2000, 11)
    result = execute(ops("48 89 18 48 8b 08"), initial)
    assert result.read_memory(0x2000) == result.get_register("rcx") == 22
    assert initial.read_memory(0x2000) == 11
    assert result.memory_bindings() == initial.memory_bindings() == ((0x2000, 11),)
    context = SolverContext(result)
    context.add_equal(MemoryRead(Constant(0x2000)), 12)
    assert context.solve().status is SolveStatus.UNSAT
    with pytest.raises(MemoryAccessError):
        result.write_memory(0x2000, 33)


def test_state_branches_and_failed_execution_do_not_mutate_input():
    initial = fixed(rax=0x2000, rbx=1)
    initial.write_memory(0x2000, 10)
    first, second = initial.copy(), initial.copy()
    first.store_memory(0x2000, 20)
    second.store_memory(0x2000, 30)
    assert [s.read_memory(0x2000) for s in (initial, first, second)] == [10, 20, 30]
    child = first.copy()
    assert child.memory_state is first.memory_state
    child.store_memory(0x2000, 40)
    assert first.read_memory(0x2000) == 20
    with pytest.raises(MemoryAccessError):
        execute(ops("48 89 18 48 8b 48 10"), initial)
    assert initial.read_memory(0x2000) == 10


def test_push_pop_round_trip_uses_same_memory():
    for code, source in [("50 5b", "rax"), ("55 5b", "rbp"), ("41 50 5b", "r8")]:
        result = execute(ops(code), fixed(rsp=0x1000, **{source: 37}))
        assert result.get_register("rbx") == 37
        assert result.get_register("rsp") == 0x1000
        assert result.read_memory(0xff8) == 37
        symbolic = execute(ops(code), make_symbolic_initial_state())
        prove_equal(symbolic.get_register("rbx"), Symbol(source + "_0"))
        prove_equal(symbolic.get_register("rsp"), Symbol("rsp_0"))


def test_multiple_push_pop_obeys_lifo_and_restores_rsp():
    code = "50 53 59 5a"
    result = execute(ops(code), fixed(rsp=0x1000, rax=17, rbx=29))
    assert (result.get_register("rcx"), result.get_register("rdx"), result.get_register("rsp")) == (29, 17, 0x1000)
    result = execute(ops(code), make_symbolic_initial_state())
    prove_equal(result.get_register("rcx"), Symbol("rbx_0"))
    prove_equal(result.get_register("rdx"), Symbol("rax_0"))
    prove_equal(result.get_register("rsp"), Symbol("rsp_0"))


def test_push_rsp_saves_predecrement_value():
    result = execute(ops("54 58"), fixed(rsp=0x1000))
    assert result.get_register("rax") == result.get_register("rsp") == 0x1000
    result = execute(ops("54 58"), make_symbolic_initial_state())
    prove_equal(result.get_register("rax"), Symbol("rsp_0"))
    prove_equal(result.get_register("rsp"), Symbol("rsp_0"))


def test_push_signed_imm8_and_imm32():
    for code, expected in [("6a ff", WORD_MASK), ("6a 80", WORD_MASK - 127),
                           ("68 ff ff ff ff", WORD_MASK), ("68 00 00 00 80", (1 << 64) - (1 << 31))]:
        result = execute(ops(code + " 58"), fixed(rsp=0x1000))
        assert result.get_register("rax") == expected
        assert result.get_register("rsp") == 0x1000
        symbolic = execute(ops(code + " 58"), make_symbolic_initial_state())
        prove_equal(symbolic.get_register("rax"), expected)


def test_pop_rsp_keeps_popped_value_and_ret_reads_pushed_target():
    result = execute(ops("50 5c"), fixed(rsp=0x1000, rax=0x2000))
    assert result.get_register("rsp") == 0x2000
    symbolic = execute(ops("50 5c"), make_symbolic_initial_state())
    prove_equal(symbolic.get_register("rsp"), Symbol("rax_0"))
    part = gadget("50 c3", 0x4000)
    result = execute_gadget(part, fixed(rsp=0x1000, rax=0x9000))
    assert result.get_register("rsp") == 0x1000 and result.get_register("rip") == 0x9000


def test_leave_and_ret_read_the_current_memory_version():
    code = "48 89 45 00 48 89 5d 08 c9 c3"
    initial = fixed(rsp=0x1000, rbp=0x2000, rax=0x3000, rbx=0x9000)
    initial.write_memory(0x2000, 1)
    initial.write_memory(0x2008, 2)
    result = execute_gadget(gadget(code, 0x4000), initial)
    assert (result.get_register("rbp"), result.get_register("rsp"), result.get_register("rip")) == (0x3000, 0x2010, 0x9000)
    symbolic = solve_gadget(gadget(code, 0x4000), {"rbp": Symbol("rax_0"), "rip": Symbol("rbx_0")})
    assert symbolic.status is SolveStatus.SAT
    effect = execute_gadget(gadget(code, 0x4000), make_symbolic_initial_state())
    prove_equal(effect.get_register("rbp"), Symbol("rax_0"))
    prove_equal(effect.get_register("rip"), Symbol("rbx_0"))


def test_sequence_solver_and_existing_search_carry_written_memory():
    a = gadget("48 89 18 c3", 0x4000)
    b = gadget("48 8b 0a c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_register("rax", 0x2000)
    initial.set_register("rdx", 0x2000)
    initial.set_register("rbx", 42)
    initial.set_register("rcx", 0)
    initial.write_memory(0x2000, 7)
    solved = solve_sequence([a, b], {"rcx": 42}, initial_state=initial)
    assert solved.status is SolveStatus.SAT
    assert isinstance(solved.effect.steps[0].after.memory_state, MemoryStore)
    assert solved.effect.steps[1].before.memory_state is solved.effect.steps[0].after.memory_state
    assert solve_sequence([a, b], {"rcx": 7}, initial_state=initial).status is SolveStatus.UNSAT
    found = search_sequences([a, b], {"rcx": 42}, max_depth=2, initial_state=initial)
    assert found.status is SearchStatus.FOUND and found.sequence == (a, b)
    assert initial.read_memory(0x2000) == 7


def test_unsupported_store_and_push_forms_fail_explicitly():
    for code in ("48 0f a4 d8 01", "66 0f 44 18", "48 0f ac d8 01", "48 0f a5 d8", "67 48 89 1c c8",
                 "48 0f ad d8", "64 48 89 18", "66 50", "ff 30", "66 6a ff"):
        initial = make_symbolic_initial_state()
        before = initial.input_bindings()
        with pytest.raises(UnsupportedInstruction):
            execute(ops(code), initial)
        assert initial.input_bindings() == before and isinstance(initial.memory_state, InitialMemory)


def test_solver_witness_describes_m0_and_export_does_not_drop_versions():
    a = Symbol("A")
    initial = CPUState()
    initial.write_memory(0x1000, 1)
    value = MemoryRead(a, write64(InitialMemory(), 0x1000, 2))
    context = SolverContext(initial)
    context.add_equal(a, 0x1000)
    context.add_equal(value, 2)
    solution = context.solve()
    assert solution.status is SolveStatus.SAT and solution.memory_values[0x1000] == 1
    assert evaluate(value, solution.assignments, memory=solution.memory_values) == 2
    result = initial.copy()
    result.store_memory(0x1000, 2)
    with pytest.raises(ValueError, match="memory versions"):
        create_analysis_manifest(sequence=[gadget("c3", 0x4000)], initial_state=result)
    with pytest.raises(ValueError, match="memory versions"):
        create_analysis_manifest(sequence=[gadget("c3", 0x4000)], goals={"rax": value})
