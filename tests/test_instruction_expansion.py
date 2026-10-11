"""Decoded instruction behavior shared by concrete execution and the solver."""
import pytest

from minirop.disasm import Disassembler
from minirop.executor import execute, execute_gadget
from minirop.ir import Lea, Pop
from minirop.search import SearchStatus, search_sequences
from minirop.semantics import UnsupportedInstruction, lift_instruction
from minirop.solver import SolverContext, SolveStatus, solve_sequence
from minirop.state import CPUState, make_symbolic_initial_state
from minirop.symbolic import MemoryRead, Symbol, WORD_MASK, evaluate, format_value
from test_sequence import gadget


def concrete(**registers):
    state = CPUState()
    state.set_register("rsp", 0x1000)
    state.write_memory(0x1000, 0x9000)
    for name, value in registers.items():
        state.set_register(name, value)
    return state


@pytest.mark.parametrize("code,expected", [
    ("48 01 d8", lambda a, b: a + b),
    ("48 29 d8", lambda a, b: a - b),
    ("48 31 d8", lambda a, b: a ^ b),
    ("48 35 ff 00 00 00", lambda a, b: a ^ 255),
    ("48 21 d8", lambda a, b: a & b),
    ("48 83 e0 f0", lambda a, b: a & -16),
    ("48 09 d8", lambda a, b: a | b),
    ("48 83 c8 80", lambda a, b: a | -128),
    ("48 ff c0", lambda a, b: a + 1),
    ("48 ff c8", lambda a, b: a - 1),
    ("48 f7 d8", lambda a, b: -a),
    ("48 f7 d0", lambda a, b: ~a),
    ("48 c1 e0 04", lambda a, b: a << 4),
    ("48 c1 e8 08", lambda a, b: a >> 8),
])
def test_register_effects_concrete_symbolic_and_solver(code, expected):
    part = gadget(code + " c3", 0x4000)
    initial = make_symbolic_initial_state()
    expression = execute_gadget(part, initial).get_register("rax")
    assert format_value(expression)
    for a, b in [(0, 1), (WORD_MASK, 1), (1 << 63, 0x123), (37, 9)]:
        answer = expected(a, b) & WORD_MASK
        result = execute_gadget(part, concrete(rax=a, rbx=b))
        assert result.get_register("rax") == answer
        assert result.get_register("rbx") == b
        assert result.get_register("rsp") == 0x1008
        assert result.get_register("rip") == 0x9000
        assert evaluate(expression, {"rax_0": a, "rbx_0": b}) == answer
        context = SolverContext()
        context.add_equal(Symbol("rax_0"), a)
        context.add_equal(Symbol("rbx_0"), b)
        context.add_equal(expression, answer)
        assert context.solve().status is SolveStatus.SAT
        context.add_equal(expression, answer ^ 1)
        assert context.solve().status is SolveStatus.UNSAT
    assert initial.get_register("rax") == Symbol("rax_0")


@pytest.mark.parametrize("code,answer", [
    ("48 8d 03", 40),  # [rbx]
    ("48 8d 43 f8", 32),  # [rbx-8]
    ("48 8d 44 8b 08", 60),  # [rbx+rcx*4+8]
])
def test_lea_is_arithmetic_without_memory_access(code, answer):
    op = lift_instruction(Disassembler().disassemble(bytes.fromhex(code))[0])
    assert isinstance(op, Lea)
    state = CPUState()
    state.set_register("rbx", 40)
    state.set_register("rcx", 3)
    assert execute([op], state).get_register("rax") == answer
    symbolic = execute([op], make_symbolic_initial_state()).get_register("rax")
    assert evaluate(symbolic, {"rbx_0": 40, "rcx_0": 3}) == answer
    assert "mem64" not in format_value(symbolic)


def test_lea_scales_aliases_and_address_wrap():
    for sib, scale in [(0x0b, 1), (0x4b, 2), (0x8b, 4), (0xcb, 8)]:
        # LEA RBX,[RBX+RCX*scale+8], with output aliasing the base.
        code = bytes([0x48, 0x8d, 0x5c, sib, 8])
        op = lift_instruction(Disassembler().disassemble(code)[0])
        state = concrete(rbx=WORD_MASK, rcx=3)
        assert execute([op], state).get_register("rbx") == 3 * scale + 7
    for code in ("48 8d 1c 0b", "48 8d 4c 0b 08"):
        op = lift_instruction(Disassembler().disassemble(bytes.fromhex(code))[0])
        result = execute([op], concrete(rbx=40, rcx=3))
        assert result.get_register(op.dst) == (43 if op.dst == "rbx" else 51)


def test_xchg_reads_both_old_values_and_self_exchange():
    part = gadget("48 93 48 83 e8 08 c3", 0x4000)
    result = execute_gadget(part, concrete(rax=7, rbx=30))
    assert (result.get_register("rax"), result.get_register("rbx")) == (22, 7)
    result = execute_gadget(part, make_symbolic_initial_state())
    assert evaluate(result.get_register("rax"), {"rbx_0": 30}) == 22
    assert result.get_register("rbx") == Symbol("rax_0")
    op = lift_instruction(Disassembler().disassemble(bytes.fromhex("48 87 c0"))[0])
    assert execute([op], concrete(rax=7)).get_register("rax") == 7


def test_xchg_rsp_and_ret_use_new_stack():
    state = concrete(rax=0x2000)
    state.write_memory(0x2000, 0x7777)
    result = execute_gadget(gadget("48 94 c3", 0x4000), state)
    assert result.get_register("rax") == 0x1000
    assert result.get_register("rsp") == 0x2008
    assert result.get_register("rip") == 0x7777
    assert state.get_register("rsp") == 0x1000


@pytest.mark.parametrize("code,offset", [("48 8b 03", 0), ("48 8b 5b f8", -8)])
def test_load_old_base_concrete_and_symbolic(code, offset):
    op = lift_instruction(Disassembler().disassemble(bytes.fromhex(code))[0])
    initial = concrete(rbx=0x2008)
    initial.write_memory(0x2008 + offset, 123)
    assert execute([op], initial).get_register(op.dst) == 123
    symbolic = make_symbolic_initial_state()
    symbolic.write_memory(0x2008 + offset, 123)
    value = execute([op], symbolic).get_register(op.dst)
    assert isinstance(value, MemoryRead)
    assert evaluate(value, {"rbx_0": 0x2008}, memory={0x2008 + offset: 123}) == 123
    context = SolverContext(symbolic)
    context.add_equal(Symbol("rbx_0"), 0x2008)
    context.add_equal(value, 124)
    assert context.solve().status is SolveStatus.UNSAT


def test_shift_count_mask_and_unsigned_right_shift():
    for opcode, expected in [(0xe0, lambda n: ((1 << 63) | 3) << n),
                             (0xe8, lambda n: ((1 << 63) | 3) >> n)]:
        for count in (0, 1, 63, 64, 65, 255):
            ins = Disassembler().disassemble(bytes([0x48, 0xc1, opcode, count]))[0]
            op = lift_instruction(ins)
            answer = expected(count & 63) & WORD_MASK
            assert execute([op], concrete(rax=(1 << 63) | 3)).get_register("rax") == answer
            value = execute([op], make_symbolic_initial_state()).get_register("rax")
            context = SolverContext()
            context.add_equal(Symbol("rax_0"), (1 << 63) | 3)
            context.add_equal(value, answer ^ 1)
            assert context.solve().status is SolveStatus.UNSAT


def test_pop_arithmetic_and_zero_idiom_compose():
    part = gadget("58 48 01 d8 48 31 c9 c3", 0x4000)
    state = make_symbolic_initial_state()
    result = execute_gadget(part, state)
    assert evaluate(result.get_register("rax"), {"stack_0": 7, "rbx_0": 9}) == 16
    assert result.get_register("rcx") == 0
    assert result.get_register("rip") == Symbol("stack_1")
    assert evaluate(result.get_register("rsp"), {"rsp_0": 0x1000}) == 0x1010
    assert isinstance(lift_instruction(part.instructions[0]), Pop)


def test_multiple_pops_load_and_leave_compose():
    part = gadget("58 5b 48 01 d8 c9 48 8b 04 24 c3", 0x4000)
    state = concrete(rbp=0x2000)
    state.write_memory(0x1000, 3)
    state.write_memory(0x1008, 5)
    state.write_memory(0x2000, 0x3000)
    state.write_memory(0x2008, 0x7000)
    result = execute_gadget(part, state)
    assert result.get_register("rbx") == 5
    assert result.get_register("rbp") == 0x3000
    assert result.get_register("rax") == result.get_register("rip") == 0x7000
    assert result.get_register("rsp") == 0x2010
    pops = execute_gadget(gadget("58 5b c3", 0x4100), make_symbolic_initial_state())
    assert pops.get_register("rax") == Symbol("stack_0")
    assert pops.get_register("rbx") == Symbol("stack_1")
    assert evaluate(pops.get_register("rsp"), {"rsp_0": 0x1000}) == 0x1018


def test_sequence_solver_and_bounded_search_use_new_semantics():
    a = gadget("48 8d 44 8b 08 48 31 d2 c3", 0x4000)
    b = gadget("48 21 d8 48 c1 e0 01 c3", 0x4100)
    initial = make_symbolic_initial_state()
    for name, value in {"rax": 0, "rbx": 15, "rcx": 2, "rdx": 8}.items():
        initial.set_register(name, value)
    goals = {"rax": 30, "rdx": 0}
    solved = solve_sequence([a, b], goals, initial_state=initial)
    assert solved.status is SolveStatus.SAT and solved.goals_satisfied
    found = search_sequences([a, b], goals, max_depth=2, initial_state=initial)
    assert found.status is SearchStatus.FOUND and found.sequence == (a, b)
    assert solve_sequence([a, b], {"rax": 31}, initial_state=initial).status is SolveStatus.UNSAT


def test_unsupported_instructions_fail_explicitly():
    for code in ["48 0f a4 d8 01", "48 0f ac d8 01", "48 0f a5 d8", "48 0f ad d8",
                 "0f 31", "0f 01 f9", "0f c7 f0", "f3 0f bc c3", "f3 0f bd c3",
                 "48 0f c7 08", "0f c7 08", "0f a0", "f3 a4",
                 "eb 00", "e8 00 00 00 00", "0f 05", "0f 57 c0",
                 "c5 f8 57 c0", "d9 e8", "f0 48 0f ab 03", "0f 01 d0",
                 "66 50", "66 6a 01", "0f 01 d1", "67 48 8b 04 cb", "0f 32",
                 "0f 30", "f4", "fa"]:
        part = gadget(code + " c3", 0x4000)
        state = make_symbolic_initial_state()
        before = state.input_bindings()
        with pytest.raises(UnsupportedInstruction):
            execute_gadget(part, state)
        assert state.input_bindings() == before


def test_old_manifest_rejects_unrepresentable_address_metadata():
    from minirop.manifest import create_analysis_manifest

    for code in ("48 8b 04 cb c3", "48 8b 05 00 00 00 00 c3"):
        with pytest.raises(ValueError, match="effective-address metadata"):
            create_analysis_manifest(sequence=[gadget(code, 0x4000)])
