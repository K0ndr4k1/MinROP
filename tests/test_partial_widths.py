"""Physical register views, access widths and decoded SIB instruction semantics."""
from dataclasses import replace

import pytest

from minirop.disasm import Disassembler
from minirop.executor import execute
from minirop.gadget import Operand
from minirop.ir import Mov
from minirop.semantics import UnsupportedInstruction, lift_instruction
from minirop.solver import SolveStatus, SolverContext, solve_sequence
from minirop.state import CPUState, REGISTER_VIEWS, make_symbolic_initial_state
from minirop.symbolic import (Constant, MemoryRead, Symbol, WORD_MASK,
                              evaluate)
from test_sequence import gadget


def ops(code):
    return tuple(lift_instruction(i) for i in Disassembler().disassemble(bytes.fromhex(code)))


def prove_value(value, expected, *, bindings=None, initial=None):
    context = SolverContext(initial)
    for name, number in (bindings or {}).items():
        context.add_equal(Symbol(name), number)
    context._solver.add(context.translator.translate(value) != context.translator.translate(expected))
    assert context.solve().status is SolveStatus.UNSAT


def test_all_dword_aliases_zero_upper_bits_and_read_current_parent():
    for alias, (parent, width, _) in REGISTER_VIEWS.items():
        if width != 32:
            continue
        initial = CPUState()
        initial.set_register(parent, WORD_MASK)
        assert initial.get_register(alias) == 0xffffffff
        initial.set_register(alias.upper(), 0x123456789)
        assert initial.get_register(parent) == 0x23456789
        assert dict(initial.register_bindings()) == {parent: 0x23456789}
        symbolic = make_symbolic_initial_state()
        symbolic.set_register(alias, Symbol("input"))
        value = symbolic.get_register(parent)
        prove_value(value, 0x23456789, bindings={"input": 0x123456789})
        assert evaluate(value, {"input": 0x123456789}) == 0x23456789


def test_all_word_and_byte_views_preserve_other_bits():
    old = 0x1122334455667788
    for alias, (parent, width, offset) in REGISTER_VIEWS.items():
        if width not in (8, 16):
            continue
        state = CPUState()
        state.set_register(parent, old)
        assert state.get_register(alias) == (old >> offset) & ((1 << width) - 1)
        state.set_register(alias, 0xbb if width == 8 else 0xaabb)
        mask = ((1 << width) - 1) << offset
        expected = (old & (WORD_MASK ^ mask)) | ((0xbb if width == 8 else 0xaabb) << offset)
        assert state.get_register(parent) == expected
        symbolic = make_symbolic_initial_state()
        symbolic.set_register(alias, 0xbb if width == 8 else 0xaabb)
        value = symbolic.get_register(parent)
        assert evaluate(value, {parent + "_0": old}) == expected
        prove_value(value, expected, bindings={parent + "_0": old})
    extended = execute(ops("41 b8 dd cc bb aa 66 41 b8 34 12 41 b0 88"), make_symbolic_initial_state())
    prove_value(extended.get_register("r8"), 0xaabb1288)
    for code, expected in [("66 89 d8", 0x11223344556600ff), ("88 d8", 0x11223344556677ff)]:
        state = CPUState()
        state.set_register("rax", old)
        state.set_register("rbx", 0xff)
        assert execute(ops(code), state).get_register("rax") == expected


def test_high_byte_encoding_and_rex_low_byte_registers():
    state = CPUState()
    state.set_register("rax", 0x1122334455667788)
    state.set_register("rsp", 0x1000)
    result = execute(ops("b4 bb 40 b4 aa 86 e0"), state)
    assert result.get_register("rax") == 0x11223344556688bb
    assert result.get_register("rsp") == 0x10aa
    instruction = Disassembler().disassemble(bytes.fromhex("40 b4 aa"))[0]
    forged = replace(instruction, operands=(Operand("register", 8, register="ah"), instruction.operands[1]))
    with pytest.raises(UnsupportedInstruction, match="REX"):
        lift_instruction(forged)
    instruction = Disassembler().disassemble(bytes.fromhex("b4 aa"))[0]
    forged = replace(instruction, operands=(Operand("register", 8, register="spl"), instruction.operands[1]))
    with pytest.raises(UnsupportedInstruction, match="REX"):
        lift_instruction(forged)


@pytest.mark.parametrize("code,expected", [
    ("89 d8", lambda a, b: {"rax": b & 0xffffffff}),
    ("01 d8", lambda a, b: {"rax": (a + b) & 0xffffffff}),
    ("66 29 d8", lambda a, b: {"rax": (a & ~0xffff) | ((a - b) & 0xffff)}),
    ("31 c0", lambda a, b: {"rax": 0}),
    ("20 d8", lambda a, b: {"rax": (a & ~255) | ((a & b) & 255)}),
    ("09 d8", lambda a, b: {"rax": (a | b) & 0xffffffff}),
    ("ff c0", lambda a, b: {"rax": (a + 1) & 0xffffffff}),
    ("66 ff c8", lambda a, b: {"rax": (a & ~0xffff) | ((a - 1) & 0xffff)}),
    ("f6 db", lambda a, b: {"rbx": (b & ~255) | ((-b) & 255)}),
    ("f7 d0", lambda a, b: {"rax": (~a) & 0xffffffff}),
    ("93", lambda a, b: {"rax": b & 0xffffffff, "rbx": a & 0xffffffff}),
    ("80 c0 01", lambda a, b: {"rax": (a & ~255) | ((a + 1) & 255)}),
])
def test_narrow_register_operations_match_concrete_symbolic_and_solver(code, expected):
    for a, b in [(0x11223344ffffffff, 1), (0x1122334455667788, 0xffeeddccbbaa00ff)]:
        initial = CPUState()
        initial.set_register("rax", a)
        initial.set_register("rbx", b)
        result = execute(ops(code), initial)
        symbolic = execute(ops(code), make_symbolic_initial_state())
        bindings = {"rax_0": a, "rbx_0": b}
        expected_registers = {"rax": a, "rbx": b, **expected(a, b)}
        for register, answer in expected_registers.items():
            assert result.get_register(register) == answer
            assert evaluate(symbolic.get_register(register), bindings) == answer
            prove_value(symbolic.get_register(register), answer, bindings=bindings)
        assert initial.get_register("rax") == a and initial.get_register("rbx") == b


@pytest.mark.parametrize("code,width", [("8a 03", 8), ("66 8b 03", 16), ("8b 03", 32)])
def test_narrow_loads_read_only_required_bytes_and_write_aliases(code, width):
    old, word = 0xffeeddccbbaa0099, 0x1122334455667788
    initial = CPUState()
    initial.set_register("rax", old)
    initial.set_register("rbx", 0x1000)
    initial.write_memory(0x1000, word)
    answer = word & ((1 << width) - 1)
    if width < 32:
        answer |= old & ~((1 << width) - 1)
    result = execute(ops(code), initial)
    assert result.get_register("rax") == answer
    alias_base = CPUState()
    alias_base.set_register("rax", 0x1000)
    alias_base.write_memory(0x1000, word)
    alias_answer = word & ((1 << width) - 1)
    if width < 32:
        alias_answer |= 0x1000 & ~((1 << width) - 1)
    assert execute(ops(code.replace("03", "00")), alias_base).get_register("rax") == alias_answer
    symbolic = make_symbolic_initial_state()
    symbolic.write_memory(0x1000, word)
    value = execute(ops(code), symbolic).get_register("rax")
    prove_value(value, answer, initial=symbolic, bindings={"rax_0": old, "rbx_0": 0x1000})
    assert evaluate(value, {"rax_0": old, "rbx_0": 0x1000}, memory={0x1000: word}) == answer
    minimal = CPUState()
    minimal.set_register("rax", old)
    minimal.set_register("rbx", 0x2000)
    minimal.store_memory(0x2000, word, width)
    assert execute(ops(code), minimal).get_register("rax") == answer


def test_multiwidth_register_and_immediate_stores():
    for width, code in [(8, "88 18"), (16, "66 89 18"), (32, "89 18"), (64, "48 89 18")]:
        initial = CPUState()
        initial.set_register("rax", 0x1000)
        initial.set_register("rbx", 0x1122334455667788)
        initial.write_memory(0x1000, WORD_MASK)
        result = execute(ops(code), initial)
        mask = (1 << width) - 1
        answer = (WORD_MASK ^ mask) | (0x1122334455667788 & mask)
        assert result.read_memory(0x1000) == answer
        symbolic = make_symbolic_initial_state()
        symbolic.write_memory(0x1000, WORD_MASK)
        value = execute(ops(code), symbolic).read_memory(Symbol("rax_0"))
        prove_value(value, answer, initial=symbolic,
                    bindings={"rax_0": 0x1000, "rbx_0": 0x1122334455667788})
    for width, code in [(8, "c6 00 01"), (16, "66 c7 00 01 00"),
                        (32, "c7 00 01 00 00 00"), (64, "48 c7 00 01 00 00 00")]:
        initial = CPUState()
        initial.set_register("rax", 0x1000)
        result = execute(ops(code), initial)
        assert result.read_memory(0x1000, width) == 1


def test_mixed_width_overlap_and_symbolic_aliases_share_one_array():
    code = "48 89 18 c6 40 01 ff 48 8b 08"
    symbolic = make_symbolic_initial_state()
    result = execute(ops(code), symbolic)
    value = result.get_register("rcx")
    answer = 0x112233445566ff88
    prove_value(value, answer, bindings={"rbx_0": 0x1122334455667788})
    # A byte write through another symbolic address can alias any byte.
    symbolic.store_memory(Symbol("A"), 0x1122334455667788)
    symbolic.store_memory(Symbol("B"), 0xaa, 8)
    read = symbolic.read_memory(Symbol("A"))
    prove_value(read, 0x1122334455aa7788, bindings={"A": 0x1000, "B": 0x1002})
    prove_value(read, 0x1122334455667788, bindings={"A": 0x1000, "B": 0x2000})
    wrapped = CPUState()
    wrapped.store_memory(WORD_MASK, 0xaabb, 16)
    assert wrapped.read_memory(0, 8) == 0xaa


def test_shared_effective_address_handles_all_common_sib_forms():
    for lea_code, address in [
        ("48 8d 03", 0x1000), ("48 8d 43 08", 0x1008),
        ("48 8d 04 0b", 0x1003), ("48 8d 04 4b", 0x1006),
        ("48 8d 44 8b 08", 0x1014), ("48 8d 44 cb f0", 0x1008),
        ("48 8d 04 cd 00 10 00 00", 0x1018),
    ]:
        initial = CPUState()
        initial.set_register("rbx", 0x1000)
        initial.set_register("rcx", 3)
        initial.write_memory(address, 77)
        lea = execute(ops(lea_code), initial)
        assert lea.get_register("rax") == address
        mov_code = lea_code.replace("8d", "8b", 1)
        load = execute(ops(mov_code), initial)
        assert load.get_register("rax") == 77
        store_code = mov_code.replace("8b", "89", 1)
        initial.set_register("rax", 91)
        stored = execute(ops(store_code), initial)
        assert stored.read_memory(address) == 91
        symbolic = make_symbolic_initial_state()
        symbolic.write_memory(address, 77)
        load = execute(ops(mov_code), symbolic)
        prove_value(load.get_register("rax"), 77, initial=symbolic,
                    bindings={"rbx_0": 0x1000, "rcx_0": 3})
    initial = CPUState()
    initial.set_register("rbx", WORD_MASK - 2)
    initial.set_register("rcx", 1)
    initial.set_register("rax", 19)
    result = execute(ops("48 89 44 cb 08"), initial)
    assert result.read_memory(13) == 19


def test_mixed_width_instruction_sequence_and_solver_goals():
    # MOV EAX,EBX; ADD AL,1; MOV [RDI+RCX*4+8],AL; MOV DX,[same]; RET.
    code = "89 d8 80 c0 01 88 44 8f 08 66 8b 54 8f 08 c3"
    initial = make_symbolic_initial_state()
    initial.set_register("rbx", 0xabcdef00123456ff)
    initial.set_register("rdi", 0x2000)
    initial.set_register("rcx", 2)
    initial.set_register("rdx", 0xffeeddccbbaa9988)
    initial.write_memory(0x2010, 0x2211)
    solved = solve_sequence([gadget(code, 0x4000)], {"rax": 0x12345600, "dx": 0x2200}, initial_state=initial)
    assert solved.status is SolveStatus.SAT
    prove_value(solved.effect.final_state.get_register("rdx"), 0xffeeddccbbaa2200, initial=initial)
    concrete = CPUState()
    for name, value in initial.register_bindings():
        if type(value) is int:
            concrete.set_register(name, value)
    concrete.set_register("rsp", 0x1000)
    concrete.write_memory(0x1000, 0x9000)
    concrete.write_memory(0x2010, 0x2211)
    effect = execute(ops(code), concrete)
    assert effect.get_register("eax") == 0x12345600 and effect.get_register("dx") == 0x2200


def test_illegal_forms_and_metadata_are_rejected():
    for code in ["67 8b 03", "64 8b 03", "48 0f a4 d8 01", "48 0f ac d8 01", "48 0f a5 d8",
                 "48 0f ad d8", "f0 01 18", "66 50", "f3 89 d8"]:
        with pytest.raises(UnsupportedInstruction):
            ops(code)
    assert Disassembler().disassemble(bytes.fromhex("f0 01 d8")) == ()
    instruction = Disassembler().disassemble(bytes.fromhex("89 d8"))[0]
    with pytest.raises(UnsupportedInstruction):
        lift_instruction(replace(instruction, operands=(instruction.operands[0], Operand("register", 16, register="bx"))))
    with pytest.raises(ValueError, match="width"):
        execute([Mov("eax", "bx")], make_symbolic_initial_state())
    with pytest.raises(ValueError, match="width"):
        CPUState().store_memory(0x1000, 1, 24)


def test_narrow_immediate_extension_and_truncation():
    initial = CPUState()
    initial.set_register("rax", 0x1122334455667788)
    assert execute(ops("83 c0 ff"), initial).get_register("rax") == 0x55667787
    assert execute(ops("66 83 c0 ff"), initial).get_register("rax") == 0x1122334455667787
    assert execute(ops("b0 ff"), initial).get_register("rax") == 0x11223344556677ff
    for code, width in [("c6 00 ff", 8), ("66 c7 00 ff ff", 16),
                        ("c7 00 ff ff ff ff", 32), ("48 c7 00 ff ff ff ff", 64)]:
        initial.set_register("rax", 0x1000)
        assert execute(ops(code), initial).read_memory(0x1000, width) == (1 << width) - 1


def test_narrow_shifts_use_operand_width_and_x86_count_mask():
    old = 0x1122334480000081
    for code, expected in [("c0 e0 08", old & ~255),
                           ("66 c1 e8 10", old & ~0xffff),
                           ("c1 e8 1f", 1), ("c1 e0 20", old & 0xffffffff),
                           ("c0 e8 20", old)]:
        state = CPUState()
        state.set_register("rax", old)
        assert execute(ops(code), state).get_register("rax") == expected
        result = execute(ops(code), make_symbolic_initial_state()).get_register("rax")
        assert evaluate(result, {"rax_0": old}) == expected
        prove_value(result, expected, bindings={"rax_0": old})
