"""Width extension and RIP-relative integration in the shared state model."""
from dataclasses import replace

import pytest

from minirop.constraints import Equality
from minirop.disasm import Disassembler
from minirop.executor import execute
from minirop.gadget import Operand
from minirop.ir import Address, Extend, Lea
from minirop.semantics import UnsupportedInstruction, lift_instruction
from minirop.solver import SolveStatus, SolverContext, solve_sequence
from minirop.state import CPUState, MemoryAccessError, make_symbolic_initial_state
from minirop.symbolic import Constant, Symbol, WORD_MASK, evaluate, sign_extend, zero_extend
from test_sequence import gadget


def ops(code, address=0x4000):
    instructions = Disassembler().disassemble(bytes.fromhex(code), address)
    assert sum(i.size for i in instructions) == len(bytes.fromhex(code))
    return tuple(lift_instruction(i) for i in instructions)


def prove(value, expected, bindings, initial=None):
    context = SolverContext(initial)
    for name, number in bindings.items():
        context.add_equal(Symbol(name), number)
    context._solver.add(context.translator.translate(value) != context.translator.translate(expected))
    assert context.solve().status is SolveStatus.UNSAT


def check_register(code, inputs, expected, destination="rax"):
    concrete = CPUState()
    symbolic = make_symbolic_initial_state()
    for name, number in inputs.items():
        concrete.set_register(name, number)
    result = execute(ops(code), symbolic).get_register(destination)
    assert execute(ops(code), concrete).get_register(destination) == expected
    bindings = {name + "_0": number for name, number in inputs.items()}
    assert evaluate(result, bindings) == expected
    prove(result, expected, bindings)


def test_movzx_byte_all_destination_widths():
    old = 0x1122334455667788
    for code, expected in [("66 0f b6 c3", 0x11223344556600ff),
                           ("0f b6 c3", 0xff), ("48 0f b6 c3", 0xff)]:
        check_register(code, {"rax": old, "rbx": WORD_MASK}, expected)
    # Reading AL must precede replacing its parent RAX.
    check_register("0f b6 c0", {"rax": WORD_MASK}, 0xff)


def test_movzx_word_all_destination_widths():
    for code in ("0f b7 c3", "48 0f b7 c3"):
        check_register(code, {"rax": WORD_MASK, "rbx": 0xabcdef008001}, 0x8001)


def test_movsx_byte_sign_and_alias_write_boundaries():
    old = 0x1122334455667788
    for source, extended in [(0, 0), (0x7f, 0x7f), (0x80, 0xff80), (0xff, 0xffff)]:
        check_register("66 0f be c3", {"rax": old, "rbx": source}, (old & ~0xffff) | extended)
        value32 = extended if source < 0x80 else 0xffff0000 | extended
        check_register("0f be c3", {"rax": old, "rbx": source}, value32)
        value64 = value32 if source < 0x80 else 0xffffffff00000000 | value32
        check_register("48 0f be c3", {"rax": old, "rbx": source}, value64)
    check_register("48 0f be c0", {"rax": WORD_MASK}, WORD_MASK)
    check_register("0f be c0", {"rax": WORD_MASK}, 0xffffffff)


def test_movsx_word_sign_boundaries():
    for number, expected in [(0x7fff, 0x7fff), (0x8000, 0xffff8000), (0xffff, 0xffffffff)]:
        check_register("0f bf c3", {"rbx": number}, expected)
        wide = expected if number < 0x8000 else 0xffffffff00000000 | expected
        check_register("48 0f bf c3", {"rbx": number}, wide)


def test_movsxd_register_positive_and_negative_dword_boundaries():
    for number, expected in [(0, 0), (0x7fffffff, 0x7fffffff),
                             (0x80000000, 0xffffffff80000000),
                             (0x80000001, 0xffffffff80000001), (0xffffffff, WORD_MASK)]:
        check_register("48 63 c1", {"rcx": 0xabcdef0000000000 | number}, expected)


def test_high_byte_sources_and_effective_rex_legality():
    for code, parent in [("0f b6 c4", "rax"), ("0f b6 c7", "rbx"),
                         ("0f b6 c5", "rcx"), ("0f b6 c6", "rdx")]:
        check_register(code, {parent: 0xff00}, 0xff)
        check_register(code.replace("b6", "be"), {parent: 0xff00}, 0xffffffff)
    check_register("40 0f b6 c4", {"rsp": 0x80}, 0x80)
    check_register("4d 0f be c1", {"r9": 0x80}, 0xffffffffffffff80, "r8")
    instruction = Disassembler().disassemble(bytes.fromhex("40 0f b6 c4"))[0]
    fake = replace(instruction, operands=(instruction.operands[0], Operand("register", 8, register="ah")))
    with pytest.raises(UnsupportedInstruction, match="high-byte"):
        lift_instruction(fake)


def test_extensions_read_narrow_memory_and_existing_sib_addresses():
    forms = [("66 0f b6 03", 0x11223344556600ff), ("0f b6 03", 0xff),
             ("48 0f b6 03", 0xff), ("0f b7 03", 0x80ff), ("48 0f b7 03", 0x80ff),
             ("66 0f be 03", 0x112233445566ffff), ("0f be 03", 0xffffffff),
             ("48 0f be 03", WORD_MASK), ("0f bf 03", 0xffff80ff),
             ("48 0f bf 03", 0xffffffffffff80ff), ("48 63 03", 0xffffffff800080ff)]
    for code, expected in forms:
        state = CPUState()
        state.set_register("rax", 0x1122334455667788)
        state.set_register("rbx", 0x1000)
        state.write_memory(0x1000, 0x12345678800080ff)
        assert execute(ops(code), state).get_register("rax") == expected
        symbolic = make_symbolic_initial_state()
        symbolic.write_memory(0x1000, Symbol("word"))
        value = execute(ops(code), symbolic).get_register("rax")
        bindings = {"rax_0": 0x1122334455667788, "rbx_0": 0x1000, "word": 0x12345678800080ff}
        assert evaluate(value, bindings, memory={0x1000: Symbol("word")}) == expected
        prove(value, expected, bindings, symbolic)
    state.set_register("rcx", 2)
    # [RBX+RCX*4-8] equals RBX, using the same address calculation.
    assert execute(ops("48 0f be 44 8b f8"), state).get_register("rax") == WORD_MASK
    empty = CPUState()
    empty.set_register("rbx", 0x1000)
    with pytest.raises(MemoryAccessError):
        execute(ops("0f b6 03"), empty)


def test_extension_value_helpers_truncate_and_match_smt():
    for source_width, target_width in [(8, 16), (8, 32), (8, 64), (16, 32), (16, 64), (32, 64)]:
        mask = (1 << source_width) - 1
        for number in [0, (1 << (source_width - 1)) - 1, 1 << (source_width - 1), mask]:
            raw = (WORD_MASK ^ mask) | number
            negative = number >= 1 << (source_width - 1)
            signed = number if not negative else ((1 << target_width) - (1 << source_width) + number)
            for helper, expected in [(zero_extend, number), (sign_extend, signed)]:
                assert helper(raw, source_width, target_width) == expected
                assert helper(Constant(raw), source_width, target_width) == Constant(expected)
                expression = helper(Symbol("x"), source_width, target_width)
                assert evaluate(expression, {"x": raw}) == expected
                prove(expression, expected, {"x": raw})
                assert "extend" in str(expression)
    for source_width, target_width in [(8, 8), (32, 16), (24, 64)]:
        with pytest.raises(ValueError):
            sign_extend(Symbol("x"), source_width, target_width)


def test_invalid_extension_widths_immediates_and_noncanonical_movsxd_rejected():
    for code in ["66 0f b7 c3", "66 0f bf c3", "63 c1", "66 63 c1", "67 48 0f be 03", "64 48 63 03"]:
        with pytest.raises(UnsupportedInstruction):
            ops(code)
    instruction = Disassembler().disassemble(bytes.fromhex("0f b6 c3"))[0]
    for source in [Operand("immediate", 8, immediate=1), Operand("register", 32, register="ebx")]:
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(replace(instruction, operands=(instruction.operands[0], source)))
    for operation in [Extend("eax", "ebx", 32, False), Extend("ax", "bx", 16, True),
                      Extend("rax", "bl", 16, True), Extend("rax", 1, 8, False)]:
        with pytest.raises((ValueError, TypeError)):
            execute([operation], make_symbolic_initial_state())


def test_rip_loads_use_next_address_signed_displacements_and_each_length():
    base = 0x400100
    for code, target, expected in [("48 8b 05 20 00 00 00", 0x400127, 0x8877665544332211),
                                    ("8b 05 f0 ff ff ff", 0x4000f6, 0x44332211),
                                    ("8a 05 20 00 00 00", 0x400126, 0xaabbccddeeff0011),
                                    ("66 8b 05 f0 ff ff ff", 0x4000f7, 0xaabbccddeeff2211)]:
        state = CPUState()
        state.set_register("rax", 0xaabbccddeeff0099)
        state.set_register("rip", 0xdeadbeef)
        state.write_memory(target, 0x8877665544332211)
        result = execute(ops(code, base), state)
        assert result.get_register("rax") == expected
        assert result.get_register("rip") == 0xdeadbeef


def test_each_rip_instruction_captures_its_own_next_address():
    state = CPUState()
    state.write_memory(0x400f, 0x0807060504030201)
    state.write_memory(0x4017, 0x11100f0e0d0c0b0a)
    operations = ops("48 8b 05 08 00 00 00 48 8b 0d 08 00 00 00")
    assert [op.address.next_rip for op in operations] == [0x4007, 0x400e]
    result = execute(operations, state)
    assert result.get_register("rax") == 0x0807060504030201
    assert result.get_register("rcx") == 0x100f0e0d0c0b0a08


def test_rip_lea_never_reads_memory_and_preserves_alias_rules():
    base = 0x1234567880000000
    old = 0xaabbccddeeff0011
    for code, expected in [("48 8d 05 20 00 00 00", base + 7 + 0x20),
                           ("8d 05 20 00 00 00", 0x80000026),
                           ("66 8d 05 20 00 00 00", 0xaabbccddeeff0027)]:
        state = CPUState()
        state.set_register("rax", old)
        assert execute(ops(code, base), state).get_register("rax") == expected
        value = execute(ops(code, base), make_symbolic_initial_state()).get_register("rax")
        assert evaluate(value, {"rax_0": old}) == expected
        prove(value, expected, {"rax_0": old})


def test_rip_addresses_wrap_at_64_bits_and_signed_disp32_extremes():
    for base, displacement in [(WORD_MASK - 3, 0x20), (0, -0x20),
                                (0x4000, -(1 << 31)), (0x4000, (1 << 31) - 1)]:
        encoded = (displacement & 0xffffffff).to_bytes(4, "little").hex()
        answer = (base + 7 + displacement) & WORD_MASK
        assert execute(ops("48 8d 05 " + encoded, base), CPUState()).get_register("rax") == answer
    state = CPUState()
    state.write_memory(WORD_MASK, 0x8877665544332211)
    # next_rip wraps to 3; disp=-4 selects the final address and reads across 0.
    assert execute(ops("48 8b 05 fc ff ff ff", WORD_MASK - 3), state).get_register("rax") == 0x8877665544332211


def test_rip_stores_share_memory_and_count_trailing_immediates():
    old = 0x1122334455667788
    for code, width, length in [("88 05 20 00 00 00", 8, 6), ("66 89 05 20 00 00 00", 16, 7),
                                ("89 05 20 00 00 00", 32, 6), ("48 89 05 20 00 00 00", 64, 7),
                                ("c6 05 20 00 00 00 ff", 8, 7),
                                ("48 c7 05 20 00 00 00 ff ff ff ff", 64, 11)]:
        target = 0x4000 + length + 0x20
        state = CPUState()
        state.set_register("rax", 0xaabbccddeeff0099)
        state.set_register("rbx", target)
        state.write_memory(target, old)
        stored = execute(ops(code), state)
        source = WORD_MASK if "c7" in code else 0xff if "c6" in code else 0xaabbccddeeff0099
        mask = (1 << width) - 1
        expected = (old & (WORD_MASK ^ mask)) | (source & mask)
        # GPR-based read must observe the RIP-relative write, including overlap.
        assert execute(ops("48 8b 13"), stored).get_register("rdx") == expected
        assert state.read_memory(target) == old


def test_rip_memory_sources_for_all_three_extension_instructions():
    for code, expected in [("0f b6 05 20 00 00 00", 0xff), ("48 0f b6 05 20 00 00 00", 0xff),
                           ("48 0f b7 05 20 00 00 00", 0x80ff), ("0f be 05 20 00 00 00", 0xffffffff),
                           ("48 0f be 05 20 00 00 00", WORD_MASK),
                           ("48 0f bf 05 20 00 00 00", 0xffffffffffff80ff),
                           ("48 63 05 20 00 00 00", 0xffffffff800080ff)]:
        target = 0x4000 + len(bytes.fromhex(code)) + 0x20
        state = CPUState()
        state.write_memory(target, 0x12345678800080ff)
        assert execute(ops(code), state).get_register("rax") == expected
        symbolic = make_symbolic_initial_state()
        symbolic.write_memory(target, Symbol("input_word"))
        value = execute(ops(code), symbolic).get_register("rax")
        assert evaluate(value, {"input_word": 0x12345678800080ff}, memory={target: 0x12345678800080ff}) == expected
        prove(value, expected, {"input_word": 0x12345678800080ff}, symbolic)


def test_missing_or_invalid_rip_metadata_is_explicitly_rejected():
    instruction = Disassembler().disassemble(bytes.fromhex("48 8b 05 00 00 00 00"), 0x4000)[0]
    for changes in [{"address": None}, {"address": -1}, {"address": WORD_MASK + 1}, {"address": False},
                    {"size": None}, {"size": 0}, {"size": 6}, {"size": True}]:
        with pytest.raises(UnsupportedInstruction, match="metadata"):
            lift_instruction(replace(instruction, **changes))
    for operand in [replace(instruction.operands[1], displacement=0xffffffff),
                    replace(instruction.operands[1], index="rax"),
                    replace(instruction.operands[1], scale=2)]:
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(replace(instruction, operands=(instruction.operands[0], operand)))
    for address in [Address("rip"), Address("rip", next_rip=False),
                    Address("rip", next_rip=WORD_MASK + 1), Address("rip", index="rax", next_rip=0x4007),
                    Address("rax", next_rip=0x4007)]:
        with pytest.raises(ValueError):
            execute([Lea("rax", address)], make_symbolic_initial_state())


def test_mixed_extension_and_rip_gadget_concrete_symbolic_and_solver_agree():
    # LEA RBX,[RIP+100]; MOV [RBX],ECX; MOVSX RAX,byte [RIP+F6];
    # MOVZX EDX,word [RBX]; MOVSXD RSI,ECX; RET.
    code = "48 8d 1d 00 01 00 00 89 0b 48 0f be 05 f6 00 00 00 0f b7 13 48 63 f1 c3"
    expected = {"rax": 0xffffffffffffff81, "rdx": 0xff81, "rsi": 0xffffffff8000ff81, "rip": 0x9000}
    initial = make_symbolic_initial_state()
    binding = Equality(Symbol("rcx_0"), 0x8000ff81)
    solved = solve_sequence([gadget(code, 0x4000)], expected, initial_state=initial, constraints=[binding])
    assert solved.status is SolveStatus.SAT
    state = CPUState()
    state.set_register("rcx", 0x8000ff81)
    state.set_register("rsp", 0x2000)
    state.write_memory(0x2000, 0x9000)
    concrete = execute(ops(code), state)
    for name, number in expected.items():
        assert concrete.get_register(name) == number
        if name != "rip":
            assert evaluate(solved.effect.final_state.get_register(name), {"rcx_0": 0x8000ff81}) == number
    impossible = solve_sequence([gadget(code, 0x4000)], {"rax": 0x81}, initial_state=initial, constraints=[binding])
    assert impossible.status is SolveStatus.UNSAT


def test_rip_and_symbolic_gpr_aliases_observe_partial_overwrites():
    # A symbolic qword write, a RIP-relative byte overwrite, then a GPR read.
    code = "48 89 18 c6 05 20 00 00 00 aa 48 8b 08"
    target = 0x4000 + 3 + 7 + 0x20
    symbolic = make_symbolic_initial_state()
    value = execute(ops(code), symbolic).get_register("rcx")
    payload = 0x1122334455667788
    for base, expected in [(target - 2, 0x1122334455aa7788), (target + 8, payload)]:
        bindings = {"rax_0": base, "rbx_0": payload}
        assert evaluate(value, bindings) == expected
        prove(value, expected, bindings)
        state = CPUState()
        state.set_register("rax", base)
        state.set_register("rbx", payload)
        assert execute(ops(code), state).get_register("rcx") == expected
