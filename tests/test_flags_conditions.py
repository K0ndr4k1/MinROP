"""Core flag production and conditional data operations, with no branches."""
from dataclasses import replace
from itertools import product

import pytest

from minirop.constraints import Equality
from minirop.disasm import Disassembler
from minirop.executor import execute
from minirop.flags import (FLAG_NAMES, UNDEFINED, UndefinedFlagError, UninitializedFlag,
                           evaluate_condition)
from minirop.gadget import Operand
from minirop.ir import Cmp, Cmovcc, Setcc
from minirop.semantics import UnsupportedInstruction, lift_instruction
from minirop.solver import SolveStatus, SolverContext, solve_gadget, solve_sequence
from minirop.state import CPUState, make_symbolic_initial_state
from minirop.symbolic import (BoolExpression, BoolSymbol, Ite, Symbol, WORD_MASK,
                              evaluate, evaluate_bool)
from test_sequence import gadget


def ops(code, address=0x4000):
    instructions = Disassembler().disassemble(bytes.fromhex(code), address)
    assert sum(i.size for i in instructions) == len(bytes.fromhex(code))
    return tuple(lift_instruction(i) for i in instructions)


def state(**registers):
    result = CPUState()
    for name, value in registers.items():
        result.set_register(name, value)
    return result


def reference(lhs, rhs, width, subtract=False):
    modulus = 1 << width
    lhs, rhs = lhs % modulus, rhs % modulus
    full = lhs - rhs if subtract else lhs + rhs
    answer = full % modulus
    signed = lambda number: number if number < modulus // 2 else number - modulus
    signed_full = signed(lhs) - signed(rhs) if subtract else signed(lhs) + signed(rhs)
    return dict(CF=full < 0 if subtract else full >= modulus, ZF=answer == 0,
                SF=answer >= modulus // 2, OF=not -modulus // 2 <= signed_full < modulus // 2,
                PF=bin(answer & 255).count("1") % 2 == 0)


def check_flags(concrete, symbolic, expected, bindings, initial=None):
    context = SolverContext(initial)
    for name, number in bindings.items():
        context.add_equal(Symbol(name), number)
    differences = []
    for name, answer in expected.items():
        assert type(concrete.get_flag(name)) is bool
        assert concrete.get_flag(name) is answer
        flag = symbolic.get_flag(name)
        assert evaluate_bool(flag, bindings) is answer
        differences.append(context.translator.translate_bool(flag) != answer)
    import z3
    context._solver.add(z3.Or(*differences))
    assert context.solve().status is SolveStatus.UNSAT


def test_flag_state_initialization_undefined_copy_and_boolean_sorts():
    original = state(rax=1)
    with pytest.raises(UninitializedFlag):
        original.get_flag("ZF")
    original.set_flag("cf", True)
    copied = original.copy()
    copied.set_flag("CF", False)
    assert original.get_flag("CF") is True
    original.set_flag("CF", UNDEFINED)
    with pytest.raises(UndefinedFlagError):
        original.get_flag("CF")
    with pytest.raises(TypeError):
        original.set_flag("ZF", 1)
    symbolic = make_symbolic_initial_state()
    for name in FLAG_NAMES:
        assert symbolic.get_flag(name) == BoolSymbol(name.lower() + "_0")
    with pytest.raises(TypeError, match="Python"):
        bool(symbolic.get_flag("CF"))
    context = SolverContext()
    context.add_equal(symbolic.get_flag("CF"), True)
    assert context.solve().assignments["cf_0"] is True
    with pytest.raises(TypeError, match="sort"):
        context.add_equal(Symbol("number"), True)


def test_add_flags_for_all_widths_carry_and_signed_overflow():
    for width, code in [(8, "00 d8"), (16, "66 01 d8"), (32, "01 d8"), (64, "48 01 d8")]:
        maximum, sign = (1 << width) - 1, 1 << (width - 1)
        for left, right in [(maximum, 1), (sign - 1, 1), (sign, sign), (1, 2), (0, 0)]:
            initial = state(rax=left, rbx=right)
            concrete = execute(ops(code), initial)
            symbolic = execute(ops(code), make_symbolic_initial_state())
            check_flags(concrete, symbolic, reference(left, right, width), {"rax_0": left, "rbx_0": right})
            assert initial.flag_bindings() == (("AF", UNDEFINED),)
    check_flags(execute(ops("83 c0 ff"), state(rax=1)),
                execute(ops("83 c0 ff"), make_symbolic_initial_state()),
                reference(1, -1, 32), {"rax_0": 1})


def test_sub_flags_for_all_widths_borrow_and_signed_overflow():
    for width, code in [(8, "28 d8"), (16, "66 29 d8"), (32, "29 d8"), (64, "48 29 d8")]:
        maximum, sign = (1 << width) - 1, 1 << (width - 1)
        for left, right in [(0, 1), (sign, 1), (sign - 1, maximum), (7, 7), (2, 1)]:
            check_flags(execute(ops(code), state(rax=left, rbx=right)),
                        execute(ops(code), make_symbolic_initial_state()),
                        reference(left, right, width, True), {"rax_0": left, "rbx_0": right})
    check_flags(execute(ops("48 83 e8 ff"), state(rax=0)),
                execute(ops("48 83 e8 ff"), make_symbolic_initial_state()),
                reference(0, -1, 64, True), {"rax_0": 0})


def test_cmp_register_widths_only_update_flags():
    for width, code in [(8, "38 d8"), (16, "66 39 d8"), (32, "39 d8"), (64, "48 39 d8")]:
        left, right = WORD_MASK, 1
        initial = state(rax=left, rbx=right)
        concrete = execute(ops(code), initial)
        symbolic_initial = make_symbolic_initial_state()
        symbolic = execute(ops(code), symbolic_initial)
        assert concrete.register_bindings() == initial.register_bindings()
        assert symbolic.register_bindings() == symbolic_initial.register_bindings()
        assert concrete.memory_state is initial.memory_state
        check_flags(concrete, symbolic, reference(left, right, width, True), {"rax_0": left, "rbx_0": right})


def test_cmp_encoded_immediates_extend_to_operand_width():
    for code, width, immediate in [("80 f8 ff", 8, -1), ("66 83 f8 ff", 16, -1),
                                   ("83 f8 ff", 32, -1), ("48 83 f8 ff", 64, -1),
                                   ("48 3d 00 00 00 80", 64, -0x80000000),
                                   ("66 81 f8 00 80", 16, 0x8000)]:
        check_flags(execute(ops(code), state(rax=0)), execute(ops(code), make_symbolic_initial_state()),
                    reference(0, immediate, width, True), {"rax_0": 0})
    instruction = Disassembler().disassemble(bytes.fromhex("48 83 f8 ff"))[0]
    for immediate in [Operand("immediate", 32, immediate=-1), Operand("immediate", 64, immediate=1)]:
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(replace(instruction, operands=(instruction.operands[0], immediate)))


def test_cmp_memory_orders_sib_rip_and_current_version():
    for code, width, left, right, address in [
        ("48 3b 03", 64, 1, 2, 0x1000), ("39 48 08", 32, 2, 1, 0x1008),
        ("66 3b 44 8b f8", 16, 1, 2, 0x1000),
        ("48 3b 05 20 00 00 00", 64, 1, 2, 0x4027),
        ("83 38 ff", 32, 2, 0xffffffff, 0x1000),
    ]:
        initial = state(rax=0x1000 if code.startswith(("39", "83")) else 1, rbx=0x1000, rcx=2)
        if code.startswith("39"):
            initial.set_register("rcx", 1)
        initial.store_memory(address, 2)
        version, registers = initial.memory_state, initial.register_bindings()
        result = execute(ops(code), initial)
        assert result.register_bindings() == registers and result.memory_state is version
        for name, answer in reference(left, right, width, True).items():
            assert result.get_flag(name) is answer


def test_test_register_and_immediate_forms_never_write_operands():
    for width, code in [(8, "84 d8"), (16, "66 85 d8"), (32, "85 d8"), (64, "48 85 d8")]:
        old = WORD_MASK
        initial = state(rax=old, rbx=1)
        result = execute(ops(code), initial)
        assert result.register_bindings() == initial.register_bindings()
        expected = dict(reference(0, 1, width), CF=False, OF=False, ZF=False, SF=False, PF=False)
        check_flags(result, execute(ops(code), make_symbolic_initial_state()), expected, {"rax_0": old, "rbx_0": 1})
    for code, width in [("a8 80", 8), ("66 a9 00 80", 16),
                        ("a9 00 00 00 80", 32), ("48 a9 00 00 00 80", 64)]:
        initial = state(rax=WORD_MASK)
        result = execute(ops(code), initial)
        assert result.get_register("rax") == WORD_MASK
        assert result.get_flag("SF") is True and result.get_flag("CF") is False and result.get_flag("OF") is False


def test_test_memory_operands_reuse_reads_without_storing():
    for code, width in [("f6 00 01", 8), ("66 f7 00 01 00", 16),
                        ("f7 00 01 00 00 00", 32), ("48 85 0b", 64)]:
        initial = state(rax=0x1000, rbx=0x1000, rcx=1)
        initial.write_memory(0x1000, 0xff)
        result = execute(ops(code), initial)
        assert result.register_bindings() == initial.register_bindings()
        assert result.memory_state is initial.memory_state and result.read_memory(0x1000) == 0xff
        assert result.get_flag("ZF") is False and result.get_flag("PF") is False
        assert result.get_flag("CF") is False and result.get_flag("OF") is False


def test_xor_and_or_flags_and_uninitialized_self_xor_zero_idiom():
    for operation in ("30 d8", "20 d8", "08 d8"):
        result = execute(ops(operation), state(rax=0x80, rbx=0))
        assert result.get_flag("SF") is (operation != "20 d8")
        assert result.get_flag("ZF") is (operation == "20 d8")
        assert result.get_flag("CF") is False and result.get_flag("OF") is False
        with pytest.raises(UndefinedFlagError):
            result.get_flag("AF")
    for code in ("30 c0", "66 31 c0", "31 c0", "48 31 c0"):
        # Partial writes require an initialized parent; full writes can initialize.
        initial = state(rax=WORD_MASK) if code.startswith(("30", "66")) else CPUState()
        result = execute(ops(code), initial)
        assert {name: result.get_flag(name) for name in FLAG_NAMES} == dict(CF=False, OF=False, ZF=True, SF=False, PF=True)


def test_non_flag_writers_preserve_all_existing_flags():
    codes = ["b8 00 00 00 00", "48 f7 d0", "0f b6 c3", "48 0f be c3", "48 63 c3",
             "48 8d 03", "48 87 d8", "48 8b 03", "48 89 03", "50 59", "c9"]
    for code in codes:
        initial = state(rax=0x80, rbx=0x1000, rcx=1, rsp=0x2000, rbp=0x3000)
        initial.write_memory(0x1000, 0x80)
        initial.write_memory(0x3000, 0x4000)
        for index, name in enumerate(FLAG_NAMES):
            initial.set_flag(name, bool(index % 2))
        result = execute(ops(code), initial)
        assert result.flag_bindings() == initial.flag_bindings()
    symbolic = make_symbolic_initial_state()
    assert execute(ops("b8 00 00 00 00 48 f7 d0"), symbolic).flag_bindings() == symbolic.flag_bindings()


def test_inc_dec_preserve_concrete_symbolic_and_undefined_carry():
    for code, old, subtract in [("fe c0", 0x7f, False), ("fe c8", 0x80, True)]:
        for carry in (False, True, UNDEFINED, BoolSymbol("carry")):
            initial = state(rax=old)
            initial.set_flag("CF", carry)
            result = execute(ops(code), initial)
            assert dict(result.flag_bindings())["CF"] == carry
            for name, answer in reference(old, 1, 8, subtract).items():
                if name != "CF":
                    assert result.get_flag(name) is answer
    unknown = execute(ops("ff c0"), state(rax=0xffffffff))
    with pytest.raises(UninitializedFlag):
        unknown.get_flag("CF")
    assert unknown.get_flag("ZF") is True


def test_neg_flags_for_zero_nonzero_and_minimum_signed_value():
    for width, code in [(8, "f6 d8"), (16, "66 f7 d8"), (32, "f7 d8"), (64, "48 f7 d8")]:
        for old in (0, 1, 1 << (width - 1), (1 << width) - 1):
            result = execute(ops(code), state(rax=old))
            symbolic = execute(ops(code), make_symbolic_initial_state())
            check_flags(result, symbolic, reference(0, old, width, True), {"rax_0": old})
            assert result.get_flag("CF") is (old != 0)
            assert result.get_flag("OF") is (old == 1 << (width - 1))


def test_shifts_preserve_zero_count_and_update_defined_flags():
    initial = state(rax=0x80)
    for name in FLAG_NAMES:
        initial.set_flag(name, True)
    for code in ("c0 e0 00", "c0 e8 20", "c1 e0 20", "48 c1 e8 40"):
        assert execute(ops(code), initial).flag_bindings() == initial.flag_bindings()
    for code, zero, carry, overflow in [("c0 e0 01", True, True, True),
                                        ("c0 e8 01", False, False, True),
                                        ("48 c1 e0 02", False, False, None)]:
        result = execute(ops(code), initial)
        assert result.get_flag("ZF") is zero and result.get_flag("CF") is carry
        if overflow is None:
            with pytest.raises(UndefinedFlagError):
                result.get_flag("OF")
        else:
            assert result.get_flag("OF") is overflow
        with pytest.raises(UndefinedFlagError):
            result.get_flag("AF")
        assert execute(ops("0f 94 c0"), result).get_register("al") == int(zero)
    initial.set_flag("CF", UNDEFINED)
    assert execute(ops("0f 94 c0"), initial).get_register("al") == 1


def test_all_condition_codes_and_aliases_concrete_and_symbolic():
    for cf, zf, sf, of, pf in product((False, True), repeat=5):
        flags = dict(CF=cf, ZF=zf, SF=sf, OF=of, PF=pf)
        initial = CPUState()
        for name, value in flags.items():
            initial.set_flag(name, value)
        expected = dict(o=of, no=not of, b=cf, ae=not cf, e=zf, ne=not zf,
                        be=cf or zf, a=not cf and not zf, s=sf, ns=not sf, p=pf, np=not pf,
                        l=sf != of, ge=sf == of, le=zf or sf != of, g=not zf and sf == of)
        symbolic = make_symbolic_initial_state()
        bindings = {name.lower() + "_0": value for name, value in flags.items()}
        context = SolverContext()
        for name, number in bindings.items():
            context.add_equal(BoolSymbol(name), number)
        differences = []
        for condition, answer in expected.items():
            assert evaluate_condition(condition, initial) is answer
            assert evaluate_condition(condition, flags) is answer
            expression = evaluate_condition(condition, symbolic)
            assert evaluate_bool(expression, bindings) is answer
            differences.append(context.translator.translate_bool(expression) != answer)
        import z3
        context._solver.add(z3.Or(*differences))
        assert context.solve().status is SolveStatus.UNSAT
        for alias, canonical in [("Z", "e"), ("NZ", "ne"), ("C", "b"), ("NC", "ae"),
                                 ("nae", "b"), ("nb", "ae"), ("na", "be"), ("nbe", "a"),
                                 ("nge", "l"), ("nl", "ge"), ("ng", "le"), ("nle", "g"),
                                 ("pe", "p"), ("po", "np")]:
            assert evaluate_condition(alias, initial) is expected[canonical]


def test_setcc_byte_aliases_high_bytes_rex_and_memory_destinations():
    for code, parent, offset in [("0f 94 c0", "rax", 0), ("0f 95 c3", "rbx", 0),
                                ("0f 94 c4", "rax", 8), ("41 0f 94 c0", "r8", 0),
                                ("40 0f 94 c4", "rsp", 0)]:
        initial = state(**{parent: 0x1122334455667788})
        initial.set_flag("ZF", not "95" in code)
        result = execute(ops(code), initial)
        expected = (0x1122334455667788 & ~(255 << offset)) | (1 << offset)
        assert result.get_register(parent) == expected
        assert result.flag_bindings() == initial.flag_bindings()
    initial = state(rax=0x1000)
    initial.set_flag("ZF", True)
    initial.write_memory(0x1000, 0x1122334455667788)
    result = execute(ops("0f 94 00"), initial)
    assert result.read_memory(0x1000) == 0x1122334455667701
    assert result.flag_bindings() == initial.flag_bindings()
    instruction = Disassembler().disassemble(bytes.fromhex("40 0f 94 c4"))[0]
    with pytest.raises(UnsupportedInstruction, match="high-byte"):
        lift_instruction(replace(instruction, operands=(Operand("register", 8, register="ah"),)))
    with pytest.raises(ValueError, match="8-bit"):
        execute([Setcc("e", "eax")], make_symbolic_initial_state())


def test_symbolic_setcc_goals_and_unknown_initial_flag_witness():
    code = gadget("48 39 d8 0f 94 c1 c3", 0x4000)
    equal = [Equality(Symbol("rax_0"), 9), Equality(Symbol("rbx_0"), 9)]
    unequal = [Equality(Symbol("rax_0"), 9), Equality(Symbol("rbx_0"), 10)]
    assert solve_gadget(code, {"cl": 1}, constraints=equal).status is SolveStatus.SAT
    assert solve_gadget(code, {"cl": 1}, constraints=unequal).status is SolveStatus.UNSAT
    initial = make_symbolic_initial_state()
    result = solve_gadget(gadget("0f 95 c1 c3", 0x4000), {"cl": 1}, initial_state=initial)
    assert result.status is SolveStatus.SAT and result.assignments["zf_0"] is False
    value = execute(ops("0f 95 c1"), initial).get_register("cl")
    assert evaluate(value, result.assignments) == 1
    with pytest.raises(UninitializedFlag):
        execute(ops("0f 95 c1"), state(rcx=0))


def test_cmov_true_false_all_widths_follow_x86_64_destination_rules():
    old, source = 0xaabbccddeeff0011, 0x1122334455667788
    for code, taken in [("66 0f 45 c3", 0xaabbccddeeff7788),
                        ("0f 45 c3", 0x55667788), ("48 0f 45 c3", source)]:
        for zf in (False, True):
            initial = state(rax=old, rbx=source)
            initial.set_flag("ZF", zf)
            result = execute(ops(code), initial)
            untaken = old & 0xffffffff if code == "0f 45 c3" else old
            assert result.get_register("rax") == (untaken if zf else taken)
            assert result.get_register("rbx") == source and result.flag_bindings() == initial.flag_bindings()
    assert Disassembler().disassemble(bytes.fromhex("f0 39 18")) == ()
    with pytest.raises(ValueError, match="matching"):
        execute([Cmovcc("e", "al", "bl")], make_symbolic_initial_state())
    for code in ("48 0f 44 03", "eb 00", "e9 00 00 00 00", "67 39 18", "64 85 18"):
        with pytest.raises(UnsupportedInstruction):
            ops(code)


def test_symbolic_cmov_ite_selects_physical_register_and_is_solvable():
    old = 0xaabbccddeeff0011
    for code, target, source, taken in [("48 39 d8 48 0f 44 ca", "rcx", "rdx", 0x778899),
                                        ("48 39 d8 0f 44 ca", "rcx", "rdx", 0x778899),
                                        ("48 39 d8 66 0f 44 ca", "rcx", "rdx", 0xaabbccddeeff8899)]:
        initial = make_symbolic_initial_state()
        value = execute(ops(code), initial).get_register(target)
        assert isinstance(value, Ite) and isinstance(value.condition, BoolExpression)
        assert "if(" in str(value)
        for lhs, rhs in [(1, 1), (1, 2)]:
            bindings = {"rax_0": lhs, "rbx_0": rhs, "rcx_0": old, "rdx_0": 0x778899}
            untaken = old & 0xffffffff if code.endswith("d8 0f 44 ca") else old
            answer = taken if lhs == rhs else untaken
            assert evaluate(value, bindings) == answer
            conditions = [Equality(Symbol(name), number) for name, number in bindings.items()]
            solved = solve_gadget(gadget(code + " c3", 0x4000), {target: answer}, constraints=conditions)
            assert solved.status is SolveStatus.SAT
            assert solve_gadget(gadget(code + " c3", 0x4000), {target: answer ^ 1}, constraints=conditions).status is SolveStatus.UNSAT


def test_required_mixed_cases_distinguish_signed_and_unsigned_conditions():
    cases = [("b8 0a 00 00 00 83 f8 14 0f 9c c3", {"bl": 1}),
             ("b8 ff ff ff ff 83 f8 01 0f 9f c3 0f 97 c1", {"bl": 0, "cl": 1}),
             ("48 85 c0 0f 95 c2", {"dl": 1})]
    for code, expected in cases:
        initial = state(rax=7, rbx=0x1122334455667788, rcx=0x8877665544332211, rdx=0)
        result = execute(ops(code), initial)
        symbolic = execute(ops(code), make_symbolic_initial_state())
        bindings = {name + "_0": number for name, number in initial.register_bindings()}
        for name, answer in expected.items():
            assert result.get_register(name) == answer
            assert evaluate(symbolic.get_register(name), bindings) == answer
        assert solve_gadget(gadget(code + " c3", 0x4000), expected,
                            constraints=[Equality(Symbol(name), number) for name, number in bindings.items()]).status is SolveStatus.SAT


def test_flags_flow_across_gadgets_and_conditional_memory_writes():
    first = gadget("48 39 d8 c3", 0x4000)
    second = gadget("0f 92 c1 0f 9c c2 c3", 0x4100)
    initial = make_symbolic_initial_state()
    initial.set_stack_slot(0, 0x4100)
    initial.set_stack_slot(1, 0x9000)
    conditions = [Equality(Symbol("rax_0"), WORD_MASK), Equality(Symbol("rbx_0"), 1)]
    solved = solve_sequence([first, second], {"cl": 0, "dl": 1}, initial_state=initial, constraints=conditions)
    assert solved.status is SolveStatus.SAT
    assert solved.effect.steps[0].after.flags == solved.effect.steps[1].before.flags
    assert solved.effect.steps[0].after.flags == solved.effect.final_state.flags
    assert evaluate_bool(solved.effect.final_state.get_flag("CF"), {"rax_0": WORD_MASK, "rbx_0": 1}) is False
    # Symbolic SETcc store and MOV load share the existing byte memory.
    code = "48 39 d8 0f 94 05 20 00 00 00 0f b6 05 19 00 00 00"
    value = execute(ops(code), make_symbolic_initial_state()).get_register("rax")
    for left, right in [(1, 1), (1, 2)]:
        assert evaluate(value, {"rax_0": left, "rbx_0": right}) == int(left == right)
    from minirop.manifest import create_analysis_manifest
    custom = state(rax=0)
    custom.set_flag("ZF", True)
    with pytest.raises(ValueError, match="flag states"):
        create_analysis_manifest(sequence=[gadget("0f 94 c0 c3", 0x4000)], initial_state=custom)
