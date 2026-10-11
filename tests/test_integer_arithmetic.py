"""CPU integer arithmetic, independent numeric references and existing Z3 integration."""
from dataclasses import replace

import pytest

from minirop.constraints import Equality
from minirop.disasm import Disassembler
from minirop.executor import execute
from minirop.flags import FLAG_NAMES, UNDEFINED, UndefinedFlagError, UninitializedFlag
from minirop.gadget import Gadget
from minirop.ir import Adc, Sbb, Multiply, Sar
from minirop.program import execute_program, ProgramStatus
from minirop.semantics import lift_instruction, UnsupportedInstruction
from minirop.solver import SolverContext, SolveStatus, solve_gadget
from minirop.state import CPUState, make_symbolic_initial_state
from minirop.symbolic import (BoolSymbol, ProductPart, Symbol, WORD_MASK, evaluate, evaluate_bool,
                              format_value, multiply_values)


def instructions(raw, address=0x4000):
    data = bytes.fromhex(raw)
    result = Disassembler().disassemble(data, address)
    assert sum(item.size for item in result) == len(data)
    return result


def run(raw, initial):
    return execute(tuple(lift_instruction(item) for item in instructions(raw)), initial)


def state(**registers):
    result = CPUState()
    for name, value in registers.items():
        result.set_register(name, value)
    return result


def signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value >= 1 << (width - 1) else value


def reference_arithmetic(left, right, carry, width, subtract):
    modulus = 1 << width
    left, right = left % modulus, right % modulus
    full = left - right - carry if subtract else left + right + carry
    signed_full = signed(left, width) - signed(right, width) - carry if subtract else signed(left, width) + signed(right, width) + carry
    result = full % modulus
    flags = dict(CF=full < 0 if subtract else full >= modulus,
                 OF=not -modulus // 2 <= signed_full < modulus // 2,
                 ZF=result == 0, SF=result >= modulus // 2, PF=(result & 255).bit_count() % 2 == 0)
    return result, flags


def assert_symbolic_matches(effect, inputs, values, flags=(), *, memory=None):
    import z3
    context = SolverContext()
    for name, value in inputs.items():
        context.add_equal(BoolSymbol(name) if type(value) is bool else Symbol(name), value)
    differences = []
    for register, expected in values.items():
        value = effect.get_register(register)
        assert evaluate(value, inputs, memory=memory) == expected
        differences.append(context.translator.translate(value) != expected)
    for name, expected in dict(flags).items():
        value = effect.get_flag(name)
        assert evaluate_bool(value, inputs, memory=memory) is expected
        differences.append(context.translator.translate_bool(value) != expected)
    if memory:
        from minirop.symbolic import MemoryRead
        for address, value in memory.items():
            context.add_equal(MemoryRead(address), value)
    context._solver.add(z3.Or(*differences))
    assert context.solve().status is SolveStatus.UNSAT


def test_mul_implicit_results_all_widths_preserve_uncovered_register_bits():
    for raw, width, low, high in [("f6 e3", 8, "ax", None), ("66 f7 e3", 16, "ax", "dx"),
                                  ("f7 e3", 32, "eax", "edx"), ("48 f7 e3", 64, "rax", "rdx")]:
        original, source, old_high = 0xaabbccddeeff0080, 2, 0x1122334455667788
        initial = state(rax=original, rbx=source, rdx=old_high)
        result = run(raw, initial)
        product = (original & ((1 << width) - 1)) * source
        if width == 8:
            expected_low, expected_high = (original & ~0xffff) | product, old_high
        elif width == 16:
            expected_low = (original & ~0xffff) | (product & 0xffff)
            expected_high = (old_high & ~0xffff) | (product >> 16)
        else:
            expected_low, expected_high = product & ((1 << width) - 1), product >> width
        assert result.get_register("rax") == expected_low and result.get_register("rdx") == expected_high
        assert result.get_register("rbx") == source and initial.get_register("rax") == original


def test_mul_cf_of_use_high_half_and_32bit_writes_clear_upper_bits():
    for raw, width in [("f6 e3", 8), ("66 f7 e3", 16), ("f7 e3", 32), ("48 f7 e3", 64)]:
        for operand in (2, (1 << width) - 1):
            result = run(raw, state(rax=operand, rbx=2, rdx=WORD_MASK))
            overflow = operand * 2 >= 1 << width
            assert result.get_flag("CF") is overflow and result.get_flag("OF") is overflow
    result = run("f7 e3", state(rax=0xaabbccdd00000003, rbx=2, rdx=WORD_MASK))
    assert result.get_register("rax") == 6 and result.get_register("rdx") == 0


def test_mul64_full_128bit_boundary_and_overlapping_rdx_source():
    for raw, multiplier in [("48 f7 e3", 2), ("48 f7 e3", WORD_MASK), ("48 f7 e2", 2)]:
        result = run(raw, state(rax=WORD_MASK, rbx=multiplier, rdx=multiplier))
        product = WORD_MASK * multiplier
        assert result.get_register("rax") == product & WORD_MASK
        assert result.get_register("rdx") == product >> 64
        assert result.get_flag("CF") is True and result.get_flag("OF") is True
    assert run("48 f7 e3", state(rax=WORD_MASK, rbx=2)).get_register("rdx") == 1


def test_mul_memory_sources_all_widths_and_rip_relative():
    for raw, width in [("f6 21", 8), ("66 f7 21", 16), ("f7 21", 32), ("48 f7 21", 64)]:
        initial = state(rax=3, rdx=0, rcx=0x2000)
        initial.write_memory(0x2000, 2)
        result = run(raw, initial)
        assert result.get_register("rax") == 6 and result.get_register("rdx") == 0
        assert result.read_memory(0x2000) == 2
    initial = state(rax=WORD_MASK)
    initial.write_memory(0x4017, 2)
    result = run("48 f7 25 10 00 00 00", initial)
    assert result.get_register("rax") == WORD_MASK - 1 and result.get_register("rdx") == 1


def test_symbolic_mul_high_low_values_match_concrete_and_solver():
    for raw, width in [("f6 e3", 8), ("66 f7 e3", 16), ("f7 e3", 32), ("48 f7 e3", 64)]:
        inputs = dict(rax_0=WORD_MASK, rbx_0=2, rdx_0=0x1122334455667788)
        concrete = run(raw, state(rax=inputs["rax_0"], rbx=2, rdx=inputs["rdx_0"]))
        symbolic = run(raw, make_symbolic_initial_state(0))
        assert_symbolic_matches(symbolic, inputs,
                                {name: concrete.get_register(name) for name in ("rax", "rdx")},
                                {name: concrete.get_flag(name) for name in ("CF", "OF")})
    symbolic = run("48 f7 21", make_symbolic_initial_state(0))
    assert_symbolic_matches(symbolic, dict(rax_0=WORD_MASK, rcx_0=0x2000),
                            dict(rax=WORD_MASK - 1, rdx=1), dict(CF=True, OF=True),
                            memory={0x2000: 2})
    low, high = multiply_values(Symbol("a"), Symbol("b"), 64)
    assert isinstance(high, ProductPart) and "unsigned_mul64_high" in format_value(high)
    assert evaluate(high, dict(a=WORD_MASK, b=2)) == 1
    with pytest.raises(ValueError):
        multiply_values(Symbol("a"), 1, 128)


def test_single_imul_signed_double_width_positive_negative_and_overflow():
    for raw, width in [("f6 eb", 8), ("66 f7 eb", 16), ("f7 eb", 32), ("48 f7 eb", 64)]:
        for left, right in [(3, 4), (-8, 2), (1 << (width - 1), 2), (1 << (width - 1), -1)]:
            initial = state(rax=left & WORD_MASK, rbx=right & WORD_MASK, rdx=0)
            result = run(raw, initial)
            full = signed(left, width) * signed(right, width)
            product = full % (1 << (2 * width))
            low = result.get_register("ax" if width == 8 else {16: "ax", 32: "eax", 64: "rax"}[width])
            assert low == product & ((1 << (16 if width == 8 else width)) - 1)
            if width != 8:
                assert result.get_register({16: "dx", 32: "edx", 64: "rdx"}[width]) == product >> width
            overflow = not -(1 << (width - 1)) <= full < (1 << (width - 1))
            assert result.get_flag("CF") is overflow and result.get_flag("OF") is overflow
    assert run("f6 eb", state(rax=0x80, rbx=2)).get_register("ax") == 0xff00


def test_two_operand_imul_register_and_memory_truncation_overflow():
    for raw, width in [("66 0f af c3", 16), ("0f af c3", 32), ("48 0f af c3", 64)]:
        for left, right in [(-3, 4), (5, -2), ((1 << (width - 1)) - 1, 2), (1 << (width - 1), -1)]:
            initial = state(rax=left & WORD_MASK, rbx=right & WORD_MASK, rdx=0x12345678)
            result = run(raw, initial)
            product = signed(left, width) * signed(right, width)
            name = {16: "ax", 32: "eax", 64: "rax"}[width]
            assert result.get_register(name) == product % (1 << width)
            assert result.get_register("rdx") == 0x12345678
            assert result.get_flag("OF") is (not -(1 << (width - 1)) <= product < (1 << (width - 1)))
    for raw, width in [("66 0f af 0e", 16), ("0f af 0e", 32), ("48 0f af 0e", 64)]:
        initial = state(rcx=0xaabbccdd00000003, rsi=0x2000, rdx=99)
        initial.write_memory(0x2000, WORD_MASK)
        result = run(raw, initial)
        assert result.get_register({16: "cx", 32: "ecx", 64: "rcx"}[width]) == (-signed(initial.get_register("rcx"), width)) % (1 << width)
        assert result.read_memory(0x2000) == WORD_MASK and result.get_register("rdx") == 99
    assert run("0f af c3", state(rax=0xaabbccdd00000003, rbx=2)).get_register("rax") == 6


def test_three_operand_imul_imm8_sign_extension_and_memory_source():
    for raw, width, immediate in [("66 6b c3 fd", 16, -3), ("6b c3 80", 32, -128),
                                  ("48 6b c3 fd", 64, -3), ("48 6b c3 7f", 64, 127)]:
        result = run(raw, state(rax=WORD_MASK, rbx=2))
        assert result.get_register({16: "ax", 32: "eax", 64: "rax"}[width]) == (2 * immediate) % (1 << width)
    initial = state(rsi=0x2000)
    initial.write_memory(0x2000, -3)
    result = run("48 6b 0e 08", initial)
    assert result.get_register("rcx") == (-24) & WORD_MASK
    # Three-operand IMUL does not read an uninitialized explicit 32/64-bit destination.
    assert run("6b c3 0a", state(rbx=3)).get_register("rax") == 30


def test_imul_imm16_imm32_encodings_and_illegal_imm64_are_rejected():
    cases = [("66 69 c3 00 80", 16, -32768), ("69 c3 00 00 00 80", 32, -2147483648),
             ("48 69 c3 00 00 00 80", 64, -2147483648), ("48 69 c3 ff ff ff ff", 64, -1)]
    for raw, width, immediate in cases:
        item = instructions(raw)[0]
        operation = lift_instruction(item)
        assert operation.immediate == immediate
        result = execute([operation], state(rax=0, rbx=2))
        assert result.get_register({16: "ax", 32: "eax", 64: "rax"}[width]) == (2 * immediate) % (1 << width)
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(replace(item, operands=item.operands[:2] + (replace(item.operands[2], immediate=123),)))
    item = instructions("48 69 c3 01 00 00 00")[0]
    with pytest.raises(UnsupportedInstruction):
        lift_instruction(replace(item, bytes=item.bytes + bytes(4), size=item.size + 4))
    with pytest.raises(ValueError, match="encoding"):
        execute([Multiply("rbx", 64, signed=True, dst="rax", immediate=1 << 40)], make_symbolic_initial_state(0))


def test_symbolic_imul_full_product_and_signed_overflow_match_z3():
    for raw in ("f6 eb", "66 f7 eb", "f7 eb", "48 f7 eb", "48 0f af c3", "48 6b c3 fd"):
        inputs = dict(rax_0=1 << 63 if "48" in raw else WORD_MASK, rbx_0=WORD_MASK, rdx_0=0)
        concrete = run(raw, state(rax=inputs["rax_0"], rbx=inputs["rbx_0"], rdx=0))
        symbolic = run(raw, make_symbolic_initial_state(0))
        assert_symbolic_matches(symbolic, inputs,
                                {name: concrete.get_register(name) for name in ("rax", "rdx")},
                                {name: concrete.get_flag(name) for name in ("CF", "OF")})
    result = run("48 f7 eb", state(rax=1 << 63, rbx=WORD_MASK))
    assert result.get_register("rax") == 1 << 63 and result.get_register("rdx") == 0
    assert result.get_flag("CF") is True


def test_multiply_marks_zf_sf_pf_af_undefined_and_condition_reads_fail():
    for raw in ("f6 e3", "48 f7 e3", "f6 eb", "48 0f af c3", "48 6b c3 02"):
        initial = state(rax=3, rbx=2)
        for name in FLAG_NAMES:
            initial.set_flag(name, True)
        result = run(raw, initial)
        for name in ("ZF", "SF", "PF", "AF"):
            with pytest.raises(UndefinedFlagError):
                result.get_flag(name)
        with pytest.raises(UndefinedFlagError):
            run("0f 94 c1", result)
        assert run("0f 90 c1", state_from(result, rcx=0)).get_register("cl") == 0


def state_from(initial, **registers):
    copied = initial.copy()
    for name, value in registers.items():
        copied.set_register(name, value)
    return copied


def test_adc_all_widths_carry_and_signed_overflow_boundaries():
    for raw, width in [("10 d8", 8), ("66 11 d8", 16), ("11 d8", 32), ("48 11 d8", 64)]:
        mask = (1 << width) - 1
        for left, right in [(0, 0), (mask, 0), (mask, 1), (mask, mask), ((mask >> 1), 0), (1 << (width - 1), mask)]:
            for carry in (False, True):
                initial = state(rax=left, rbx=right)
                initial.set_flag("CF", carry)
                result = run(raw, initial)
                expected, flags = reference_arithmetic(left, right, int(carry), width, False)
                assert result.get_register({8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]) == expected
                for name, answer in flags.items():
                    assert result.get_flag(name) is answer
                assert result.get_flag("AF") is bool((left ^ right ^ expected) & 16)


def test_sbb_all_widths_include_borrow_even_when_source_plus_cf_wraps():
    for raw, width in [("18 d8", 8), ("66 19 d8", 16), ("19 d8", 32), ("48 19 d8", 64)]:
        mask = (1 << width) - 1
        for left, right in [(0, 0), (0, mask), (mask, mask), (mask >> 1, mask), (1 << (width - 1), 0)]:
            for carry in (False, True):
                initial = state(rax=left, rbx=right)
                initial.set_flag("CF", carry)
                result = run(raw, initial)
                expected, flags = reference_arithmetic(left, right, int(carry), width, True)
                assert result.get_register({8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]) == expected
                for name, answer in flags.items():
                    assert result.get_flag(name) is answer
    initial = state(rax=0, rbx=0)
    initial.set_flag("CF", True)
    assert run("18 d8", initial).get_register("al") == 255
    assert run("28 d8", initial).get_register("al") == 0  # ordinary SUB


def test_adc_sbb_memory_directions_widths_and_rip_sources():
    original = 0x1122334455667788
    for prefix, width in [("", 8), ("66 ", 16), ("", 32), ("48 ", 64)]:
        for subtract, base in [(False, 0x10), (True, 0x18)]:
            for memory_destination in (True, False):
                opcode = base + (0 if width == 8 else 1) + (0 if memory_destination else 2)
                raw = prefix + f"{opcode:02x} 06"
                initial = state(rax=3, rsi=0x2000)
                initial.set_flag("CF", True)
                initial.write_memory(0x2000, original)
                result = run(raw, initial)
                left, right = (original, 3) if memory_destination else (3, original)
                expected, flags = reference_arithmetic(left, right, 1, width, subtract)
                if memory_destination:
                    assert result.read_memory(0x2000) == (original & ~((1 << width) - 1)) | expected
                    assert result.get_register("rax") == 3
                else:
                    assert result.get_register({8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]) == expected
                    assert result.read_memory(0x2000) == original
                for name, answer in flags.items():
                    assert result.get_flag(name) is answer
    initial = state(rax=3)
    initial.set_flag("CF", True)
    initial.write_memory(0x4017, 2)
    assert run("48 13 05 10 00 00 00", initial).get_register("rax") == 6
    # Symbolic memory operands and CF reuse the existing byte-array context.
    symbolic = make_symbolic_initial_state(0)
    effect = run("48 11 06", symbolic)
    assert evaluate(effect.read_memory(0x2000), dict(rax_0=3, cf_0=True, rsi_0=0x2000), memory={0x2000: 5}) == 9
    context = SolverContext(symbolic)
    from minirop.symbolic import MemoryRead
    context.add_equal(MemoryRead(0x2000), 5)
    context.add_equal(Symbol("rax_0"), 3)
    context.add_equal(Symbol("rsi_0"), 0x2000)
    context.add_equal(BoolSymbol("cf_0"), True)
    context.add_equal(effect.read_memory(0x2000), 10)
    assert context.solve().status is SolveStatus.UNSAT


def test_adc_sbb_immediate_forms_sign_extension_and_encoding_validation():
    for raw, width, subtract, right in [("14 ff", 8, False, -1), ("1c ff", 8, True, -1),
                                        ("80 d0 ff", 8, False, -1), ("80 d8 ff", 8, True, -1),
                                        ("66 15 00 80", 16, False, -32768), ("66 1d 00 80", 16, True, -32768),
                                        ("15 00 00 00 80", 32, False, -2147483648), ("1d 00 00 00 80", 32, True, -2147483648),
                                        ("48 15 ff ff ff ff", 64, False, -1), ("48 1d ff ff ff ff", 64, True, -1),
                                        ("66 83 d0 ff", 16, False, -1), ("83 d8 ff", 32, True, -1),
                                        ("48 83 d0 ff", 64, False, -1), ("48 81 d8 00 00 00 80", 64, True, -2147483648)]:
        initial = state(rax=3)
        initial.set_flag("CF", True)
        item = instructions(raw)[0]
        result = execute([lift_instruction(item)], initial)
        expected, flags = reference_arithmetic(3, right, 1, width, subtract)
        assert result.get_register({8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]) == expected
        for name, answer in flags.items():
            assert result.get_flag(name) is answer
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(replace(item, operands=(item.operands[0], replace(item.operands[1], immediate=123))))
    for raw in ("48 81 16 ff ff ff ff", "48 83 1e ff"):
        initial = state(rsi=0x2000)
        initial.set_flag("CF", True)
        initial.write_memory(0x2000, 3)
        subtract = "1e" in raw
        expected, _ = reference_arithmetic(3, -1, 1, 64, subtract)
        assert run(raw, initial).read_memory(0x2000) == expected
    item = instructions("48 81 d0 01 00 00 00")[0]
    with pytest.raises(UnsupportedInstruction):
        lift_instruction(replace(item, bytes=item.bytes + bytes(4), size=item.size + 4))


def prove_carry_arithmetic(subtract):
    import z3
    for register, width in [("al", 8), ("ax", 16), ("eax", 32), ("rax", 64)]:
        initial = make_symbolic_initial_state(0)
        source = {8: "bl", 16: "bx", 32: "ebx", 64: "rbx"}[width]
        effect = execute([(Sbb if subtract else Adc)(register, source, width)], initial)
        context = SolverContext()
        left64 = context.translator.translate(Symbol("rax_0"))
        right64 = context.translator.translate(Symbol("rbx_0"))
        left, right = z3.Extract(width - 1, 0, left64), z3.Extract(width - 1, 0, right64)
        carry = context.translator.translate_bool(BoolSymbol("cf_0"))
        bit = z3.If(carry, z3.BitVecVal(1, width + 1, ctx=context.translator.context),
                    z3.BitVecVal(0, width + 1, ctx=context.translator.context))
        full = z3.ZeroExt(1, left) - z3.ZeroExt(1, right) - bit if subtract else z3.ZeroExt(1, left) + z3.ZeroExt(1, right) + bit
        signed_full = z3.SignExt(1, left) - z3.SignExt(1, right) - bit if subtract else z3.SignExt(1, left) + z3.SignExt(1, right) + bit
        low = z3.Extract(width - 1, 0, full)
        expected = dict(CF=z3.Extract(width, width, full) == 1,
                        OF=signed_full != z3.SignExt(1, low), ZF=low == 0,
                        SF=z3.Extract(width - 1, width - 1, low) == 1)
        parity = z3.BoolVal(False, ctx=context.translator.context)
        for index in range(8):
            parity = z3.Xor(parity, z3.Extract(index, index, low) == 1)
        expected["PF"] = z3.Not(parity)
        value = z3.ZeroExt(64 - width, low) if width < 64 else low
        differences = [context.translator.translate(effect.get_register(register)) != value]
        differences.extend(context.translator.translate_bool(effect.get_flag(name)) != answer for name, answer in expected.items())
        context._solver.add(z3.Or(*differences))
        assert context.solve().status is SolveStatus.UNSAT
        for carry_value in (False, True):
            inputs = dict(rax_0=WORD_MASK, rbx_0=0, cf_0=carry_value)
            answer, flags = reference_arithmetic(WORD_MASK, 0, int(carry_value), width, subtract)
            assert evaluate(effect.get_register(register), inputs) == answer
            for name, expected_flag in flags.items():
                assert evaluate_bool(effect.get_flag(name), inputs) is expected_flag


def test_symbolic_adc_cf_and_all_flags_match_widened_z3_reference():
    prove_carry_arithmetic(False)


def test_symbolic_sbb_cf_and_all_flags_match_widened_z3_reference():
    prove_carry_arithmetic(True)


def test_clc_stc_cmc_preserve_other_flags_and_handle_symbolic_undefined_cf():
    initial = state(rax=9)
    for name in FLAG_NAMES:
        initial.set_flag(name, BoolSymbol(name))
    for raw, answer in [("f8", False), ("f9", True)]:
        result = run(raw, initial)
        assert result.get_flag("CF") is answer and result.get_register("rax") == 9
        assert {name: value for name, value in result.flag_bindings() if name != "CF"} == {name: value for name, value in initial.flag_bindings() if name != "CF"}
    inverted = run("f5", initial)
    for value in (False, True):
        assert evaluate_bool(inverted.get_flag("CF"), {"CF": value}) is (not value)
    restored = run("f5 f5", initial).get_flag("CF")
    for value in (False, True):
        assert evaluate_bool(restored, {"CF": value}) is value
    context = SolverContext()
    context.add_equal(restored, initial.get_flag("CF"))
    assert context.solve().status is SolveStatus.SAT
    initial.set_flag("CF", UNDEFINED)
    with pytest.raises(UndefinedFlagError):
        run("f5", initial).get_flag("CF")
    assert run("f8", initial).get_flag("CF") is False
    assert run("f9", initial).get_flag("CF") is True
    with pytest.raises(UninitializedFlag):
        run("f5", state())
    assert run("f9 48 11 d8", state(rax=0, rbx=0)).get_register("rax") == 1
    assert run("f8 48 19 d8", state(rax=5, rbx=2)).get_register("rax") == 3


def test_adc_sbb_reject_undefined_or_uninitialized_cf_without_mutating_input():
    for raw in ("48 11 d8", "48 19 d8"):
        for undefined in (False, True):
            initial = state(rax=0, rbx=0)
            if undefined:
                initial.set_flag("CF", UNDEFINED)
            before = initial.input_bindings(), initial.flag_bindings()
            with pytest.raises(UndefinedFlagError if undefined else UninitializedFlag):
                run(raw, initial)
            assert (initial.input_bindings(), initial.flag_bindings()) == before
    for raw in ("0f a2", "0f 05", "d9 e8", "f0 48 11 18", "67 48 19 18"):
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(instructions(raw)[0])
    with pytest.raises(ValueError, match="memory/memory"):
        from minirop.ir import Address
        execute([Adc(Address("rax"), Address("rbx"), 64)], make_symbolic_initial_state(0))


def test_sar_preserves_negative_sign_and_rounds_toward_negative_infinity():
    for raw, width, shift in [("c0 f8 01", 8, 1), ("66 c1 f8 02", 16, 2),
                              ("c1 f8 04", 32, 4), ("48 c1 f8 01", 64, 1)]:
        for operand in (-8, -9, -(1 << (width - 1)), 7):
            initial = state(rax=operand & WORD_MASK)
            result = run(raw, initial)
            name = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]
            expected = (operand >> shift) % (1 << width)
            assert result.get_register(name) == expected
            if width < 32:
                assert result.get_register("rax") >> width == initial.get_register("rax") >> width
            elif width == 32:
                assert result.get_register("rax") == expected
    assert run("c0 f8 01", state(rax=0xf8)).get_register("al") == 0xfc
    assert run("c0 f8 02", state(rax=0xf7)).get_register("al") == 0xfd
    assert run("d1 f8", state(rax=0xffffffff)).get_register("rax") == 0xffffffff


def test_sar_masked_counts_flags_large_counts_and_symbolic_z3_consistency():
    import z3
    for register, width in [("al", 8), ("ax", 16), ("eax", 32), ("rax", 64)]:
        for count in (0, 1, 2, width, width + 1, 31, 64, 65):
            effective = count & (63 if width == 64 else 31)
            initial = make_symbolic_initial_state(0)
            effect = execute([Sar(register, count)], initial)
            if effective == 0:
                assert effect.flag_bindings() == initial.flag_bindings()
                continue
            for original in (0x7f, WORD_MASK, (1 << (width - 1)) | 1):
                concrete = state(rax=original)
                actual = execute([Sar(register, count)], concrete)
                expected = (signed(original, width) >> effective) % (1 << width)
                carry = bool(original & (1 << (min(effective, width) - 1)))
                flags = dict(CF=carry, ZF=expected == 0, SF=bool(expected >> (width - 1)), PF=(expected & 255).bit_count() % 2 == 0)
                if effective == 1:
                    flags["OF"] = False
                else:
                    with pytest.raises(UndefinedFlagError):
                        actual.get_flag("OF")
                for name, answer in flags.items():
                    assert actual.get_flag(name) is answer
                    assert evaluate_bool(effect.get_flag(name), {"rax_0": original}) is answer
                with pytest.raises(UndefinedFlagError):
                    actual.get_flag("AF")
            context = SolverContext()
            source = z3.Extract(width - 1, 0, context.translator.translate(Symbol("rax_0")))
            shifted = source >> effective
            expected = z3.ZeroExt(64 - width, shifted) if width < 64 else shifted
            carry = z3.Extract(min(effective, width) - 1, min(effective, width) - 1, source) == 1
            context._solver.add(z3.Or(context.translator.translate(effect.get_register(register)) != expected,
                                     context.translator.translate_bool(effect.get_flag("CF")) != carry))
            assert context.solve().status is SolveStatus.UNSAT


def test_new_arithmetic_composes_with_setcc_cmovcc_jcc_and_existing_solver():
    initial = state(rax=WORD_MASK, rbx=2, rcx=0, rsi=7, rdi=99)
    result = run("48 f7 e3 0f 92 c1 48 0f 42 f7", initial)
    assert result.get_register("cl") == 1 and result.get_register("rsi") == 99
    raw = "48 f7 e3 0f 92 c1 48 0f 42 f7 c3"
    items = instructions(raw)
    gadget = Gadget(0x4000, bytes.fromhex(raw), items)
    constraints = [Equality(Symbol("rax_0"), WORD_MASK), Equality(Symbol("rbx_0"), 2), Equality(Symbol("rdi_0"), 99)]
    solved = solve_gadget(gadget, {"rax": WORD_MASK - 1, "rdx": 1, "cl": 1, "rsi": 99}, constraints=constraints)
    assert solved.status is SolveStatus.SAT
    assert solve_gadget(gadget, {"rdx": 0}, constraints=constraints).status is SolveStatus.UNSAT
    program = instructions("48 f7 e3 72 05 b9 11 00 00 00 ba 22 00 00 00")
    execution = execute_program(program, initial)
    assert execution.status is ProgramStatus.COMPLETE and execution.paths[0].state.get_register("rcx") == 0
    # Carry production and consumption flow through unchanged Jcc handling.
    program = instructions("f9 48 11 d8 74 05 b9 11 00 00 00 ba 22 00 00 00")
    execution = execute_program(program, state(rax=WORD_MASK, rbx=0, rcx=0))
    assert execution.status is ProgramStatus.COMPLETE and execution.paths[0].state.get_register("rcx") == 0
    rejected = execute_program(instructions("48 f7 e3 74 00 b9 01 00 00 00"), initial)
    assert rejected.status is ProgramStatus.UNSUPPORTED and "undefined" in rejected.paths[0].reason
