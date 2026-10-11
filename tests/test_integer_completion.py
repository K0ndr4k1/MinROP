"""Integer boundaries and shared concrete/BitVec execution, using explicit bytes."""
from dataclasses import replace

import pytest
import z3

from minirop.constraints import Equality
from minirop.disasm import Disassembler
from minirop.executor import execute
from minirop.flags import UNDEFINED, UndefinedFlagError
from minirop.gadget import Gadget, Operand
from minirop.ir import IntegerOp
from minirop.program import execute_program, ProgramStatus
from minirop.semantics import lift_instruction, UnsupportedInstruction
from minirop.sequence import StateSnapshot
from minirop.solver import (SolverContext, SolveResult, SolveStatus, solve_gadget, solve_effect,
                           solve_sequence, solve_program_path)
from minirop.search import search_sequences, SearchStatus
from minirop.state import CPUState, REGISTER_NAMES, make_symbolic_initial_state
from minirop.symbolic import Symbol, BoolSymbol, WORD_MASK, UndefinedValueError, evaluate, evaluate_bool


def instructions(raw, address=0x4000):
    data = bytes.fromhex(raw)
    result = Disassembler().disassemble(data, address)
    assert sum(i.size for i in result) == len(data)
    return result


def run(raw, initial):
    return execute(tuple(lift_instruction(i) for i in instructions(raw)), initial)


def state(**registers):
    result = CPUState()
    for name in REGISTER_NAMES:
        result.set_register(name, 0)
    for name, value in registers.items():
        result.set_register(name, value)
    for name in ("CF", "OF", "ZF", "SF", "PF", "AF"):
        result.set_flag(name, True)
    return result


def gadget(raw, address=0x4000):
    return Gadget(address, bytes.fromhex(raw), instructions(raw, address))


def signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def prove(effect, initial, bindings, registers=(), flags=()):
    problem = SolverContext(initial)
    for name, value in bindings.items():
        problem.add_equal(BoolSymbol(name) if type(value) is bool else Symbol(name), value)
    differences = []
    for name, expected in dict(registers).items():
        value = effect.get_register(name)
        assert evaluate(value, bindings) == expected
        differences.append(problem.translator.translate(value) != expected)
    for name, expected in dict(flags).items():
        value = effect.get_flag(name)
        assert evaluate_bool(value, bindings) is expected
        differences.append(problem.translator.translate_bool(value) != expected)
    problem.add_execution_conditions(effect)
    problem._solver.add(z3.Or(*differences))
    assert problem.solve().status is SolveStatus.UNSAT


WIDTHS = [("f6", 8), ("66 f7", 16), ("f7", 32), ("48 f7", 64)]
ALIASES = {8: "al", 16: "ax", 32: "eax", 64: "rax"}
REMAINDERS = {8: "ah", 16: "dx", 32: "edx", 64: "rdx"}


def test_div_implicit_halves_partial_writes_and_high32_clear():
    for prefix, width in WIDTHS:
        initial = state(rax=0x1234000000000065 if width < 64 else 101,
                        rdx=0x5678000000000000 if width < 64 else 0, rbx=10)
        result = run(prefix + " f3", initial)
        assert result.get_register(ALIASES[width]) == 10
        assert result.get_register(REMAINDERS[width]) == 1
        assert result.exception is None
        if width <= 16:
            assert result.get_register("rax") >> 16 == initial.get_register("rax") >> 16
        if width == 16:
            assert result.get_register("rdx") >> 16 == initial.get_register("rdx") >> 16
        if width == 32:
            assert result.get_register("rax") == 10 and result.get_register("rdx") == 1


def test_div64_uses_complete_128bit_dividend():
    result = run("48 f7 f3", state(rax=5, rdx=1, rbx=3))
    assert result.get_register("rax") == ((1 << 64) + 5) // 3
    assert result.get_register("rdx") == ((1 << 64) + 5) % 3


def test_idiv_widths_truncate_to_zero_and_remainder_follows_dividend():
    for prefix, width in WIDTHS:
        for dividend, divisor in [(-9, 2), (9, -2), (-9, -2), (9, 2)]:
            initial = state(rax=dividend & WORD_MASK, rdx=WORD_MASK if dividend < 0 else 0,
                            rbx=divisor & WORD_MASK)
            result = run(prefix + " fb", initial)
            quotient = abs(dividend) // abs(divisor) * (-1 if (dividend < 0) != (divisor < 0) else 1)
            remainder = dividend - quotient * divisor
            assert result.get_register(ALIASES[width]) == quotient & ((1 << width) - 1)
            assert result.get_register(REMAINDERS[width]) == remainder & ((1 << width) - 1)


def test_div_idiv_memory_sources_all_widths():
    for prefix, width in WIDTHS:
        for modrm in ("31", "39"):
            initial = state(rax=100, rdx=0, rcx=0x2000)
            initial.write_memory(0x2000, 7)
            result = run(prefix + " " + modrm, initial)
            assert result.get_register(ALIASES[width]) == 14
            assert result.get_register(REMAINDERS[width]) == 2
            assert result.read_memory(0x2000) == 7


def test_division_fault_is_explicit_and_stops_before_any_write():
    for prefix, width in WIDTHS:
        for modrm in ("f3", "fb"):
            for divisor in (0, 1):
                initial = state(rax=1 << (width - 1), rdx=1, rbx=divisor, rcx=99)
                if width == 8:
                    initial.set_register("ax", 0x8000)
                result = run(prefix + " " + modrm + " b9 00 00 00 00", initial)
                assert result.exception.startswith("#DE")
                assert result.register_bindings() == initial.register_bindings()
                assert result.flag_bindings() == initial.flag_bindings()


def test_idiv_signed_quotient_overflow_and_minimum_boundary():
    for prefix, width in WIDTHS:
        minimum = -(1 << (width - 1))
        for dividend, divisor, fault in [(minimum, -1, True), (minimum, 1, False), (-1, -1, False)]:
            initial = state(rax=dividend & WORD_MASK, rdx=WORD_MASK, rbx=divisor & WORD_MASK)
            assert (run(prefix + " fb", initial).exception is not None) is fault


def test_symbolic_division_matches_concrete_and_z3_all_widths():
    initial = make_symbolic_initial_state()
    for prefix, width in WIDTHS:
        for modrm in ("f3", "fb"):
            low, high, divisor = (WORD_MASK - 8, WORD_MASK, 2) if modrm == "fb" else (100, 0, 7)
            concrete = run(prefix + " " + modrm, state(rax=low, rdx=high, rbx=divisor))
            effect = run(prefix + " " + modrm, initial)
            prove(effect, initial, dict(rax_0=low, rdx_0=high, rbx_0=divisor),
                  {ALIASES[width]: concrete.get_register(ALIASES[width]),
                   REMAINDERS[width]: concrete.get_register(REMAINDERS[width])})


def test_symbolic_divisor_zero_and_quotient_overflow_are_not_normal_sat():
    for raw in ("48 f7 f3 c3", "48 f7 fb c3"):
        constraints = [Equality(Symbol("rax_0"), 3), Equality(Symbol("rdx_0"), 0), Equality(Symbol("rbx_0"), 0)]
        assert solve_gadget(gadget(raw), {}, constraints=constraints).status is SolveStatus.UNSAT
    constraints = [Equality(Symbol("rax_0"), 0), Equality(Symbol("rdx_0"), 1), Equality(Symbol("rbx_0"), 1)]
    assert solve_gadget(gadget("48 f7 f3 c3"), {}, constraints=constraints).status is SolveStatus.UNSAT
    constraints = [Equality(Symbol("rax_0"), 1 << 63), Equality(Symbol("rdx_0"), WORD_MASK),
                   Equality(Symbol("rbx_0"), WORD_MASK)]
    assert solve_gadget(gadget("48 f7 fb c3"), {}, constraints=constraints).status is SolveStatus.UNSAT


def test_division_flags_cannot_reuse_valid_old_values():
    result = run("48 f7 f3", state(rax=100, rbx=7))
    for flag in ("CF", "OF", "ZF", "SF", "PF", "AF"):
        with pytest.raises(UndefinedFlagError):
            result.get_flag(flag)
    with pytest.raises(UndefinedFlagError):
        run("48 f7 f3 0f 94 c1", state(rax=100, rbx=7))


def test_cl_shifts_mask_counts_and_handle_flag_boundaries():
    for prefix, width in [("d2", 8), ("66 d3", 16), ("d3", 32), ("48 d3", 64)]:
        for modrm, name in [("e0", "shl"), ("e8", "shr"), ("f8", "sar")]:
            for count in (0, 1, 2, 8, 16, 31, 32, 63, 64, 255):
                value = (1 << (width - 1)) | 3
                initial = state(rax=value, rcx=count)
                actual = run(prefix + " " + modrm, initial)
                effective = count & (63 if width == 64 else 31)
                expected = ((value << effective) if name == "shl" else
                            ((signed(value, width) if name == "sar" else value) >> effective)) & ((1 << width) - 1)
                assert actual.get_register(ALIASES[width]) == expected
                if effective == 0:
                    assert actual.flag_bindings() == initial.flag_bindings()
                    continue
                assert actual.get_flag("ZF") is (expected == 0)
                if effective == 1:
                    cf = bool(value & (1 << (width - 1 if name == "shl" else 0)))
                    of = False if name == "sar" else bool(expected >> (width - 1)) ^ cf if name == "shl" else bool(value >> (width - 1))
                    assert actual.get_flag("OF") is of
                else:
                    with pytest.raises(UndefinedFlagError):
                        actual.get_flag("OF")
                if name != "sar" and effective >= width:
                    with pytest.raises(UndefinedFlagError):
                        actual.get_flag("CF")
                else:
                    bit = width - effective if name == "shl" else min(effective, width) - 1
                    assert actual.get_flag("CF") is bool(value & (1 << bit))


def test_symbolic_cl_shift_values_match_concrete_and_z3():
    initial = make_symbolic_initial_state()
    for prefix, width in [("d2", 8), ("66 d3", 16), ("d3", 32), ("48 d3", 64)]:
        for modrm in ("e0", "e8", "f8"):
            effect = run(prefix + " " + modrm, initial)
            for count in (0, 1, 3, 31, 64, 255):
                concrete = run(prefix + " " + modrm, state(rax=WORD_MASK, rcx=count))
                prove(effect, initial, dict(rax_0=WORD_MASK, rcx_0=count), {"rax": concrete.get_register("rax")})


def test_symbolic_cl_flag_definedness_excludes_undefined_counts():
    initial = make_symbolic_initial_state()
    effect = run("48 d3 e0 0f 90 c3", initial)
    for count, expected in [(0, SolveStatus.SAT), (1, SolveStatus.SAT), (2, SolveStatus.UNSAT), (65, SolveStatus.SAT)]:
        result = solve_effect(effect, {}, initial_state=initial, constraints=[Equality(Symbol("rcx_0"), count)])
        assert result.status is expected
    snapshot = StateSnapshot.capture(run("d2 e0", initial))
    problem = SolverContext(initial)
    problem.add_equal(snapshot.get_flag("CF"), True)
    problem.add_equal(Symbol("rcx_0"), 8)
    assert problem.solve().status is SolveStatus.UNSAT


def test_memory_shifts_immediate_and_cl_all_widths():
    for prefix, width in [("c0", 8), ("66 c1", 16), ("c1", 32), ("48 c1", 64)]:
        for modrm, name in [("23", "shl"), ("2b", "shr"), ("3b", "sar")]:
            for count_kind in ("immediate", "cl"):
                raw = prefix + " " + modrm + " 02" if count_kind == "immediate" else prefix.replace("c0", "d2").replace("c1", "d3") + " " + modrm
                initial = state(rbx=0x2000, rcx=2)
                initial.write_memory(0x2000, WORD_MASK)
                actual = run(raw, initial)
                mask = (1 << width) - 1
                expected = (mask << 2) & mask if name == "shl" else mask >> 2 if name == "shr" else mask
                assert actual.read_memory(0x2000, width) == expected
                assert initial.read_memory(0x2000) == WORD_MASK


def rotate_reference(value, carry, count, width, name):
    mask = (1 << width) - 1
    value &= mask
    count &= 63 if width == 64 else 31
    size = width + int(name in ("rcl", "rcr"))
    for _ in range(count % size):
        if name in ("rol", "rcl"):
            outgoing = bool(value >> (width - 1))
            value = ((value << 1) | (carry if name == "rcl" else outgoing)) & mask
            carry = outgoing
        else:
            outgoing = bool(value & 1)
            value = (value >> 1) | ((carry if name == "rcr" else outgoing) << (width - 1))
            carry = outgoing
    if count and name in ("rol", "ror"):
        carry = bool(value & 1) if name == "rol" else bool(value >> (width - 1))
    return value, carry


def test_rotates_immediate_cl_widths_and_full_carry_ring_turns():
    for prefix, width in [("c0", 8), ("66 c1", 16), ("c1", 32), ("48 c1", 64)]:
        for modrm, name in [("c0", "rol"), ("c8", "ror"), ("d0", "rcl"), ("d8", "rcr")]:
            for count in (0, 1, 2, 8, 9, 16, 17, 31, 32, 63, 64, 255):
                for carry in (False, True):
                    value = (1 << (width - 1)) | 3
                    initial = state(rax=value, rcx=count)
                    initial.set_flag("CF", carry)
                    expected, cf = rotate_reference(value, carry, count, width, name)
                    raw_forms = (prefix + " " + modrm + f" {count:02x}",
                                 prefix.replace("c0", "d2").replace("c1", "d3") + " " + modrm)
                    for raw in raw_forms:
                        actual = run(raw, initial)
                        assert actual.get_register(ALIASES[width]) == expected
                        assert actual.get_flag("CF") is cf
                        for flag in ("ZF", "SF", "PF", "AF"):
                            assert actual.get_flag(flag) is True
                        effective = count & (63 if width == 64 else 31)
                        if effective == 0:
                            assert actual.get_flag("OF") is True
                        elif effective == 1:
                            of = bool(expected >> (width - 1)) ^ (cf if name in ("rol", "rcl") else bool(expected & (1 << (width - 2))))
                            assert actual.get_flag("OF") is of
                        else:
                            with pytest.raises(UndefinedFlagError):
                                actual.get_flag("OF")


def test_symbolic_rotates_match_concrete_and_z3():
    initial = make_symbolic_initial_state()
    for prefix, width in [("d2", 8), ("66 d3", 16), ("d3", 32), ("48 d3", 64)]:
        for modrm, name in [("c0", "rol"), ("c8", "ror"), ("d0", "rcl"), ("d8", "rcr")]:
            effect = run(prefix + " " + modrm, initial)
            for count in (0, 1, 9, 17, 63, 64):
                expected, cf = rotate_reference(WORD_MASK, True, count, width, name)
                prove(effect, initial, dict(rax_0=WORD_MASK, rcx_0=count, cf_0=True),
                      {ALIASES[width]: expected}, {"CF": cf})


def test_rotate_memory_destinations_preserve_unaffected_flags():
    for raw in ("d0 03", "d0 0b", "d0 13", "d0 1b", "48 d3 13", "66 c1 1b 01"):
        initial = state(rbx=0x2000, rcx=1)
        initial.write_memory(0x2000, 0x80)
        op = lift_instruction(instructions(raw)[0])
        expected, cf = rotate_reference(0x80, True, 1, op.width, op.mnemonic)
        actual = run(raw, initial)
        assert actual.read_memory(0x2000, op.width) == expected
        assert actual.get_flag("CF") is cf
        for flag in ("ZF", "SF", "PF", "AF"):
            assert actual.get_flag(flag) is True


def test_undefined_cf_is_consumed_only_for_effective_nonzero_carry_rotate():
    initial = state(rax=7)
    initial.set_flag("CF", UNDEFINED)
    assert run("c0 d0 00", initial).get_register("al") == 7
    assert run("c0 d0 09", initial).get_register("al") == 7
    assert run("c0 c0 01", initial).get_flag("CF") is False
    with pytest.raises(UndefinedFlagError):
        run("c0 d0 01", initial)
    symbolic = make_symbolic_initial_state()
    symbolic.set_flag("CF", UNDEFINED)
    effect = run("d2 d0", symbolic)
    for count, expected in [(0, SolveStatus.SAT), (9, SolveStatus.SAT), (1, SolveStatus.UNSAT)]:
        result = solve_effect(effect, {}, initial_state=symbolic, constraints=[Equality(Symbol("rcx_0"), count)])
        assert result.status is expected


def test_sign_extension_auxiliaries_partial_writes_flags_and_high32_clear():
    cases = [("66 98", "ax", 0xff80), ("98", "eax", 0xffffff80), ("48 98", "rax", WORD_MASK - 127),
             ("66 99", "dx", 0xffff), ("99", "edx", 0xffffffff), ("48 99", "rdx", WORD_MASK)]
    for raw, register, expected in cases:
        initial = state(rax=WORD_MASK - 127, rdx=0x1234000000000000)
        actual = run(raw, initial)
        assert actual.get_register(register) == expected
        assert actual.flag_bindings() == initial.flag_bindings()
        if raw == "66 99":
            assert actual.get_register("rdx") >> 16 == initial.get_register("rdx") >> 16
        if raw in ("98", "99"):
            assert actual.get_register("rax" if raw == "98" else "rdx") <= 0xffffffff


def test_symbolic_sign_extension_auxiliaries_match_concrete_and_z3():
    initial = make_symbolic_initial_state()
    for raw in ("66 98", "98", "48 98", "66 99", "99", "48 99"):
        concrete = run(raw, state(rax=WORD_MASK - 8, rdx=0))
        effect = run(raw, initial)
        prove(effect, initial, dict(rax_0=WORD_MASK - 8, rdx_0=0),
              {"rax": concrete.get_register("rax"), "rdx": concrete.get_register("rdx")})


def test_bswap_32_64_and_extended_registers_preserve_flags():
    for raw, register, width in [("0f c8", "eax", 32), ("48 0f c8", "rax", 64), ("49 0f c8", "r8", 64)]:
        initial = state(rax=0x1122334455667788, r8=0x1122334455667788)
        actual = run(raw, initial)
        expected = int.from_bytes((0x1122334455667788 & ((1 << width) - 1)).to_bytes(width // 8, "little"), "big")
        assert actual.get_register(register) == expected
        assert actual.flag_bindings() == initial.flag_bindings()
        if width == 32:
            assert actual.get_register("rax") == expected


def test_bsf_bsr_register_memory_widths_and_zero_input_definedness():
    for prefix, width in [("66", 16), ("", 32), ("48", 64)]:
        for opcode, expected in [("bc", 3), ("bd", width - 1)]:
            for modrm in ("c3", "03"):
                initial = state(rax=WORD_MASK, rbx=(1 << (width - 1)) | 8)
                if modrm == "03":
                    initial.set_register("rbx", 0x2000)
                    initial.write_memory(0x2000, (1 << (width - 1)) | 8)
                result = run(f"{prefix} 0f {opcode} {modrm}", initial)
                assert result.get_register(ALIASES[width]) == expected
                assert result.get_flag("ZF") is False
                for flag in ("CF", "OF", "SF", "AF", "PF"):
                    with pytest.raises(UndefinedFlagError):
                        result.get_flag(flag)
            zero = run(f"{prefix} 0f {opcode} c3", state(rax=99, rbx=0))
            assert zero.get_flag("ZF") is True
            with pytest.raises(UndefinedValueError):
                zero.get_register("rax")


def test_bsf_zero_input_can_branch_on_zf_without_consuming_destination():
    result = execute_program(instructions("0f bc c3 74 05 b9 11 00 00 00 ba 22 00 00 00"), state(rbx=0, rcx=7))
    assert result.status is ProgramStatus.COMPLETE
    assert result.paths[0].state.get_register("rcx") == 7


def test_symbolic_bit_scan_destination_goals_require_nonzero_source():
    for opcode, expected in [("bc", 2), ("bd", 63)]:
        part = gadget("48 0f " + opcode + " c3 c3")
        assert solve_gadget(part, {"rax": expected}, constraints=[Equality(Symbol("rbx_0"), (1 << 63) | 4)]).status is SolveStatus.SAT
        assert solve_gadget(part, {"rax": 0}, constraints=[Equality(Symbol("rbx_0"), 0)]).status is SolveStatus.UNSAT


def test_bit_register_indices_are_modulo_width_and_cf_tracks_original_bit():
    for prefix, width in [("66", 16), ("", 32), ("48", 64)]:
        for opcode, expected in [("a3", 2), ("ab", 2), ("b3", 0), ("bb", 0)]:
            result = run(f"{prefix} 0f {opcode} d8", state(rax=2, rbx=width + 1))
            assert result.get_register("rax") == expected
            assert result.get_flag("CF") is True and result.get_flag("ZF") is True
            with pytest.raises(UndefinedFlagError):
                result.get_flag("OF")
        assert run(f"{prefix} 0f ba e8 ff", state(rax=0)).get_register("rax") == 1 << (width - 1)


def test_bit_memory_register_indices_adjust_address_including_negative_offsets():
    for prefix, width in [("66", 16), ("", 32), ("48", 64)]:
        for index in (width + 1, -1):
            initial = state(rbx=0x2000, rcx=index & WORD_MASK)
            target = 0x2000 + (index // width) * (width // 8)
            initial.write_memory(target, 0)
            result = run(f"{prefix} 0f ab 0b", initial)
            assert result.read_memory(target, width) == 1 << (index % width)
            assert result.get_flag("CF") is False


def test_bit_memory_immediate_masks_index_without_inventing_displacement():
    initial = state(rbx=0x2000)
    initial.write_memory(0x2000, 0)
    assert run("0f ba 2b 21", initial).read_memory(0x2000, 32) == 2


def test_symbolic_bit_memory_signed_address_and_store_solve_together():
    initial = make_symbolic_initial_state()
    initial.set_register("rbx", 0x2000)
    initial.write_memory(0x1ff8, 0)
    effect = run("48 0f ab 0b", initial)
    problem = SolverContext(initial)
    problem.add_equal(Symbol("rcx_0"), WORD_MASK)
    problem.add_equal(effect.read_memory(0x1ff8), 1 << 63)
    assert problem.solve().status is SolveStatus.SAT
    problem.add_equal(effect.read_memory(0x1ff8), 0)
    assert problem.solve().status is SolveStatus.UNSAT


def test_popcnt_widths_register_memory_and_all_six_flags():
    for prefix, width in [("66 f3", 16), ("f3", 32), ("f3 48", 64)]:
        for modrm in ("c3", "03"):
            for value in (0, WORD_MASK, 0x8000000000000001):
                initial = state(rax=WORD_MASK, rbx=value)
                if modrm == "03":
                    initial.set_register("rbx", 0x2000)
                    initial.write_memory(0x2000, value)
                result = run(f"{prefix} 0f b8 {modrm}", initial)
                assert result.get_register(ALIASES[width]) == (value & ((1 << width) - 1)).bit_count()
                assert result.get_flag("ZF") is ((value & ((1 << width) - 1)) == 0)
                for flag in ("CF", "OF", "SF", "PF", "AF"):
                    assert result.get_flag(flag) is False


def test_symbolic_popcnt_bswap_and_scan_match_concrete_and_z3():
    initial = make_symbolic_initial_state()
    for raw in ("f3 48 0f b8 c3", "48 0f c8", "48 0f bc c3", "48 0f bd c3"):
        concrete = run(raw, state(rax=0x1122334455667788, rbx=(1 << 63) | 4))
        effect = run(raw, initial)
        prove(effect, initial, dict(rax_0=0x1122334455667788, rbx_0=(1 << 63) | 4),
              {"rax": concrete.get_register("rax")})


def test_xadd_all_widths_register_memory_and_identical_operands():
    for prefix, width in [("", 8), ("66", 16), ("", 32), ("48", 64)]:
        opcode = "c0" if width == 8 else "c1"
        for modrm in ("d8", "18"):
            initial = state(rax=15, rbx=1)
            if modrm == "18":
                initial.set_register("rax", 0x2000)
                initial.write_memory(0x2000, 15)
            result = run(f"{prefix} 0f {opcode} {modrm}", initial)
            assert result.get_register({8: "bl", 16: "bx", 32: "ebx", 64: "rbx"}[width]) == 15
            assert (result.read_memory(0x2000, width) if modrm == "18" else result.get_register(ALIASES[width])) == 16
            assert result.get_flag("AF") is True
        assert run(f"{prefix} 0f {opcode} c0", state(rax=3)).get_register(ALIASES[width]) == 6


def test_cmpxchg_success_failure_accumulator_and_memory_address_capture():
    for prefix, width in [("", 8), ("66", 16), ("", 32), ("48", 64)]:
        opcode = "b0" if width == 8 else "b1"
        for equal in (False, True):
            result = run(f"{prefix} 0f {opcode} cb", state(rax=5 if equal else 9, rbx=5, rcx=7))
            assert result.get_register({8: "bl", 16: "bx", 32: "ebx", 64: "rbx"}[width]) == (7 if equal else 5)
            assert result.get_register(ALIASES[width]) == 5
            assert result.get_flag("ZF") is equal
    initial = state(rax=0x2000, rbx=7)
    initial.write_memory(0x2000, 99)
    result = run("48 0f b1 18", initial)
    assert result.get_register("rax") == 99 and result.read_memory(0x2000) == 99
    initial = state(rax=5, rbx=0x2000, rcx=7)
    initial.write_memory(0x2000, 5)
    result = run("48 0f b1 0b", initial)
    assert result.get_register("rax") == 5 and result.read_memory(0x2000) == 7


def test_symbolic_cmpxchg_xadd_preserve_both_outputs_and_flags():
    initial = make_symbolic_initial_state()
    for raw in ("48 0f c1 d8", "0f c1 d8", "48 0f b1 cb", "0f b1 cb"):
        for accumulator in (5, 9):
            concrete = run(raw, state(rax=accumulator, rbx=5, rcx=7))
            effect = run(raw, initial)
            prove(effect, initial, dict(rax_0=accumulator, rbx_0=5, rcx_0=7),
                  {name: concrete.get_register(name) for name in ("rax", "rbx", "rcx")},
                  {name: concrete.get_flag(name) for name in ("CF", "OF", "ZF", "SF", "PF", "AF")})


def test_af_arithmetic_widths_carry_input_and_inc_dec_preserved_cf():
    for prefix, width in [("", 8), ("66", 16), ("", 32), ("48", 64)]:
        forms = [(0x00, False, False), (0x28, True, False), (0x10, False, True),
                 (0x18, True, True), (0x38, True, False)]
        for opcode, subtract, carry in forms:
            for left, right in [(15, 0), (15, 1), (0, 1), (16, 15), (0, 0)]:
                result = run(f"{prefix} {opcode + int(width != 8):02x} d8", state(rax=left, rbx=right))
                expected = (left & 15) < (right & 15) + carry if subtract else (left & 15) + (right & 15) + carry > 15
                assert result.get_flag("AF") is expected
    for raw, value, expected in [("fe c0", 15, True), ("fe c8", 16, True), ("f6 d8", 1, True), ("f6 d8", 16, False)]:
        result = run(raw, state(rax=value))
        assert result.get_flag("AF") is expected
        if raw.startswith("fe"):
            assert result.get_flag("CF") is True


def test_symbolic_af_proves_low_nibble_reference_for_all_8bit_inputs():
    initial = make_symbolic_initial_state()
    for raw, subtract, carry in [("00 d8", False, False), ("28 d8", True, False),
                                 ("10 d8", False, True), ("18 d8", True, True)]:
        effect = run(raw, initial)
        problem = SolverContext(initial)
        left = z3.ZeroExt(1, z3.Extract(3, 0, problem.translator.translate(Symbol("rax_0"))))
        right = z3.ZeroExt(1, z3.Extract(3, 0, problem.translator.translate(Symbol("rbx_0"))))
        cf = problem.translator.translate_bool(BoolSymbol("cf_0"))
        extra = z3.If(cf, z3.BitVecVal(1, 5, ctx=problem.translator.context), z3.BitVecVal(0, 5, ctx=problem.translator.context)) if carry else 0
        expected = z3.ULT(left, right + extra) if subtract else z3.UGT(left + right + extra, 15)
        problem._solver.add(problem.translator.translate_bool(effect.get_flag("AF")) != expected)
        assert problem.solve().status is SolveStatus.UNSAT


def test_remaining_memory_alu_forms_update_af_and_preserve_input_memory():
    for raw in ("48 01 03", "48 03 03", "48 83 03 01", "48 29 03", "48 2b 03", "48 83 2b 01",
                "48 21 03", "48 0b 03", "48 31 03", "48 ff 03", "48 ff 0b", "48 f7 1b", "48 f7 13"):
        initial = state(rax=1, rbx=0x2000)
        initial.write_memory(0x2000, 15)
        result = run(raw, initial)
        assert result.exception is None and initial.read_memory(0x2000) == 15
        if raw in ("48 01 03", "48 83 03 01", "48 ff 03"):
            assert result.read_memory(0x2000) == 16 and result.get_flag("AF") is True


def test_integer_illegal_widths_encodings_and_prefixes_are_explicitly_rejected():
    for raw in ("66 0f c8", "f0 0f c1 18", "67 d3 20", "f2 f3 0f b8 c3", "f3 0f bc c3", "0f a2", "d9 e8"):
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(instructions(raw)[0])
    item = instructions("48 f7 f3")[0]
    with pytest.raises(UnsupportedInstruction):
        lift_instruction(replace(item, operands=(Operand("immediate", 64, immediate=2),)))
    item = instructions("d3 e0")[0]
    with pytest.raises(UnsupportedInstruction):
        lift_instruction(replace(item, operands=(item.operands[0], Operand("register", 8, register="bl"))))
    invalid = [IntegerOp("div", src=1, width=64), IntegerOp("bswap", dst="ax", width=16),
               IntegerOp("bt", dst="al", src="bl", width=8),
               IntegerOp("bsf", dst="eax", src=1, width=32),
               IntegerOp("rol", dst="rax", src=256, width=64),
               IntegerOp("add", dst="rax", src=1 << 40, width=64)]
    for operation in invalid:
        with pytest.raises(ValueError):
            execute([operation], make_symbolic_initial_state())


def test_finite_single_candidate_sat_joins_initial_constraints_and_final_goal():
    part = gadget("48 83 c0 01 c3")
    constraints = [Equality(Symbol("rax_0"), 6)]
    found = search_sequences([part], {"rax": 7}, max_depth=1, constraints=constraints)
    assert found.status is SearchStatus.FOUND and found.sequence == (part,)
    assert found.tested_candidates == 1 and found.solve_result.status is SolveStatus.SAT
    assert found.solve_result.goals_satisfied is True
    assert search_sequences([part], {"rax": 8}, max_depth=1, constraints=constraints).status is SearchStatus.NOT_FOUND


def test_finite_multiple_candidates_only_one_sat_preserves_input_order_and_state():
    initial = make_symbolic_initial_state()
    initial.set_register("rsi", 0x2000)
    initial.write_memory(0x2000, 7)
    bad = gadget("48 c7 06 09 00 00 00 48 8b 06 c3", 0x4000)
    good = gadget("48 8b 06 c3", 0x4100)
    found = search_sequences([bad, good], {"rax": 7}, max_depth=1, initial_state=initial)
    assert found.status is SearchStatus.FOUND and found.sequence == (good,)
    assert found.tested_candidates == 2 and found.solve_result.goals_satisfied is True
    assert initial.read_memory(0x2000) == 7


def test_finite_all_unsat_is_bounded_not_found_and_stops():
    parts = [gadget("b8 01 00 00 00 c3", 0x4000), gadget("b8 02 00 00 00 c3", 0x4100)]
    result = search_sequences(parts, {"rax": 3}, max_depth=2)
    assert result.status is SearchStatus.NOT_FOUND
    assert result.tested_candidates == 6 and result.unknown_candidates == 0
    assert result.sequence == () and result.solve_result is None


def test_finite_unknown_does_not_become_unsat_or_not_found(monkeypatch):
    parts = [gadget("b8 01 00 00 00 c3", 0x4000), gadget("b8 02 00 00 00 c3", 0x4100)]
    outcomes = iter([SolveResult(SolveStatus.UNKNOWN, reason_unknown="controlled resource limit"),
                     SolveResult(SolveStatus.UNSAT)])
    monkeypatch.setattr(SolverContext, "solve", lambda self: next(outcomes))
    result = search_sequences(parts, {"rax": 3}, max_depth=1)
    assert result.status is SearchStatus.INCONCLUSIVE
    assert result.tested_candidates == 2 and result.unknown_candidates == 1


def test_finite_memory_transitions_and_condition_targets_share_one_context():
    initial = make_symbolic_initial_state()
    initial.set_register("rsi", 0x2000)
    initial.write_memory(0x2000, 7)
    part = gadget("48 8b 0e 48 83 c1 01 48 89 0e 48 83 f9 08 0f 94 c0 c3")
    effect = run("48 8b 0e 48 83 c1 01 48 89 0e 48 83 f9 08 0f 94 c0", initial)
    stored = effect.read_memory(0x2000)
    found = search_sequences([part], {"rcx": 8, "al": 1}, max_depth=1, initial_state=initial,
                             constraints=[Equality(stored, 8)])
    assert found.status is SearchStatus.FOUND
    assert search_sequences([part], {"al": 1}, max_depth=1, initial_state=initial,
                            constraints=[Equality(stored, 9)]).status is SearchStatus.NOT_FOUND
    assert search_sequences([part], {"al": 0}, max_depth=1, initial_state=initial).status is SearchStatus.NOT_FOUND


def test_finite_cmov_ite_goal_depends_on_initial_flags_and_operand_constraints():
    false_choice = gadget("48 39 d8 48 0f 45 ca c3", 0x4000)
    true_choice = gadget("48 39 d8 48 0f 44 ca c3", 0x4100)
    constraints = [Equality(Symbol("rax_0"), 5), Equality(Symbol("rbx_0"), 5),
                   Equality(Symbol("rcx_0"), 0), Equality(Symbol("rdx_0"), 7)]
    found = search_sequences([false_choice, true_choice], {"rcx": 7}, max_depth=1, constraints=constraints)
    assert found.status is SearchStatus.FOUND and found.sequence == (true_choice,)
    assert found.tested_candidates == 2


def test_finite_division_faults_never_found_and_do_not_block_later_valid_candidate():
    zero = gadget("31 db 48 f7 f3 c3", 0x4000)
    overflow = gadget("b8 00 00 00 00 ba 01 00 00 00 bb 01 00 00 00 48 f7 f3 c3", 0x4100)
    good = gadget("b8 07 00 00 00 c3", 0x4200)
    result = search_sequences([zero, overflow], {"rax": 7}, max_depth=2)
    assert result.status is SearchStatus.NOT_FOUND and result.tested_candidates == 6
    found = search_sequences([zero, overflow, good], {"rax": 7}, max_depth=1)
    assert found.status is SearchStatus.FOUND and found.sequence == (good,) and found.tested_candidates == 3
    solved = solve_sequence([zero], {"rax": 7}, diagnose=True)
    assert solved.status is SolveStatus.UNSAT and solved.diagnostics.status is SolveStatus.UNSAT


def test_finite_undefined_symbolic_flags_do_not_produce_false_found():
    part = gadget("48 d3 e0 0f 90 c3 c3")
    constraints = [Equality(Symbol("rcx_0"), 2)]
    result = search_sequences([part], {"bl": 1}, max_depth=1, constraints=constraints)
    assert result.status is SearchStatus.NOT_FOUND
    assert search_sequences([part], max_depth=1, constraints=[Equality(Symbol("rcx_0"), 1)]).status is SearchStatus.FOUND


def test_completed_program_path_targets_include_original_branch_constraints():
    initial = make_symbolic_initial_state()
    initial.set_register("rcx", 0)
    result = execute_program(instructions("48 39 d8 74 05 b9 01 00 00 00 ba 02 00 00 00"), initial)
    assert result.status is ProgramStatus.COMPLETE and len(result.paths) == 2
    taken = next(path for path in result.paths if path.state.get_register("rcx") == 0)
    fallthrough = next(path for path in result.paths if path.state.get_register("rcx") == 1)
    same = [Equality(Symbol("rax_0"), 5), Equality(Symbol("rbx_0"), 5)]
    different = [Equality(Symbol("rax_0"), 5), Equality(Symbol("rbx_0"), 6)]
    assert solve_program_path(taken, {"rcx": 0}, constraints=same).status is SolveStatus.SAT
    assert solve_program_path(taken, {"rcx": 0}, constraints=different).status is SolveStatus.UNSAT
    assert solve_program_path(fallthrough, {"rcx": 1}, constraints=same).status is SolveStatus.UNSAT


def test_program_limits_unresolved_and_division_exceptions_cannot_solve_as_complete():
    initial = state(rax=5, rbx=5)
    limited = execute_program(instructions("39 d8 74 fc"), initial, max_steps=2)
    assert limited.status is ProgramStatus.INCOMPLETE
    assert solve_program_path(limited.paths[0], {"rax": 5}).status is SolveStatus.UNKNOWN
    unresolved = execute_program(instructions("74 20"), initial)
    assert solve_program_path(unresolved.paths[0]).status is SolveStatus.UNKNOWN
    fault = execute_program(instructions("31 db 48 f7 f3 b9 01 00 00 00"), initial)
    assert fault.status is ProgramStatus.UNSAT and fault.paths == ()
