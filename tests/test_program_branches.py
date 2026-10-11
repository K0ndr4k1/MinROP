"""Instruction semantics and bounded branches over explicitly supplied code."""
from dataclasses import replace

import pytest

from minirop.constraints import Equality
from minirop.disasm import Disassembler
from minirop.executor import execute, execute_gadget
from minirop.flags import UNDEFINED, UndefinedFlagError, CONDITION_ALIASES
from minirop.gadget import Gadget, Operand
from minirop.ir import Jcc, Shl, Shr
from minirop.program import execute_program, ProgramStatus
from minirop.semantics import lift_instruction, UnsupportedInstruction
from minirop.solver import SolverContext, SolveResult, SolveStatus
from minirop.state import CPUState, make_symbolic_initial_state
from minirop.symbolic import (BoolSymbol, Ite, Symbol, WORD_MASK,
                              compare_values, evaluate, evaluate_bool)


def code(hexadecimal, address=0x4000):
    raw = bytes.fromhex(hexadecimal)
    instructions = Disassembler().disassemble(raw, address)
    assert sum(i.size for i in instructions) == len(raw)
    return instructions


def state(**registers):
    result = CPUState()
    for name, value in registers.items():
        result.set_register(name, value)
    return result


def run(hexadecimal, initial):
    return execute(tuple(lift_instruction(i) for i in code(hexadecimal)), initial)


def problem(path):
    result = SolverContext(path.state)
    for equality in path.constraints:
        result.add(equality)
    return result


def branch_program():
    return code("74 05 b8 11 00 00 00 ba 22 00 00 00")


def reference_shift(original, count, width, left):
    count &= 63 if width == 64 else 31
    operand = original & ((1 << width) - 1)
    result = ((operand << count) if left else (operand >> count)) & ((1 << width) - 1)
    expected = dict(ZF=result == 0, SF=bool(result & (1 << (width - 1))),
                    PF=(result & 255).bit_count() % 2 == 0)
    if count < width:
        expected["CF"] = bool(operand & (1 << (width - count if left else count - 1)))
    if count == 1:
        expected["OF"] = expected["SF"] != expected["CF"] if left else bool(operand >> (width - 1))
    return result, expected


def test_cmov32_clears_upper_half_in_both_ite_branches():
    old, source = 0xaabbccdd12345678, 0xdeadbeef87654321
    for flag in (True, False):
        initial = state(rax=old, rbx=source)
        initial.set_flag("ZF", flag)
        actual = run("0f 44 c3", initial)
        assert actual.get_register("rax") == (source if flag else old) & 0xffffffff
        assert actual.flag_bindings() == initial.flag_bindings()
        assert initial.get_register("rax") == old
    initial = make_symbolic_initial_state()
    value = run("0f 44 c3", initial).get_register("rax")
    assert isinstance(value, Ite)
    for flag in (True, False):
        expected = (source if flag else old) & 0xffffffff
        assert evaluate(value, dict(rax_0=old, rbx_0=source, zf_0=flag)) == expected
        context = SolverContext()
        for lhs, rhs in [(Symbol("rax_0"), old), (Symbol("rbx_0"), source),
                         (BoolSymbol("zf_0"), flag)]:
            context.add_equal(lhs, rhs)
        context.add_equal(value, expected ^ (1 << 32))
        assert context.solve().status is SolveStatus.UNSAT


def test_shl_flags_match_shifted_out_bits_and_overflow_all_widths():
    for register, width in [("al", 8), ("ax", 16), ("eax", 32), ("rax", 64)]:
        for original in (1, (1 << (width - 1)), (1 << (width - 2)), (1 << width) - 1):
            for count in (1, 2, width - 1):
                expected_result, expected = reference_shift(original, count, width, True)
                result = execute([Shl(register, count)], state(rax=original))
                assert result.get_register(register) == expected_result
                for name, flag in expected.items():
                    assert result.get_flag(name) is flag
                for name in ("AF",) + (() if count == 1 else ("OF",)):
                    with pytest.raises(UndefinedFlagError):
                        result.get_flag(name)
    # Narrow counts can reach/exceed the operand width; CF is then undefined.
    for register, width in [("al", 8), ("ax", 16)]:
        for count in (width, width + 1, 31):
            result = execute([Shl(register, count)], state(rax=WORD_MASK))
            assert result.get_register(register) == 0 and result.get_flag("ZF") is True
            for name in ("CF", "OF", "AF"):
                with pytest.raises(UndefinedFlagError):
                    result.get_flag(name)


def test_shr_flags_match_original_msb_and_last_bit_all_widths():
    for register, width in [("al", 8), ("ax", 16), ("eax", 32), ("rax", 64)]:
        for original in (1, 2, 1 << (width - 1), (1 << width) - 1):
            for count in (1, 2, width - 1):
                expected_result, expected = reference_shift(original, count, width, False)
                result = execute([Shr(register, count)], state(rax=original))
                assert result.get_register(register) == expected_result
                for name, flag in expected.items():
                    assert result.get_flag(name) is flag
                for name in ("AF",) + (() if count == 1 else ("OF",)):
                    with pytest.raises(UndefinedFlagError):
                        result.get_flag(name)
    for register, width in [("al", 8), ("ax", 16)]:
        for count in (width, width + 1):
            result = execute([Shr(register, count)], state(rax=WORD_MASK))
            assert result.get_register(register) == 0 and result.get_flag("ZF") is True
            with pytest.raises(UndefinedFlagError):
                result.get_flag("CF")


def test_masked_zero_shifts_preserve_even_unknown_flags():
    initial = state(rax=WORD_MASK)
    initial.set_flag("CF", UNDEFINED)
    initial.set_flag("ZF", BoolSymbol("original_zero"))
    initial.set_flag("OF", True)
    for operation in (Shl, Shr):
        for register, counts in [("al", (0, 32, 64)), ("ax", (0, 32)),
                                 ("eax", (0, 32)), ("rax", (0, 64, 128))]:
            for count in counts:
                result = execute([operation(register, count)], initial)
                assert result.flag_bindings() == initial.flag_bindings()
                expected = 0xffffffff if register == "eax" else WORD_MASK
                assert result.get_register("rax") == expected
    # Count 33/65 is effective count one, so OF is defined, rather than undefined.
    for register, count in [("al", 33), ("ax", 33), ("eax", 33), ("rax", 65)]:
        assert execute([Shr(register, count)], initial).get_flag("OF") is True


def test_symbolic_shift_values_flags_agree_with_concrete_and_z3():
    import z3
    for operation, left in [(Shl, True), (Shr, False)]:
        for register, width in [("al", 8), ("ax", 16), ("eax", 32), ("rax", 64)]:
            for count in (1, 2):
                original = (1 << (width - 1)) | 3
                initial = make_symbolic_initial_state()
                symbolic = execute([operation(register, count)], initial)
                expected_value, expected = reference_shift(original, count, width, left)
                assert evaluate(symbolic.get_register(register), {"rax_0": original}) == expected_value
                context = SolverContext()
                context.add_equal(Symbol("rax_0"), original)
                mismatches = [context.translator.translate(symbolic.get_register(register)) != expected_value]
                for name, answer in expected.items():
                    expression = symbolic.get_flag(name)
                    assert evaluate_bool(expression, {"rax_0": original}) is answer
                    mismatches.append(context.translator.translate_bool(expression) != answer)
                context._solver.add(z3.Or(*mismatches))
                assert context.solve().status is SolveStatus.UNSAT


def test_jcc_rel8_rel32_aliases_and_own_instruction_end():
    names = ("o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g")
    for index, name in enumerate(names):
        for raw, displacement in [(bytes([0x70 + index, 0xf9]), -7),
                                  (bytes([0x0f, 0x80 + index]) + (0x1234).to_bytes(4, "little"), 0x1234),
                                  (bytes([0x0f, 0x80 + index]) + (-0x1234).to_bytes(4, "little", signed=True), -0x1234)]:
            instruction = code(raw.hex(), 0x5000)[0]
            lifted = lift_instruction(instruction)
            assert lifted == Jcc(name, 0x5000 + len(raw) + displacement, 0x5000 + len(raw))
            for alias, canonical in CONDITION_ALIASES.items():
                if canonical == name:
                    assert lift_instruction(replace(instruction, mnemonic="j" + alias)) == lifted
    instructions = code("b8 01 00 00 00 74 f9", 0x6000)
    assert lift_instruction(instructions[-1]).target == 0x6000
    instruction = code("74 fe", WORD_MASK - 1)[0]
    assert lift_instruction(instruction) == Jcc("e", WORD_MASK - 1, 0)
    valid = code("74 05")[0]
    for invalid in [replace(valid, size=1), replace(valid, address=True), replace(valid, mnemonic="jne"),
                    replace(valid, operands=(Operand("immediate", 64, immediate=0x1234),)),
                    replace(valid, operands=(Operand("register", 64, register="rax"),))]:
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(invalid)
    for raw in ("66 74 00", "67 74 00", "e3 00"):
        with pytest.raises(UnsupportedInstruction):
            lift_instruction(code(raw)[0])
    prefixed = code("66 74 00")[0]
    with pytest.raises(UnsupportedInstruction, match="prefix"):
        lift_instruction(replace(prefixed, prefixes=(),
                                 operands=(replace(prefixed.operands[0], width=64),)))
    instructions = code("74 00 c3")
    with pytest.raises(UnsupportedInstruction, match="fixed-program"):
        execute_gadget(Gadget(0x4000, bytes.fromhex("74 00 c3"), instructions), make_symbolic_initial_state())


def test_concrete_jcc_chooses_successor_and_preserves_gprs_flags():
    for flag, successor in [(True, 0x4007), (False, 0x4002)]:
        initial = state(rax=9, rdx=8)
        initial.set_flag("ZF", flag)
        limited = execute_program(branch_program(), initial, max_steps=1)
        assert limited.status is ProgramStatus.INCOMPLETE
        path, = limited.paths
        assert path.address == path.state.get_register("rip") == successor and path.steps == 1
        assert path.state.get_register("rax") == 9 and path.state.get_register("rdx") == 8
        assert path.state.flag_bindings() == initial.flag_bindings()
        complete = execute_program(branch_program(), initial)
        assert complete.status is ProgramStatus.COMPLETE
        path, = complete.paths
        assert path.address == 0x400c
        assert path.state.get_register("rax") == (9 if flag else 0x11)
        assert path.state.get_register("rdx") == 0x22
        assert path.state.flag_bindings() == initial.flag_bindings()
        assert initial.get_register("rax") == 9


def test_signed_unsigned_jcc_differs_after_same_cmp():
    for branch, expected in [("7f", 1), ("77", 0)]:
        instructions = code("b8 ff ff ff ff 83 f8 01 " + branch + " 05 b9 01 00 00 00 ba 02 00 00 00")
        result = execute_program(instructions, state(rcx=0, rdx=0))
        assert result.status is ProgramStatus.COMPLETE
        assert result.paths[0].state.get_register("rcx") == expected
    # A logical shift produces flags that a following Jcc can consume.
    result = execute_program(code("c0 e0 01 74 05 b9 01 00 00 00 ba 02 00 00 00"),
                             state(rax=0x80, rcx=0, rdx=0))
    assert result.status is ProgramStatus.COMPLETE and result.paths[0].state.get_register("rcx") == 0


def test_symbolic_jcc_forks_two_constrained_exclusive_successors():
    instructions = code("48 39 d8 74 05 b9 11 00 00 00 ba 22 00 00 00")
    initial = state(rax=Symbol("left"), rbx=Symbol("right"), rcx=7, rdx=0)
    result = execute_program(instructions, initial)
    assert result.status is ProgramStatus.COMPLETE and len(result.paths) == 2
    equal = compare_values("eq", Symbol("left"), Symbol("right"))
    for path in result.paths:
        took = path.state.get_register("rcx") == 7
        context = problem(path)
        assert context.solve().status is SolveStatus.SAT
        context.add_equal(equal, not took)
        assert context.solve().status is SolveStatus.UNSAT
        assert evaluate_bool(path.path_condition, dict(left=1, right=1 if took else 2)) is True
        assert path.state.get_register("rdx") == 0x22
    both = SolverContext()
    for path in result.paths:
        both.add_equal(path.path_condition, True)
    assert both.solve().status is SolveStatus.UNSAT


def test_unreachable_branch_pruned_with_shared_constraints_and_memory():
    # E/NE exhaust the comparison; the UD2 fallthrough can never execute.
    instructions = (code("48 39 d8 74 0b 75 09 0f 0b") + code("b9 07 00 00 00", 0x4010))
    initial = state(rax=Symbol("left"), rbx=Symbol("right"), rcx=0)
    result = execute_program(instructions, initial)
    assert result.status is ProgramStatus.COMPLETE and len(result.paths) == 2
    constrained = execute_program(instructions, initial,
                                  constraints=[Equality(Symbol("left"), Symbol("right"))])
    assert constrained.status is ProgramStatus.COMPLETE and len(constrained.paths) == 1
    assert constrained.paths[0].state.get_register("rcx") == 7
    assert execute_program(instructions, initial, constraints=[Equality(False, True)]).status is ProgramStatus.UNSAT
    # Input-memory bindings participate in the very same branch feasibility checks.
    initial = state(rdi=0x2000, rcx=0, rdx=0)
    initial.write_memory(0x2000, 5)
    instructions = code("48 8b 07 48 83 f8 05 74 05 b9 01 00 00 00 ba 02 00 00 00")
    result = execute_program(instructions, initial)
    assert result.status is ProgramStatus.COMPLETE and len(result.paths) == 1
    assert result.paths[0].state.get_register("rcx") == 0


def test_forked_register_memory_and_flags_are_independent():
    instructions = (code("74 0e c7 07 01 00 00 00 83 c0 01 75 13")
                    + code("c7 07 02 00 00 00 83 e8 01 75 05", 0x4010)
                    + code("bb 63 00 00 00", 0x4020))
    initial = state(rax=0, rdi=0x2000, rcx=10)
    initial.set_flag("ZF", BoolSymbol("choice"))
    initial.write_memory(0x2000, 0)
    result = execute_program(instructions, initial)
    assert result.status is ProgramStatus.COMPLETE and len(result.paths) == 2
    paths = {path.state.read_memory(0x2000, 32): path for path in result.paths}
    first, second = paths[1].state, paths[2].state
    assert first.get_register("rax") == 1 and second.get_register("rax") == 0xffffffff
    assert first.get_flag("CF") is False and second.get_flag("CF") is True
    assert first.get_flag("SF") is False and second.get_flag("SF") is True
    first.set_register("rcx", 999)
    first.set_flag("CF", True)
    first.store_memory(0x2000, 99, 32)
    assert second.get_register("rcx") == 10 and second.read_memory(0x2000, 32) == 2
    assert initial.get_register("rax") == 0 and initial.read_memory(0x2000) == 0
    assert initial.get_flag("ZF") == BoolSymbol("choice")
    for number, path in paths.items():
        context = problem(path)
        context.add_equal(BoolSymbol("choice"), number == 2)
        assert context.solve().status is SolveStatus.SAT


def test_step_bound_is_incomplete_and_exact_terminal_bound_completes():
    initial = state(rax=0)
    initial.set_flag("ZF", True)
    result = execute_program(code("74 fe"), initial, max_steps=3)
    assert result.status is ProgramStatus.INCOMPLETE
    path, = result.paths
    assert path.reason == "step_limit" and path.steps == 3 and path.address == 0x4000
    assert problem(path).solve().status is SolveStatus.SAT
    exact = execute_program(code("b8 01 00 00 00"), initial, max_steps=1)
    assert exact.status is ProgramStatus.COMPLETE and exact.paths[0].steps == 1
    for options in ({"max_steps": 0}, {"max_paths": True}, {"timeout_ms": 0}):
        with pytest.raises(ValueError):
            execute_program(code("74 fe"), initial, **options)
    for instructions in ((), code("74 fe") * 2, (replace(code("74 fe")[0], size=1),)):
        with pytest.raises(ValueError):
            execute_program(instructions, initial)


def test_active_path_bound_preserves_feasible_unfinished_frontier():
    initial = state(rax=9, rdx=8)
    initial.set_flag("ZF", BoolSymbol("choice"))
    result = execute_program(branch_program(), initial, max_paths=1)
    assert result.status is ProgramStatus.INCOMPLETE and len(result.paths) == 2
    assert {path.address for path in result.paths} == {0x4002, 0x4007}
    for path in result.paths:
        assert path.status is ProgramStatus.INCOMPLETE and path.reason == "path_limit"
        assert path.steps == 1 and path.state.get_register("rax") == 9
        assert problem(path).solve().status is SolveStatus.SAT
    enough = execute_program(branch_program(), initial, max_paths=2)
    assert enough.status is ProgramStatus.COMPLETE and len(enough.paths) == 2


def test_missing_targets_undefined_flags_and_other_control_flow_are_rejected():
    initial = state(rax=0)
    for flag in (True, False):
        initial.set_flag("ZF", flag)
        for raw in ("74 05 b8 01 00 00 00", "74 01 b8 01 00 00 00"):
            result = execute_program(code(raw), initial)
            assert result.status is ProgramStatus.UNSUPPORTED
            assert result.paths[0].reason == "unresolved_target"
    for raw in ("ff e0", "eb fe", "e8 00 00 00 00", "c3"):
        result = execute_program(code(raw), initial)
        assert result.status is ProgramStatus.UNSUPPORTED
    for value in (None, UNDEFINED):
        flags = state(rax=0)
        if value is not None:
            flags.set_flag("ZF", value)
        result = execute_program(code("74 fe"), flags)
        assert result.status is ProgramStatus.UNSUPPORTED and "flag" in result.paths[0].reason
    partial = execute_program(code("b8 01 00 00 00") + code("b9 02 00 00 00", 0x4010), initial)
    assert partial.status is ProgramStatus.UNSUPPORTED and partial.paths[0].reason == "unresolved_target"
    assert execute_program(code("b8 01 00 00 00"), initial, entry=0x5000).status is ProgramStatus.UNSUPPORTED


def test_solver_unknown_preserves_paths_as_incomplete(monkeypatch):
    original = SolverContext.solve
    calls = 0

    def uncertain_branches(context):
        nonlocal calls
        calls += 1
        return original(context) if calls == 1 else SolveResult(SolveStatus.UNKNOWN, reason_unknown="timeout")

    initial = state(rax=0, rdx=0)
    initial.set_flag("ZF", BoolSymbol("choice"))
    monkeypatch.setattr(SolverContext, "solve", uncertain_branches)
    result = execute_program(branch_program(), initial)
    assert result.status is ProgramStatus.INCOMPLETE and len(result.paths) == 2
    assert all(path.reason == "solver_unknown" and path.steps == 1 for path in result.paths)
    monkeypatch.setattr(SolverContext, "solve", lambda context: SolveResult(SolveStatus.UNKNOWN))
    result = execute_program(branch_program(), initial)
    assert result.status is ProgramStatus.INCOMPLETE and result.paths[0].steps == 0
