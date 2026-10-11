import pytest

from dataclasses import replace

from minirop import CodeRegion, ELFImage, GadgetScanner
from minirop.disasm import Disassembler
from minirop.executor import NonConcreteValue, execute, execute_gadget
from minirop.ir import Add, Leave, Mov, Pop, Ret, Sub, Xor
from minirop.semantics import UnsupportedInstruction, lift_gadget, lift_instruction
from minirop.state import CPUState, InvalidRegister, MemoryAccessError, WORD_MASK


def instruction(hex_bytes):
    decoded = Disassembler().disassemble(bytes.fromhex(hex_bytes), 0x4000)
    assert len(decoded) == 1
    return decoded[0]


def gadget(hex_bytes):
    data = bytes.fromhex(hex_bytes)
    return next(g for g in GadgetScanner(max_gadget_bytes=64, max_instructions=12).scan(
        [CodeRegion("test", 0x4000, 0, data)]) if g.address == 0x4000 and g.bytes == data)


def stack(*values):
    state = CPUState()
    state.set_register("rsp", 0x1000)
    for i, value in enumerate(values):
        state.write_memory(0x1000 + 8 * i, value)
    return state


def test_pop():
    op = lift_instruction(instruction("5f"))
    assert op == Pop("rdi")
    state = stack(0x41414141)
    result = execute([op], state)
    assert result.get_register("rdi") == 0x41414141
    assert result.get_register("rsp") == 0x1008
    assert state.get_register("rsp") == 0x1000


def test_ret():
    op = lift_instruction(instruction("c3"))
    assert op == Ret()
    result = execute([op], stack(0x401000))
    assert result.get_register("rip") == 0x401000
    assert result.get_register("rsp") == 0x1008


def test_pop_ret():
    state = stack(0x41414141, 0x401000)
    result = execute_gadget(gadget("5f c3"), state)
    assert result.get_register("rdi") == 0x41414141
    assert result.get_register("rip") == 0x401000
    assert result.get_register("rsp") == 0x1010
    assert state.get_register("rsp") == 0x1000


def test_multiple_pop():
    result = execute_gadget(gadget("5f 5e c3"), stack(11, 22, 0x401000))
    assert result.get_register("rdi") == 11
    assert result.get_register("rsi") == 22
    assert result.get_register("rip") == 0x401000
    assert result.get_register("rsp") == 0x1018


def test_pop_rsp_pivots_before_ret():
    state = stack(0x2000)
    state.write_memory(0x2000, 0x401000)
    result = execute_gadget(gadget("5c c3"), state)
    assert result.get_register("rsp") == 0x2008
    assert result.get_register("rip") == 0x401000


def test_memory_failure_does_not_mutate_input():
    state = stack(42)
    state.set_register("rdi", 7)
    with pytest.raises(MemoryAccessError):
        execute_gadget(gadget("5f c3"), state)
    assert state.get_register("rdi") == 7
    assert state.get_register("rsp") == 0x1000


@pytest.mark.parametrize("code", ["66 5f", "8f 04 24", "c2 10 00", "48 87 07", "48 0f a4 d8 01"])
def test_unsupported(code):
    with pytest.raises(UnsupportedInstruction):
        lift_instruction(instruction(code))


@pytest.mark.parametrize("code, expected", [
    ("48 89 f8", Mov("rax", "rdi")),
    ("48 c7 c0 ff ff ff ff", Mov("rax", -1)),
    ("48 b8 88 77 66 55 44 33 22 11", Mov("rax", 0x1122334455667788)),
    ("48 83 c0 08", Add("rax", 8)),
    ("48 83 ef 01", Sub("rdi", 1)),
    ("48 31 c0", Xor("rax", "rax")),
    ("c9", Leave()),
])
def test_lift_operations(code, expected):
    assert lift_instruction(instruction(code)) == expected


def test_mov_register_gadget():
    state = stack(0x401000)
    state.set_register("rdi", 0x123456789abcdef0)
    result = execute_gadget(gadget("48 89 f8 c3"), state)
    assert result.get_register("rax") == state.get_register("rdi")
    assert result.get_register("rdi") == state.get_register("rdi")
    assert result.get_register("rip") == 0x401000


@pytest.mark.parametrize("code, expected", [
    ("48 c7 c0 ff ff ff ff c3", WORD_MASK),
    ("48 b8 88 77 66 55 44 33 22 11 c3", 0x1122334455667788),
    ("48 b8 ff ff ff ff ff ff ff ff c3", WORD_MASK),
])
def test_mov_immediate(code, expected):
    result = execute_gadget(gadget(code), stack(0x401000))
    assert result.get_register("rax") == expected


@pytest.mark.parametrize("code, register, initial, expected", [
    ("48 83 c0 08", "rax", 4, 12),
    ("48 83 ef 01", "rdi", 4, 3),
    ("48 83 c0 01", "rax", WORD_MASK, 0),
    ("48 83 ef 01", "rdi", 0, WORD_MASK),
    ("48 83 c0 ff", "rax", 5, 4),
    ("48 83 ef ff", "rdi", 5, 6),
    ("48 05 ff ff ff ff", "rax", 5, 4),
    ("49 83 c0 08", "r8", 1, 9),
])
def test_arithmetic(code, register, initial, expected):
    state = CPUState()
    state.set_register(register, initial)
    result = execute([lift_instruction(instruction(code))], state)
    assert result.get_register(register) == expected
    assert state.get_register(register) == initial


def test_add_rsp_changes_return_slot():
    result = execute_gadget(gadget("48 83 c4 08 c3"), stack(123, 0x401000))
    assert result.get_register("rip") == 0x401000
    assert result.get_register("rsp") == 0x1010


def test_mov_rsp_changes_stack_base():
    state = stack(123)
    state.set_register("rdi", 0x2000)
    state.write_memory(0x2000, 0x401000)
    result = execute_gadget(gadget("48 89 fc c3"), state)
    assert result.get_register("rip") == 0x401000
    assert result.get_register("rsp") == 0x2008


def test_xor_self_and_distinct_registers():
    state = stack(0x401000)
    result = execute_gadget(gadget("48 31 c0 c3"), state)
    assert result.get_register("rax") == 0
    state.set_register("rax", WORD_MASK)
    state.set_register("rdi", 0xff)
    result = execute_gadget(gadget("48 31 f8 c3"), state)
    assert result.get_register("rax") == WORD_MASK ^ 0xff
    assert result.get_register("rdi") == 0xff


def test_leave_ret_pivots_stack():
    state = stack(0xdeadbeef)
    state.set_register("rbp", 0x2000)
    state.write_memory(0x2000, 0x3000)
    state.write_memory(0x2008, 0x401000)
    result = execute_gadget(gadget("c9 c3"), state)
    assert result.get_register("rbp") == 0x3000
    assert result.get_register("rip") == 0x401000
    assert result.get_register("rsp") == 0x2010
    assert state.get_register("rbp") == 0x2000
    assert state.get_register("rsp") == 0x1000


def test_leave_without_reading_old_rsp():
    state = CPUState()
    state.set_register("rbp", 0x2000)
    state.write_memory(0x2000, 0x3000)
    result = execute([Leave()], state)
    assert result.get_register("rsp") == 0x2008
    assert result.get_register("rbp") == 0x3000


@pytest.mark.parametrize("code", [
    "48 0f ac d8 01", "48 0f a5 d8", "48 0f ad d8", "0f 31", "0f 01 f9", "0f c7 f0",
    "67 48 8b 04 cf", "64 48 89 04 cf", "f3 0f bc c3", "f3 0f bd c3", "66 c9",
    "eb 00", "e8 00 00 00 00", "0f 05", "90", "48 0f c7 08",
    "f3 5f", "f2 48 89 f8", "40 66 5f", "67 5f",
])
def test_reject_unsupported_widths_forms_and_control_flow(code):
    with pytest.raises(UnsupportedInstruction) as error:
        lift_instruction(instruction(code))
    assert "0x4000" in str(error.value)


def test_unsupported_gadget_does_not_mutate_state():
    state = stack(123, 0x401000)
    state.set_register("rdi", 7)
    with pytest.raises(UnsupportedInstruction):
        execute_gadget(gadget("5f 48 87 07 c3"), state)
    assert state.get_register("rdi") == 7
    assert state.get_register("rsp") == 0x1000


def test_elf_to_state(elf_path):
    gadgets = GadgetScanner().scan(ELFImage.load(elf_path).executable_segments)
    target = next(g for g in gadgets if g.address == 0x401000 and g.bytes == b"\x5f\xc3")
    result = execute_gadget(target, stack(0x41414141, 0x401000))
    assert result.get_register("rdi") == 0x41414141
    assert result.get_register("rip") == 0x401000
    assert result.get_register("rsp") == 0x1010


def test_ir_is_independent_of_disassembly_text():
    decoded = instruction("48 89 f8")
    assert lift_instruction(replace(decoded, op_str="ignored display text")) == Mov("rax", "rdi")


def test_invalid_ir_and_early_return():
    with pytest.raises(ValueError, match="after return"):
        execute([Ret(), Pop("rdi")], stack(1, 2))
    with pytest.raises(TypeError, match="unsupported IR"):
        execute([object()], CPUState())
    with pytest.raises(InvalidRegister):
        execute([Pop("rip")], stack(1))


def test_opaque_transfer_but_not_arithmetic_or_address():
    token = ("opaque", 1)
    state = stack(token)
    result = execute([Pop("rdi"), Mov("rax", "rdi")], state)
    assert result.get_register("rax") is token
    with pytest.raises(NonConcreteValue):
        execute([Add("rax", 1)], result)
    state.set_register("rsp", token)
    with pytest.raises(NonConcreteValue):
        execute([Ret()], state)


def test_bad_gadget_records():
    original = gadget("5f c3")
    with pytest.raises(ValueError, match="layout"):
        lift_gadget(replace(original, address=0))
    with pytest.raises(ValueError, match="bytes"):
        lift_gadget(replace(original, bytes=b"\xc3"))
    with pytest.raises(ValueError, match="empty"):
        lift_gadget(replace(original, instructions=()))
    with pytest.raises(ValueError, match="exactly one return"):
        lift_gadget(replace(original, bytes=b"\x5f", instructions=original.instructions[:1]))
