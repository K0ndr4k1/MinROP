import random

import pytest

from minirop import CodeRegion, ELFImage, GadgetScanner, execute, execute_gadget
from minirop.ir import Add as AddOp, Leave, Pop, Ret
from minirop.semantics import UnsupportedInstruction
from minirop.state import CPUState, MemoryAccessError, REGISTER_NAMES, make_symbolic_initial_state
from minirop.symbolic import (
    Add, Constant, MemoryRead, NonConcreteValue, Sub, Symbol, WORD_MASK,
    add_values, evaluate, sub_values,
)


def gadget(code):
    data = bytes.fromhex(code)
    return next(g for g in GadgetScanner(max_gadget_bytes=64, max_instructions=16).scan(
        [CodeRegion("test", 0x4000, 0, data)]) if g.address == 0x4000 and g.bytes == data)


def test_initial_state_and_copy():
    state = make_symbolic_initial_state(2)
    for register in REGISTER_NAMES:
        assert state.get_register(register) == Symbol(f"{register}_0")
    other = state.copy()
    other.set_stack_slot(0, 42)
    other.set_register("rdi", 7)
    assert state.read_memory(Symbol("rsp_0")) == Symbol("stack_0")
    assert other.read_memory(Symbol("rsp_0")) == 42
    assert state.get_register("rdi") == Symbol("rdi_0")


@pytest.mark.parametrize("code, rdi, rip, rsp", [
    ("5f c3", Symbol("stack_0"), Symbol("stack_1"), Add(Symbol("rsp_0"), Constant(16))),
    ("5f 48 83 c7 08 c3", Add(Symbol("stack_0"), Constant(8)), Symbol("stack_1"), Add(Symbol("rsp_0"), Constant(16))),
])
def test_required_examples(code, rdi, rip, rsp):
    state = make_symbolic_initial_state(2)
    result = execute_gadget(gadget(code), state)
    assert result.get_register("rdi") == rdi
    assert result.get_register("rip") == rip
    assert result.get_register("rsp") == rsp
    assert state.get_register("rsp") == Symbol("rsp_0")


def test_pop_and_ret_individually():
    state = make_symbolic_initial_state()
    popped = execute([Pop("rdi")], state)
    assert popped.get_register("rdi") == Symbol("stack_0")
    assert popped.get_register("rsp") == Add(Symbol("rsp_0"), Constant(8))
    returned = execute([Ret()], state)
    assert returned.get_register("rip") == Symbol("stack_0")


def test_multiple_pop():
    result = execute_gadget(gadget("5f 5e c3"), make_symbolic_initial_state())
    assert result.get_register("rdi") == Symbol("stack_0")
    assert result.get_register("rsi") == Symbol("stack_1")
    assert result.get_register("rip") == Symbol("stack_2")
    assert result.get_register("rsp") == Add(Symbol("rsp_0"), Constant(24))


def test_mov_add_preserves_source():
    result = execute_gadget(gadget("48 89 f8 48 83 c0 08 c3"), make_symbolic_initial_state())
    assert result.get_register("rax") == Add(Symbol("rdi_0"), Constant(8))
    assert result.get_register("rdi") == Symbol("rdi_0")
    assert result.get_register("rip") == Symbol("stack_0")
    assert result.get_register("rsp") == Add(Symbol("rsp_0"), Constant(8))


def test_sub_and_xor():
    result = execute_gadget(gadget("48 83 e8 08 c3"), make_symbolic_initial_state())
    assert result.get_register("rax") == Sub(Symbol("rax_0"), Constant(8))
    result = execute_gadget(gadget("48 31 c0 c3"), make_symbolic_initial_state())
    assert result.get_register("rax") == 0


def test_pop_rsp_reads_new_base_without_incrementing_popped_value():
    state = make_symbolic_initial_state()
    popped = execute([Pop("rsp")], state)
    assert popped.get_register("rsp") == Symbol("stack_0")
    result = execute_gadget(gadget("5c c3"), state)
    assert result.get_register("rip") == MemoryRead(Symbol("stack_0"))
    assert result.get_register("rsp") == Add(Symbol("stack_0"), Constant(8))
    assert state.get_register("rsp") == Symbol("rsp_0")


def test_pop_rsp_can_return_to_known_stack_region():
    state = make_symbolic_initial_state(3)
    state.set_stack_slot(0, add_values(Symbol("rsp_0"), 16))
    result = execute_gadget(gadget("5c c3"), state)
    assert result.get_register("rip") == Symbol("stack_2")
    assert result.get_register("rsp") == Add(Symbol("rsp_0"), Constant(24))


def test_leave_ret_uses_old_rbp():
    state = make_symbolic_initial_state()
    result = execute_gadget(gadget("c9 c3"), state)
    assert result.get_register("rbp") == MemoryRead(Symbol("rbp_0"))
    assert result.get_register("rip") == MemoryRead(Add(Symbol("rbp_0"), Constant(8)))
    assert result.get_register("rsp") == Add(Symbol("rbp_0"), Constant(16))
    assert state.get_register("rbp") == Symbol("rbp_0")


def test_leave_with_frame_in_initial_stack():
    state = make_symbolic_initial_state(4)
    state.set_register("rbp", add_values(Symbol("rsp_0"), 8))
    result = execute_gadget(gadget("c9 c3"), state)
    assert result.get_register("rbp") == Symbol("stack_1")
    assert result.get_register("rip") == Symbol("stack_2")
    assert result.get_register("rsp") == Add(Symbol("rsp_0"), Constant(24))


def test_stack_offset_tracking_and_no_hidden_stack_cursor():
    state = make_symbolic_initial_state(3)
    state.set_stack_slot(-1, Symbol("below_stack"))
    result = execute_gadget(gadget("48 83 ec 08 c3"), state)
    assert result.get_register("rip") == Symbol("below_stack")
    assert result.get_register("rsp") == Symbol("rsp_0")
    result = execute_gadget(gadget("48 83 c4 10 c3"), state)
    assert result.get_register("rip") == Symbol("stack_2")
    state.set_register("rdi", Symbol("other_base"))
    result = execute_gadget(gadget("48 89 fc c3"), state)
    assert result.get_register("rip") == MemoryRead(Symbol("other_base"))


def test_unknown_symbolic_address_is_deferred_not_zero():
    state = make_symbolic_initial_state(1)
    for offset in (1, 8, -8):
        address = add_values(Symbol("rsp_0"), offset)
        assert state.read_memory(address) == MemoryRead(address)
    address = sub_values(Add(Symbol("rsp_0"), Constant(16)), 16)
    assert state.read_memory(address) == Symbol("stack_0")
    assert state.read_memory(Symbol("rbp_0")) == MemoryRead(Symbol("rbp_0"))
    with pytest.raises(MemoryAccessError):
        state.read_memory(0x1234)
    with pytest.raises(MemoryAccessError):
        state.write_memory(Symbol("rsp_0"), 4)


def test_concrete_memory_can_hold_symbolic_words():
    state = CPUState()
    state.set_register("rsp", Constant(0x1000))
    state.write_memory(0x1000, Symbol("input"))
    state.write_memory(0x1008, Symbol("target"))
    result = execute_gadget(gadget("5f 48 83 c7 08 c3"), state)
    assert result.get_register("rdi") == Add(Symbol("input"), Constant(8))
    assert result.get_register("rip") == Symbol("target")
    assert evaluate(result.get_register("rsp"), {}) == 0x1010


@pytest.mark.parametrize("slots", [-1, 1.5, True])
def test_bad_initial_stack_size(slots):
    with pytest.raises(ValueError):
        make_symbolic_initial_state(slots)


def test_stack_api_errors():
    with pytest.raises(MemoryAccessError):
        CPUState().set_stack_slot(0, 1)
    with pytest.raises(MemoryAccessError):
        make_symbolic_initial_state().set_stack_slot(1 << 60, 1)
    with pytest.raises(TypeError):
        CPUState(stack_base=123)


@pytest.mark.parametrize("code", ["0f a2 c3", "67 48 8b 04 cf c3", "f3 5f c3", "48 87 07 c3"])
def test_unsupported_subset_is_unchanged(code):
    state = make_symbolic_initial_state()
    with pytest.raises(UnsupportedInstruction):
        execute_gadget(gadget(code), state)
    assert state.get_register("rsp") == Symbol("rsp_0")


def test_invalid_value_failure_preserves_input():
    state = make_symbolic_initial_state()
    state.set_register("rax", ("opaque",))
    with pytest.raises(NonConcreteValue):
        execute([Pop("rdi"), AddOp("rax", 1)], state)
    assert state.get_register("rdi") == Symbol("rdi_0")
    assert state.get_register("rsp") == Symbol("rsp_0")


def test_elf_to_symbolic_state(elf_path):
    target = next(g for g in GadgetScanner().scan(ELFImage.load(elf_path).executable_segments)
                  if g.bytes == b"\x5f\xc3" and g.address == 0x401000)
    result = execute_gadget(target, make_symbolic_initial_state())
    assert result.get_register("rdi") == Symbol("stack_0")
    assert result.get_register("rip") == Symbol("stack_1")


# Every supported operation participates in real decoded gadgets. Deterministic
# edge/random inputs compare all registers, including preserved registers.
@pytest.mark.parametrize("code", [
    "c3", "5f c3", "5f 5e c3", "5f 48 83 c7 08 c3",
    "48 89 f8 c3", "48 89 f8 48 83 c0 08 c3",
    "48 c7 c0 ff ff ff ff c3", "48 b8 88 77 66 55 44 33 22 11 c3",
    "48 83 c0 08 c3", "48 83 e8 08 c3", "48 83 c0 ff c3",
    "48 31 c0 c3", "48 31 f8 c3", "5c c3", "c9 c3",
    "48 83 c4 08 c3", "48 83 ec 08 c3", "48 89 fc c3",
])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_concrete_symbolic_consistency(code, seed):
    rng = random.Random(seed)
    edge = [0, 1, WORD_MASK, 1 << 63, rng.getrandbits(64)][seed]
    bindings = {f"{reg}_0": rng.getrandbits(64) for reg in REGISTER_NAMES}
    bindings.update(rax_0=edge, rdi_0=0x4000, rsp_0=0x1000, rbp_0=0x2000)
    symbolic = make_symbolic_initial_state(4)
    symbolic.set_stack_slot(-1, Symbol("stack_minus_one"))
    bindings["stack_minus_one"] = 0x5050
    bindings.update({f"stack_{i}": rng.getrandbits(64) for i in range(4)})
    bindings["stack_0"] = 0x3000 if code == "5c c3" else edge
    memory = {0xff8: bindings["stack_minus_one"], 0x2000: 0x7000, 0x2008: 0x5555,
              0x3000: 0x6060, 0x4000: 0x8080}
    memory.update({0x1000 + i * 8: bindings[f"stack_{i}"] for i in range(4)})
    concrete = CPUState()
    for reg in REGISTER_NAMES:
        concrete.set_register(reg, bindings[f"{reg}_0"])
    for address, value in memory.items():
        concrete.write_memory(address, value)
    target = gadget(code)
    concrete_result = execute_gadget(target, concrete)
    symbolic_result = execute_gadget(target, symbolic)
    for reg in REGISTER_NAMES:
        assert evaluate(symbolic_result.get_register(reg), bindings, memory=memory) == concrete_result.get_register(reg)
        assert concrete.get_register(reg) == bindings[f"{reg}_0"]
        assert symbolic.get_register(reg) == Symbol(f"{reg}_0")
