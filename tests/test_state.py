from dataclasses import dataclass

import pytest

from minirop.state import CPUState, InvalidRegister, MemoryAccessError, REGISTER_NAMES, UninitializedRegister, WORD_MASK


@pytest.mark.parametrize("name", REGISTER_NAMES)
def test_register_read_write(name):
    state = CPUState()
    state.set_register(name.upper(), -1)
    assert state.get_register(name) == WORD_MASK
    state.set_register(name, 1 << 64)
    assert state.get_register(name) == 0


@pytest.mark.parametrize("name", ["eip", "ip", "r8h", "xmm0", "r16", ""])
def test_invalid_register(name):
    state = CPUState()
    with pytest.raises(InvalidRegister):
        state.set_register(name, 1)
    with pytest.raises(InvalidRegister):
        state.get_register(name)


def test_uninitialized():
    with pytest.raises(UninitializedRegister):
        CPUState().get_register("rax")
    with pytest.raises(MemoryAccessError):
        CPUState().read_memory(0x1000)


def test_memory_and_copy():
    state = CPUState()
    state.set_register("rsp", 0x1001)
    state.write_memory(0x1001, -1)
    state.write_memory(0x1009, 20)
    other = state.copy()
    other.set_register("rsp", 0x2000)
    other.write_memory(0x1001, 0)
    other.write_memory(0x2000, 1)
    assert state.get_register("rsp") == 0x1001
    assert state.read_memory(0x1001) == WORD_MASK
    assert other.read_memory(0x1001) == 0
    assert other.read_memory(0x1009) == 20
    with pytest.raises(MemoryAccessError):
        state.read_memory(0x2000)
    with pytest.raises(MemoryAccessError):
        state.write_memory(0x1002, 4)


@pytest.mark.parametrize("address", [-1, 1 << 64, True, "address"])
def test_invalid_memory_address(address):
    with pytest.raises(MemoryAccessError):
        CPUState().write_memory(address, 1)


def test_opaque_immutable_values():
    @dataclass(frozen=True)
    class Token:
        name: str

    value = Token("input")
    state = CPUState[Token]()
    state.set_register("rax", value)
    state.write_memory(0x1000, value)
    other = state.copy()
    assert other.get_register("rax") is value
    assert other.read_memory(0x1000) is value
