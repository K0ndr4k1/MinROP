"""Conditional analysis of a caller-supplied sequence; no solver or search."""

from collections.abc import Iterable
from dataclasses import dataclass

from .constraints import Equality
from .flags import UndefinedFlag, ConditionalFlag, read_flag
from .executor import execute_gadget
from .gadget import Gadget
from .state import (CPUState, InvalidRegister, REGISTER_NAMES, UninitializedRegister, register_view,
                    make_symbolic_initial_state)
from .symbolic import BoolValue, Expression, InitialMemory, MemoryState, Value, WORD_MASK, extract_value


@dataclass(frozen=True)
class StateSnapshot:
    """Registers, flags, M0 bindings, and the immutable current memory version."""

    registers: tuple[tuple[str, Value], ...]
    memory: tuple[tuple[int | Expression, Value], ...]
    memory_state: MemoryState = InitialMemory()
    flags: tuple[tuple[str, BoolValue | UndefinedFlag | ConditionalFlag], ...] = ()

    execution_conditions: tuple[BoolValue, ...] = ()
    exception: str | None = None

    @classmethod
    def capture(cls, state: CPUState) -> "StateSnapshot":
        return cls(state.register_bindings(), state.memory_bindings(), state.memory_state, state.flag_bindings(), state.execution_conditions, state.exception)

    def get_register(self, name: str) -> Value:
        parent, width, offset = register_view(name)
        for register, value in self.registers:
            if register == parent:
                return extract_value(value, width, offset)
        raise UninitializedRegister(f"uninitialized register: {parent}")

    def get_flag(self, name: str) -> BoolValue:
        return read_flag(dict(self.flags), name)

    def flag_bindings(self) -> tuple:
        return self.flags

    def register_bindings(self) -> tuple[tuple[str, Value], ...]:
        return self.registers

    def memory_bindings(self) -> tuple[tuple[int | Expression, Value], ...]:
        return self.memory


@dataclass(frozen=True)
class SequenceStep:
    index: int
    gadget: Gadget
    before: StateSnapshot
    after: StateSnapshot


@dataclass(frozen=True)
class ControlFlowCheck:
    source_index: int | None  # None denotes optional initial RIP checking.
    target_index: int
    equality: Equality


@dataclass(frozen=True)
class SequenceEffect:
    """Conditional effects, not proof that this sequence can execute.

    A contradictory control-flow condition makes the path infeasible; its final
    state must not be described as a reachable result until solving succeeds.
    """

    initial_state: StateSnapshot
    steps: tuple[SequenceStep, ...]
    control_flow: tuple[ControlFlowCheck, ...]

    @property
    def final_state(self) -> StateSnapshot:
        return self.steps[-1].after

    @property
    def constraints(self) -> tuple[Equality, ...]:
        return tuple(check.equality for check in self.control_flow)


def analyze_sequence(gadgets: Iterable[Gadget], initial_state: CPUState | None = None, *,
                     check_entry: bool = False) -> SequenceEffect:
    """Execute exactly the given order, carrying each resulting state forward.

    By default invocation starts at the first gadget (as execute_gadget does).
    check_entry=True additionally constrains the supplied initial RIP. No return
    destination is overwritten to force a match and no final target is invented.
    """
    sequence = tuple(gadgets)
    if not sequence:
        raise ValueError("sequence must contain at least one gadget")
    # All pieces describe one static code image. Overlap is allowed only when
    # bytes agree, including repetitions and instruction-interior entry points.
    code: dict[int, int] = {}
    for gadget in sequence:
        if not isinstance(gadget, Gadget):
            raise TypeError("sequence entries must be Gadget objects")
        if type(gadget.address) is not int or not 0 <= gadget.address <= WORD_MASK or gadget.end_address > WORD_MASK + 1:
            raise ValueError("gadget code must fit the 64-bit address space")
        for offset, byte in enumerate(gadget.bytes):
            address = gadget.address + offset
            if address in code and code[address] != byte:
                raise ValueError(f"conflicting code bytes at 0x{address:x}")
            code[address] = byte

    state = make_symbolic_initial_state() if initial_state is None else initial_state.copy()
    initial = StateSnapshot.capture(state)
    before = initial
    steps = []
    checks = []
    if check_entry:
        checks.append(ControlFlowCheck(None, 0, Equality(state.get_register("rip"), sequence[0].address)))
    for index, gadget in enumerate(sequence):
        if index:
            equality = (Equality(state.get_register("rip"), gadget.address) if state.exception is None
                        else Equality(False, True))
            checks.append(ControlFlowCheck(index - 1, index, equality))
        if state.exception is None:
            state = execute_gadget(gadget, state)
        after = StateSnapshot.capture(state)
        steps.append(SequenceStep(index, gadget, before, after))
        before = after
    return SequenceEffect(initial, tuple(steps), tuple(checks))
