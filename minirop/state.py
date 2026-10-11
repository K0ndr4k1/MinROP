"""Registers, input bindings and persistent little-endian byte memory."""

from typing import Generic, TypeVar

from .flags import FLAG_NAMES, UNDEFINED, UndefinedFlag, ConditionalFlag, flag_name, read_flag

from .symbolic import (Constant, Expression, BoolValue, BoolSymbol, as_bool, InitialMemory, MemoryRead, MemoryState,
                       MemoryStore, Symbol, WORD_MASK, add_values,
                       evaluate, value_requirements, UndefinedValueError, BoolConstant, check_width, extract_value, and_values, or_values, shl_values,
                       read_memory as memory_read, relative_offset, write_memory as memory_write)
GPR_NAMES = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
             "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
REGISTER_NAMES = GPR_NAMES + ("rip",)
REGISTER_VIEWS = {name: (name, 64, 0) for name in REGISTER_NAMES}
for parent, dword, word, byte in (
    ("rax", "eax", "ax", "al"), ("rbx", "ebx", "bx", "bl"),
    ("rcx", "ecx", "cx", "cl"), ("rdx", "edx", "dx", "dl"),
    ("rsi", "esi", "si", "sil"), ("rdi", "edi", "di", "dil"),
    ("rbp", "ebp", "bp", "bpl"), ("rsp", "esp", "sp", "spl"),
    *((f"r{i}", f"r{i}d", f"r{i}w", f"r{i}b") for i in range(8, 16)),
):
    for name, width in ((dword, 32), (word, 16), (byte, 8)):
        REGISTER_VIEWS[name] = (parent, width, 0)
for high, parent in (("ah", "rax"), ("bh", "rbx"), ("ch", "rcx"), ("dh", "rdx")):
    REGISTER_VIEWS[high] = (parent, 8, 8)


def register_view(name: str) -> tuple[str, int, int]:
    try:
        return REGISTER_VIEWS[name.lower()]
    except (KeyError, AttributeError) as error:
        raise InvalidRegister(f"unsupported register: {name}") from error


Value = TypeVar("Value")


class InvalidRegister(ValueError):
    pass


class UninitializedRegister(ValueError):
    pass


class MemoryAccessError(ValueError):
    pass


class CPUState(Generic[Value]):
    """Integers wrap to 64 bits; other values must be immutable.

    Input word/stack bindings describe M0 and remain separate from instruction
    stores. Copies share immutable memory versions and duplicate input maps.
    Execution reads/writes eight little-endian bytes, including overlaps and
    wrapping addresses. Missing concrete bytes raise; symbolic reads are nodes.
    """

    def __init__(self, *, stack_base: Symbol | None = None) -> None:
        if stack_base is not None and not isinstance(stack_base, Symbol):
            raise TypeError("symbolic stack base must be a Symbol")
        self.execution_conditions: tuple[BoolValue, ...] = ()
        self.exception: str | None = None
        self._flags: dict[str, BoolValue | UndefinedFlag | ConditionalFlag] = {"AF": UNDEFINED}
        self._registers: dict[str, Value | int | Expression] = {}
        self._memory: dict[int, Value | int | Expression] = {}
        self._stack_base = stack_base
        self._stack_slots: dict[int, Value | int | Expression] = {}
        self._memory_state: MemoryState = InitialMemory()

    @staticmethod
    def _register(name: str) -> str:
        name = name.lower()
        if name not in REGISTER_NAMES:
            raise InvalidRegister(f"unsupported register: {name}")
        return name

    @staticmethod
    def _value(value: Value | int | Expression) -> Value | int | Expression:
        return value & WORD_MASK if isinstance(value, int) else value

    @staticmethod
    def _address(address: int) -> None:
        if type(address) is not int or not 0 <= address <= WORD_MASK:
            raise MemoryAccessError("memory requires a concrete 64-bit starting address")

    def get_register(self, name: str) -> Value | int | Expression:
        parent, width, offset = register_view(name)
        if parent not in self._registers:
            raise UninitializedRegister(f"uninitialized register: {parent}")
        value = extract_value(self._registers[parent], width, offset)
        for guard in value_requirements(value):
            self.require(guard, undefined=True)
        return value

    def set_register(self, name: str, value: Value | int | Expression) -> None:
        parent, width, offset = register_view(name)
        if width == 64:
            self._registers[parent] = self._value(value)
        elif width == 32:
            self._registers[parent] = extract_value(value, width)
        else:
            # 8/16-bit writes preserve all other physical bits, including AH.
            old = self.get_register(parent)
            mask = ((1 << width) - 1) << offset
            inserted = shl_values(extract_value(value, width), offset)
            self._registers[parent] = or_values(and_values(old, WORD_MASK ^ mask), inserted)

    def require(self, condition, *, undefined=False):
        if isinstance(condition, BoolConstant):
            condition = condition.value
        if condition is False and undefined:
            raise UndefinedValueError("architecture-undefined integer result")
        if condition is not True and condition not in self.execution_conditions:
            self.execution_conditions += (condition,)

    def get_flag(self, name: str) -> BoolValue:
        raw = self._flags.get(flag_name(name))
        if isinstance(raw, ConditionalFlag):
            self.require(raw.defined)
            return raw.value
        return read_flag(self._flags, name)

    def set_flag(self, name: str, value: BoolValue | UndefinedFlag | ConditionalFlag) -> None:
        name = flag_name(name)
        if isinstance(value, ConditionalFlag):
            as_bool(value.value)
            as_bool(value.defined)
        elif not isinstance(value, UndefinedFlag):
            as_bool(value)
        self._flags[name] = value

    def flag_bindings(self) -> tuple[tuple[str, BoolValue | UndefinedFlag | ConditionalFlag], ...]:
        """Raw immutable entries, retaining the explicit undefined marker."""
        return tuple(self._flags.items())

    @property
    def memory_state(self) -> MemoryState:
        """Current immutable memory version; input bindings always refer to M0."""
        return self._memory_state

    def read_memory(self, address: int | Expression, width: int = 64) -> Value | int | Expression:
        check_width(width)
        if isinstance(address, Constant):
            address = address.value
        if type(address) is int:
            self._address(address)
        elif not isinstance(address, Expression):
            raise MemoryAccessError("memory address must be a 64-bit integer or expression")
        if isinstance(self._memory_state, InitialMemory):
            if isinstance(address, Expression):
                if self._stack_base is not None:
                    offset = relative_offset(address, self._stack_base)
                    if offset is not None and offset in self._stack_slots:
                        return extract_value(self._stack_slots[offset], width)
                return MemoryRead(address, width=width)
            if address in self._memory:
                return extract_value(self._memory[address], width)
        value = memory_read(self._memory_state, address, width)
        if type(address) is int:
            if isinstance(value, Constant):
                return value.value
            try:
                return evaluate(value, {}, memory=self._memory)
            except KeyError as error:
                # A symbolic write can alias this address: retain the read.
                version = self._memory_state
                while isinstance(version, MemoryStore):
                    if not isinstance(version.address, Constant) or not isinstance(version.value, Constant):
                        return value
                    version = version.parent
                if any(isinstance(v, Expression) for v in self._memory.values()):
                    return value
                raise MemoryAccessError(str(error)) from error
        return value

    def store_memory(self, address: int | Expression, value: int | Expression, width: int = 64) -> None:
        """Instruction write: derive a new version without changing M0."""
        check_width(width)
        if type(address) is int:
            self._address(address)
        self._memory_state = memory_write(self._memory_state, address, value, width)

    def write_memory(self, address: int, value: Value | int | Expression) -> None:
        self._address(address)
        if not isinstance(self._memory_state, InitialMemory):
            raise MemoryAccessError("configure input memory before instruction stores")
        if any(other != address and min((other - address) & WORD_MASK,
                                        (address - other) & WORD_MASK) < 8
               for other in self._memory):
            raise MemoryAccessError("overlapping slots are not supported")
        self._memory[address] = self._value(value)

    def set_stack_slot(self, index: int, value: Value | int | Expression) -> None:
        """Bind an input stack word at initial RSP + index*8; not an instruction store."""
        if not isinstance(self._memory_state, InitialMemory):
            raise MemoryAccessError("configure input stack before instruction stores")
        if self._stack_base is None:
            raise MemoryAccessError("no symbolic stack base configured")
        if type(index) is not int or not -(1 << 60) <= index < (1 << 60):
            raise MemoryAccessError("stack index must fit a signed 64-bit byte offset")
        self._stack_slots[index * 8] = self._value(value)

    def copy(self) -> "CPUState[Value]":
        state: CPUState[Value] = CPUState(stack_base=self._stack_base)
        state.execution_conditions = self.execution_conditions
        state.exception = self.exception
        state._registers = self._registers.copy()
        state._flags = self._flags.copy()
        state._memory = self._memory.copy()
        state._stack_slots = self._stack_slots.copy()
        state._memory_state = self._memory_state
        return state

    def memory_bindings(self) -> tuple[tuple[int | Expression, Value | int | Expression], ...]:
        """Read-only snapshot of input word bindings for backend consumers.

        Uses the original stack base even after RSP changes. Values are shared
        immutable objects; later state writes cannot change this tuple.
        """
        bindings: list[tuple[int | Expression, Value | int | Expression]] = list(self._memory.items())
        if self._stack_base is not None:
            bindings.extend((add_values(self._stack_base, offset), value)
                            for offset, value in self._stack_slots.items())
        return tuple(bindings)

    def register_bindings(self) -> tuple[tuple[str, Value | int | Expression], ...]:
        """Return initialized registers without exposing the mutable dictionary."""
        return tuple(self._registers.items())

    def input_bindings(self) -> tuple:
        """Legacy input containers, preserving concrete/stack storage identity.

        Flags have a separate flag_bindings() interface; this four-item tuple
        is retained for existing consumers. Copies carry both interfaces.

        Values must be immutable, as for copy(). Unlike memory_bindings(), this
        also retains an empty symbolic stack's base and original slot offsets.
        """
        return (self.register_bindings(), tuple(self._memory.items()),
                self._stack_base, tuple(self._stack_slots.items()))


def make_symbolic_initial_state(stack_slots: int = 8) -> CPUState[Expression]:
    """Initialize registers, five independent Boolean flags, and named stack inputs."""
    if type(stack_slots) is not int or stack_slots < 0:
        raise ValueError("stack_slots must be a nonnegative integer")
    state: CPUState[Expression] = CPUState(stack_base=Symbol("rsp_0"))
    for name in FLAG_NAMES:
        state.set_flag(name, BoolSymbol(f"{name.lower()}_0"))
    for name in REGISTER_NAMES:
        state.set_register(name, Symbol(f"{name}_0"))
    for index in range(stack_slots):
        state.set_stack_slot(index, Symbol(f"stack_{index}"))
    return state
