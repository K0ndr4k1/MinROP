"""Immutable 64-bit expressions and shared value operations; no solver.

Raw integers stay integers on the concrete path. Expression children are always
expressions (including Constant), and all operations use bitvector arithmetic.
"""

from collections.abc import Mapping
from dataclasses import dataclass

WORD_MASK = (1 << 64) - 1


class NonConcreteValue(TypeError):
    """A value is neither a supported expression nor a concrete integer.

    Historical name retained for callers of the Phase 2 executor.
    """


@dataclass(frozen=True, kw_only=True)
class Expression:
    bits: int = 64

    def __post_init__(self) -> None:
        if type(self.bits) is not int or self.bits != 64:
            raise ValueError("only 64-bit expressions are supported")

    def __str__(self) -> str:
        return format_value(self)


@dataclass(frozen=True)
class Symbol(Expression):
    name: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("symbol name must be a nonempty string")


@dataclass(frozen=True)
class Constant(Expression):
    value: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.value) is not int:
            raise TypeError("constant must be an integer")
        object.__setattr__(self, "value", self.value & WORD_MASK)


@dataclass(frozen=True)
class BinaryExpression(Expression):
    left: Expression
    right: Expression

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "left", as_expression(self.left))
        object.__setattr__(self, "right", as_expression(self.right))


@dataclass(frozen=True)
class Add(BinaryExpression):
    pass


@dataclass(frozen=True)
class Sub(BinaryExpression):
    pass


@dataclass(frozen=True)
class Xor(BinaryExpression):
    pass


@dataclass(frozen=True)
class And(BinaryExpression):
    pass


@dataclass(frozen=True)
class Or(BinaryExpression):
    pass


@dataclass(frozen=True)
class Shl(BinaryExpression):
    pass


@dataclass(frozen=True)
class Shr(BinaryExpression):
    pass


@dataclass(frozen=True)
class Sar(BinaryExpression):
    """Arithmetic right shift of a signed 64-bit carrier."""


ACCESS_WIDTHS = (8, 16, 32, 64)


def check_width(width: int) -> int:
    if type(width) is not int or width not in ACCESS_WIDTHS:
        raise ValueError("width must be 8, 16, 32 or 64 bits")
    return width


@dataclass(frozen=True)
class ProductPart(BinaryExpression):
    """Half of a full 2*width product, carried as a zero-extended 64-bit value."""

    width: int
    signed: bool
    high: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        check_width(self.width)
        if type(self.signed) is not bool or type(self.high) is not bool:
            raise TypeError("product signed/high selectors must be Boolean")


@dataclass(frozen=True)
class BitSlice(Expression):
    """A register view, zero-extended into the existing 64-bit value carrier."""

    value: Expression
    offset: int
    width: int

    def __post_init__(self) -> None:
        super().__post_init__()
        check_width(self.width)
        if type(self.offset) is not int or not 0 <= self.offset <= 64 - self.width:
            raise ValueError("invalid register slice")
        object.__setattr__(self, "value", as_expression(self.value))


def extract_value(value: "Value", width: int, offset: int = 0) -> "Value":
    check_width(width)
    if type(offset) is not int or not 0 <= offset <= 64 - width:
        raise ValueError("invalid register slice")
    if width == 64:
        return value
    if type(value) is int:
        return (value >> offset) & ((1 << width) - 1)
    expression = as_expression(value)
    if isinstance(expression, Constant):
        return Constant((expression.value >> offset) & ((1 << width) - 1))
    return BitSlice(expression, offset, width)


def _check_extension(source_width: int, target_width: int, signed: bool) -> None:
    check_width(source_width)
    check_width(target_width)
    if source_width >= target_width:
        raise ValueError("extension target width must exceed source width")
    if type(signed) is not bool:
        raise TypeError("extension signed flag must be bool")


@dataclass(frozen=True)
class BitExtend(Expression):
    """Extend to target_width, then zero-extend into the 64-bit value carrier."""

    value: Expression
    source_width: int
    target_width: int
    signed: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_extension(self.source_width, self.target_width, self.signed)
        object.__setattr__(self, "value", as_expression(self.value))


def _extend_integer(value: int, source_width: int, target_width: int, signed: bool) -> int:
    value &= (1 << source_width) - 1
    if signed and value & (1 << (source_width - 1)):
        value |= ((1 << target_width) - 1) ^ ((1 << source_width) - 1)
    return value


def _extend_value(value: "Value", source_width: int, target_width: int, signed: bool) -> "Value":
    _check_extension(source_width, target_width, signed)
    if type(value) is int:
        return _extend_integer(value, source_width, target_width, signed)
    expression = as_expression(value)
    if isinstance(expression, Constant):
        return Constant(_extend_integer(expression.value, source_width, target_width, signed))
    return BitExtend(expression, source_width, target_width, signed)


def zero_extend(value: "Value", source_width: int, target_width: int) -> "Value":
    return _extend_value(value, source_width, target_width, False)


def sign_extend(value: "Value", source_width: int, target_width: int) -> "Value":
    return _extend_value(value, source_width, target_width, True)


@dataclass(frozen=True)
class BoolExpression:
    """Solver-independent Boolean; never interpreted by Python truth testing."""

    def __bool__(self) -> bool:
        raise TypeError("symbolic Boolean cannot be tested by Python if")

    def __str__(self) -> str:
        return format_bool(self)


@dataclass(frozen=True)
class BoolConstant(BoolExpression):
    value: bool

    def __post_init__(self) -> None:
        if type(self.value) is not bool:
            raise TypeError("Boolean constant requires bool")


@dataclass(frozen=True)
class BoolSymbol(BoolExpression):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Boolean symbol name must be nonempty")


BoolValue = bool | BoolExpression


def as_bool(value: BoolValue) -> BoolExpression:
    if type(value) is bool:
        return BoolConstant(value)
    if isinstance(value, (BoolConstant, BoolSymbol, BoolNot, BoolBinary, BoolCompare, GuardedBool)):
        return value
    raise TypeError("flag requires bool or a Boolean expression")


@dataclass(frozen=True)
class BoolNot(BoolExpression):
    value: BoolExpression

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", as_bool(self.value))


@dataclass(frozen=True)
class BoolBinary(BoolExpression):
    operator: str
    left: BoolExpression
    right: BoolExpression

    def __post_init__(self) -> None:
        if self.operator not in ("and", "or", "xor"):
            raise ValueError("unsupported Boolean operator")
        object.__setattr__(self, "left", as_bool(self.left))
        object.__setattr__(self, "right", as_bool(self.right))


@dataclass(frozen=True)
class BoolCompare(BoolExpression):
    operator: str
    left: Expression
    right: Expression

    def __post_init__(self) -> None:
        if self.operator not in ("eq", "ult"):
            raise ValueError("unsupported bitvector comparison")
        object.__setattr__(self, "left", as_expression(self.left))
        object.__setattr__(self, "right", as_expression(self.right))


@dataclass(frozen=True)
class Ite(Expression):
    condition: BoolExpression
    when_true: Expression
    when_false: Expression

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "condition", as_bool(self.condition))
        object.__setattr__(self, "when_true", as_expression(self.when_true))
        object.__setattr__(self, "when_false", as_expression(self.when_false))


def bool_not(value: BoolValue) -> BoolValue:
    expression = as_bool(value)
    return not expression.value if isinstance(expression, BoolConstant) else BoolNot(expression)


def bool_binary(operator: str, left: BoolValue, right: BoolValue) -> BoolValue:
    expression = BoolBinary(operator, left, right)
    if isinstance(expression.left, BoolConstant) and isinstance(expression.right, BoolConstant):
        return evaluate_bool(expression, {})
    if expression.left == expression.right:
        return False if operator == "xor" else left
    return expression


def compare_values(operator: str, left: "Value", right: "Value") -> BoolValue:
    expression = BoolCompare(operator, left, right)
    if isinstance(expression.left, Constant) and isinstance(expression.right, Constant):
        return evaluate_bool(expression, {})
    if expression.left == expression.right:
        return operator == "eq"
    return expression


def select_value(condition: BoolValue, when_true: "Value", when_false: "Value") -> "Value":
    expression = as_bool(condition)
    if isinstance(expression, BoolConstant):
        return when_true if expression.value else when_false
    return Ite(expression, when_true, when_false)


def format_bool(value: BoolValue) -> str:
    expression = as_bool(value)
    if isinstance(expression, BoolConstant):
        return str(expression.value).lower()
    if isinstance(expression, BoolSymbol):
        return expression.name
    if isinstance(expression, GuardedBool):
        return f"defined_if({format_bool(expression.defined)}, {format_bool(expression.value)})"
    if isinstance(expression, BoolNot):
        return f"not({format_bool(expression.value)})"
    if isinstance(expression, BoolBinary):
        return f"({format_bool(expression.left)} {expression.operator} {format_bool(expression.right)})"
    return f"({format_value(expression.left)} {expression.operator} {format_value(expression.right)})"


def evaluate_bool(value: BoolValue, bindings: Mapping[str, int | bool], *,
                  memory: Mapping[int, "Value"] | None = None) -> bool:
    expression = as_bool(value)
    if isinstance(expression, BoolConstant):
        return expression.value
    if isinstance(expression, BoolSymbol):
        number = bindings[expression.name]
        if type(number) is bool or type(number) is int and number in (0, 1):
            return bool(number)
        raise TypeError("Boolean symbol binding must be bool or 0/1")
    if isinstance(expression, GuardedBool):
        if not evaluate_bool(expression.defined, bindings, memory=memory):
            raise UndefinedValueError("architecture-undefined flag")
        return evaluate_bool(expression.value, bindings, memory=memory)
    if isinstance(expression, BoolNot):
        return not evaluate_bool(expression.value, bindings, memory=memory)
    if isinstance(expression, BoolBinary):
        left = evaluate_bool(expression.left, bindings, memory=memory)
        right = evaluate_bool(expression.right, bindings, memory=memory)
        if expression.operator == "and":
            return left and right
        if expression.operator == "or":
            return left or right
        return left != right
    left = evaluate(expression.left, bindings, memory=memory)
    right = evaluate(expression.right, bindings, memory=memory)
    return left == right if expression.operator == "eq" else left < right


@dataclass(frozen=True)
class InitialMemory:
    """The input byte array M0, interpreted in the caller's solver context."""


@dataclass(frozen=True)
class MemoryStore:
    """Persistent width-sized little-endian write; no mutable shared map."""

    parent: "InitialMemory | MemoryStore"
    address: Expression
    value: Expression
    width: int = 64

    def __post_init__(self) -> None:
        check_width(self.width)
        if not isinstance(self.parent, (InitialMemory, MemoryStore)):
            raise TypeError("invalid memory parent")
        object.__setattr__(self, "address", as_expression(self.address))
        object.__setattr__(self, "value", as_expression(self.value))


MemoryState = InitialMemory | MemoryStore


@dataclass(frozen=True)
class MemoryRead(Expression):
    """Read width-sized bytes, zero-extended into the 64-bit value carrier."""

    address: Expression
    memory: MemoryState = InitialMemory()
    width: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        check_width(self.width)
        object.__setattr__(self, "address", as_expression(self.address))
        if not isinstance(self.memory, (InitialMemory, MemoryStore)):
            raise TypeError("invalid memory version")


def write_memory(memory: MemoryState, address: "Value", value: "Value", width: int = 64) -> MemoryStore:
    return MemoryStore(memory, as_expression(address), as_expression(value), check_width(width))


def read_memory(memory: MemoryState, address: "Value", width: int = 64) -> Expression:
    check_width(width)
    address = as_expression(address)
    # Exact, same-width read-after-write is a bitvector identity. Other
    # possible/partial aliases remain accesses to the shared byte array.
    if isinstance(memory, MemoryStore) and memory.address == address and memory.width == width:
        return as_expression(extract_value(memory.value, width))
    return MemoryRead(address, memory, width)


def write64(memory: MemoryState, address: "Value", value: "Value") -> MemoryStore:
    return write_memory(memory, address, value, 64)


def read64(memory: MemoryState, address: "Value") -> Expression:
    return read_memory(memory, address, 64)


Value = int | Expression


def as_expression(value: Value) -> Expression:
    if type(value) is int:
        return Constant(value)
    if isinstance(value, (Symbol, Constant, Add, Sub, Xor, And, Or, Shl, Shr, Sar, ProductPart, BitSlice, BitExtend, Ite, MemoryRead, IntegerValue, GuardedValue)):
        return value
    raise NonConcreteValue(f"unsupported value type: {type(value).__name__}")


def _offset(value: Expression) -> tuple[Expression, int]:
    """Peel constant offsets only; no general algebra or alias analysis."""
    if isinstance(value, (Add, Sub)) and isinstance(value.right, Constant):
        base, offset = _offset(value.left)
        delta = value.right.value if isinstance(value, Add) else -value.right.value
        return base, (offset + delta) & WORD_MASK
    return value, 0


def relative_offset(address: Expression, base: Expression) -> int | None:
    root, offset = _offset(address)
    if root != base:
        return None
    return offset if offset < (1 << 63) else offset - (1 << 64)


def add_values(left: Value, right: Value) -> Value:
    if type(left) is int and type(right) is int:
        return (left + right) & WORD_MASK
    left, right = as_expression(left), as_expression(right)
    if isinstance(left, Constant) and isinstance(right, Constant):
        return Constant(left.value + right.value)
    if isinstance(left, Constant) and left.value == 0:
        return right
    if isinstance(right, Constant):
        if right.value == 0:
            return left
        base, offset = _offset(left)
        offset = (offset + right.value) & WORD_MASK
        return base if offset == 0 else Add(base, Constant(offset))
    return Add(left, right)


def sub_values(left: Value, right: Value) -> Value:
    if type(left) is int and type(right) is int:
        return (left - right) & WORD_MASK
    left, right = as_expression(left), as_expression(right)
    if isinstance(left, Constant) and isinstance(right, Constant):
        return Constant(left.value - right.value)
    if isinstance(right, Constant) and right.value == 0:
        return left
    return Sub(left, right)


def xor_values(left: Value, right: Value) -> Value:
    if type(left) is int and type(right) is int:
        return (left ^ right) & WORD_MASK
    left, right = as_expression(left), as_expression(right)
    if left == right:
        return Constant(0)
    if isinstance(left, Constant) and isinstance(right, Constant):
        return Constant(left.value ^ right.value)
    if isinstance(left, Constant) and left.value == 0:
        return right
    if isinstance(right, Constant) and right.value == 0:
        return left
    return Xor(left, right)


def _bitwise_values(node: type[BinaryExpression], left: Value, right: Value) -> Value:
    a, b = as_expression(left), as_expression(right)
    if isinstance(a, Constant) and isinstance(b, Constant):
        value = evaluate(node(a, b), {})
        return value if type(left) is int and type(right) is int else Constant(value)
    return node(a, b)


def and_values(left: Value, right: Value) -> Value:
    return _bitwise_values(And, left, right)


def or_values(left: Value, right: Value) -> Value:
    return _bitwise_values(Or, left, right)


def shl_values(left: Value, count: int) -> Value:
    return _bitwise_values(Shl, left, count & 63)


def shr_values(left: Value, count: int) -> Value:
    return _bitwise_values(Shr, left, count & 63)


def sar_values(left: Value, count: int, width: int = 64) -> Value:
    check_width(width)
    if type(count) is not int:
        raise TypeError("SAR requires an integer count")
    count &= 63 if width == 64 else 31
    source = sign_extend(left, width, 64) if width < 64 else left
    return _bitwise_values(Sar, source, count)


def _product_parts(left: int, right: int, width: int, signed: bool) -> tuple[int, int]:
    mask = (1 << width) - 1
    left, right = left & mask, right & mask
    if signed:
        left = left - (1 << width) if left >> (width - 1) else left
        right = right - (1 << width) if right >> (width - 1) else right
    product = (left * right) & ((1 << (2 * width)) - 1)
    return product & mask, (product >> width) & mask


def multiply_values(left: Value, right: Value, width: int, *, signed: bool = False) -> tuple[Value, Value]:
    """Compute the full double-width product before extracting either half."""
    check_width(width)
    if type(signed) is not bool:
        raise TypeError("multiply signed selector must be Boolean")
    if type(left) is int and type(right) is int:
        return _product_parts(left, right, width, signed)
    left, right = as_expression(left), as_expression(right)
    if isinstance(left, Constant) and isinstance(right, Constant):
        low, high = _product_parts(left.value, right.value, width, signed)
        return Constant(low), Constant(high)
    return (ProductPart(left, right, width, signed, False),
            ProductPart(left, right, width, signed, True))


def format_value(value: Value | BoolValue) -> str:
    """Stable, explicitly parenthesized rendering; equality is structural."""
    if type(value) is bool or isinstance(value, BoolExpression):
        return format_bool(value)
    expression = as_expression(value)
    if isinstance(expression, Symbol):
        return expression.name
    if isinstance(expression, Constant):
        return hex(expression.value)
    if isinstance(expression, IntegerValue):
        return f"{expression.operation}{expression.width}(" + ", ".join(format_value(v) for v in expression.values) + ")"
    if isinstance(expression, GuardedValue):
        return f"defined_if({format_bool(expression.defined)}, {format_value(expression.value)})"
    if isinstance(expression, Ite):
        return f"if({format_bool(expression.condition)}, {format_value(expression.when_true)}, {format_value(expression.when_false)})"
    if isinstance(expression, BitSlice):
        return f"extract{expression.width}({format_value(expression.value)}, {expression.offset})"
    if isinstance(expression, BitExtend):
        mode = "sign" if expression.signed else "zero"
        return f"{mode}_extend{expression.source_width}to{expression.target_width}({format_value(expression.value)})"
    if isinstance(expression, ProductPart):
        mode, part = ("signed" if expression.signed else "unsigned"), ("high" if expression.high else "low")
        return f"{mode}_mul{expression.width}_{part}({format_value(expression.left)}, {format_value(expression.right)})"
    if isinstance(expression, MemoryRead):
        if isinstance(expression.memory, InitialMemory):
            return f"mem{expression.width}[{format_value(expression.address)}]"
        return f"read{expression.width}({format_memory(expression.memory)}, {format_value(expression.address)})"
    operator = {Add: "+", Sub: "-", Xor: "^", And: "&", Or: "|", Shl: "<<", Shr: ">>", Sar: ">>s"}[type(expression)]
    return f"({format_value(expression.left)} {operator} {format_value(expression.right)})"


def format_memory(memory: MemoryState) -> str:
    if isinstance(memory, InitialMemory):
        return "M0"
    return (f"write{memory.width}({format_memory(memory.parent)}, {format_value(memory.address)}, "
            f"{format_value(memory.value)})")


def evaluate(value: Value | BoolValue, bindings: Mapping[str, int | bool], *,
             memory: Mapping[int, Value] | None = None) -> int | bool:
    """Substitute concrete inputs for debugging/tests, raising for missing data.

    The input mapping contains word bindings for M0. Reads and persistent
    writes operate on individual bytes, wrapping addresses modulo 2**64.
    Missing input bytes raise; they never default to zero.
    """
    if type(value) is bool or isinstance(value, BoolExpression):
        return evaluate_bool(value, bindings, memory=memory)
    expression = as_expression(value)

    def word(number: int) -> int:
        if type(number) is not int:
            raise TypeError("evaluation requires concrete integer inputs")
        return number & WORD_MASK

    if isinstance(expression, Constant):
        return expression.value
    if isinstance(expression, Symbol):
        return word(bindings[expression.name])
    if isinstance(expression, IntegerValue):
        return _integer_value(expression.operation, tuple(evaluate(v, bindings, memory=memory) for v in expression.values), expression.width, expression.signed)
    if isinstance(expression, GuardedValue):
        if not evaluate_bool(expression.defined, bindings, memory=memory):
            raise UndefinedValueError("architecture-undefined integer result")
        return evaluate(expression.value, bindings, memory=memory)
    if isinstance(expression, Ite):
        branch = expression.when_true if evaluate_bool(expression.condition, bindings, memory=memory) else expression.when_false
        return evaluate(branch, bindings, memory=memory)
    if isinstance(expression, BitSlice):
        return (evaluate(expression.value, bindings, memory=memory) >> expression.offset) & ((1 << expression.width) - 1)
    if isinstance(expression, BitExtend):
        number = evaluate(expression.value, bindings, memory=memory)
        return _extend_integer(number, expression.source_width, expression.target_width, expression.signed)
    if isinstance(expression, MemoryRead):
        address = evaluate(expression.address, bindings, memory=memory)

        def byte_at(version: MemoryState, target: int) -> int:
            if isinstance(version, MemoryStore):
                written = evaluate(version.address, bindings, memory=memory)
                offset = (target - written) & WORD_MASK
                if offset < version.width // 8:
                    value = evaluate(version.value, bindings, memory=memory)
                    return (value >> (offset * 8)) & 255
                return byte_at(version.parent, target)
            if memory is not None:
                for base, value in memory.items():
                    if type(base) is not int or not 0 <= base <= WORD_MASK:
                        raise ValueError("input memory address must be a 64-bit integer")
                    offset = (target - base) & WORD_MASK
                    if offset < 8:
                        return (evaluate(value, bindings, memory=memory) >> (offset * 8)) & 255
            raise KeyError(f"input memory byte required at 0x{target:x}")

        return sum(byte_at(expression.memory, (address + i) & WORD_MASK) << (8 * i)
                   for i in range(expression.width // 8))
    left = evaluate(expression.left, bindings, memory=memory)
    right = evaluate(expression.right, bindings, memory=memory)
    if isinstance(expression, ProductPart):
        low, high = _product_parts(left, right, expression.width, expression.signed)
        return high if expression.high else low
    if isinstance(expression, Sar):
        signed = left - (1 << 64) if left & (1 << 63) else left
        return (signed >> right) & WORD_MASK
    if isinstance(expression, Add):
        return (left + right) & WORD_MASK
    if isinstance(expression, Sub):
        return (left - right) & WORD_MASK
    if isinstance(expression, Xor):
        return (left ^ right) & WORD_MASK
    if isinstance(expression, And):
        return left & right
    if isinstance(expression, Or):
        return left | right
    if isinstance(expression, Shl):
        return (left << right) & WORD_MASK if right < 64 else 0
    if isinstance(expression, Shr):
        return left >> right if right < 64 else 0
    raise NonConcreteValue(f"unsupported expression: {type(expression).__name__}")


class UndefinedValueError(ValueError):
    pass


@dataclass(frozen=True)
class GuardedValue(Expression):
    """A result usable only under its architectural definedness predicate."""
    value: Expression
    defined: BoolValue

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "value", as_expression(self.value))
        object.__setattr__(self, "defined", as_bool(self.defined))


def value_requirements(value):
    """Read obligations, including conditional operands of an ITE."""
    if isinstance(value, GuardedValue):
        yield value.defined
        yield from value_requirements(value.value)
    elif isinstance(value, Ite):
        yield from value_requirements(value.condition)
        for guard in value_requirements(value.when_true):
            yield bool_binary("or", bool_not(value.condition), guard)
        for guard in value_requirements(value.when_false):
            yield bool_binary("or", value.condition, guard)
    elif isinstance(value, GuardedBool):
        yield value.defined
        yield from value_requirements(value.value)
    elif isinstance(value, BoolCompare):
        yield from value_requirements(value.left)
        yield from value_requirements(value.right)
    elif isinstance(value, BoolNot):
        yield from value_requirements(value.value)
    elif isinstance(value, BoolBinary):
        yield from value_requirements(value.left)
        yield from value_requirements(value.right)
    elif isinstance(value, BinaryExpression):
        yield from value_requirements(value.left)
        yield from value_requirements(value.right)
    elif isinstance(value, (BitSlice, BitExtend)):
        yield from value_requirements(value.value)
    elif isinstance(value, IntegerValue):
        for operand in value.values:
            yield from value_requirements(operand)


@dataclass(frozen=True)
class IntegerValue(Expression):
    """Width-specific integer operations requiring a wider internal bitvector."""
    operation: str
    values: tuple[Expression, ...]
    width: int
    signed: bool = False

    def __post_init__(self):
        super().__post_init__()
        check_width(self.width)
        arities = {"divq": 3, "divr": 3, "divok": 3,
                   "shl": 2, "shr": 2, "sar": 2,
                   "rol": 2, "ror": 2, "rcl": 3, "rcr": 3,
                   "rcl_cf": 3, "rcr_cf": 3}
        if self.operation not in arities or len(self.values) != arities[self.operation]:
            raise ValueError("invalid width-specific integer operation")
        if type(self.signed) is not bool:
            raise TypeError("signed selector must be Boolean")
        object.__setattr__(self, "values", tuple(as_expression(v) for v in self.values))


def _signed_integer(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def _integer_value(operation, values, width, signed=False):
    mask = (1 << width) - 1
    if operation.startswith("div"):
        low, high, divisor = values
        dividend = ((high & mask) << width) | (low & mask)
        divisor &= mask
        if signed:
            dividend = _signed_integer(dividend, 2 * width)
            divisor = _signed_integer(divisor, width)
        if divisor == 0:
            return 0  # Only a total expression; divok excludes this execution.
        quotient = abs(dividend) // abs(divisor) if signed else dividend // divisor
        if signed and (dividend < 0) != (divisor < 0):
            quotient = -quotient
        remainder = dividend - quotient * divisor
        valid = (-(1 << (width - 1)) <= quotient < (1 << (width - 1))) if signed else (0 <= quotient <= mask)
        if operation == "divok":
            return int(valid)
        return (quotient if operation == "divq" else remainder) & mask
    value, count = values[:2]
    value &= mask
    count &= 63 if width == 64 else 31
    if operation in ("shl", "shr", "sar"):
        if operation == "shl":
            return (value << count) & mask
        return ((_signed_integer(value, width) if operation == "sar" else value) >> count) & mask
    through_carry = operation.startswith(("rcl", "rcr"))
    size = width + int(through_carry)
    ring = value | ((values[2] & 1) << width) if through_carry else value
    count %= size
    right = operation.startswith(("ror", "rcr"))
    if count:
        ring = ((ring >> count) | (ring << (size - count))) if right else ((ring << count) | (ring >> (size - count)))
    ring &= (1 << size) - 1
    return (ring >> width) & 1 if operation.endswith("_cf") else ring & mask


def integer_value(operation, values, width, *, signed=False):
    if operation.startswith("div") and (values[2] == 0 or isinstance(values[2], Constant) and values[2].value == 0):
        return 0  # Zero divisor is an unconditional #DE, even for a symbolic dividend.
    if all(type(v) is int or isinstance(v, Constant) for v in values):
        return _integer_value(operation, tuple(v.value if isinstance(v, Constant) else v for v in values), width, signed)
    return IntegerValue(operation, tuple(as_expression(v) for v in values), width, signed)


@dataclass(frozen=True)
class GuardedBool(BoolExpression):
    value: BoolValue
    defined: BoolValue

    def __post_init__(self):
        object.__setattr__(self, "value", as_bool(self.value))
        object.__setattr__(self, "defined", as_bool(self.defined))
