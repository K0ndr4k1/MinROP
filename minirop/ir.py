"""Small decoder-independent operations. No execution or Capstone objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pop:
    register: str


@dataclass(frozen=True)
class Ret:
    pass


@dataclass(frozen=True)
class Mov:
    dst: str
    src: str | int


@dataclass(frozen=True)
class Add:
    dst: str
    immediate: int


@dataclass(frozen=True)
class Sub:
    dst: str
    immediate: int


@dataclass(frozen=True)
class Xor:
    dst: str
    src: str | int


@dataclass(frozen=True)
class Leave:
    pass


@dataclass(frozen=True)
class Binary:
    dst: str
    src: str | int


@dataclass(frozen=True)
class AddRegister(Binary):
    pass


@dataclass(frozen=True)
class SubRegister(Binary):
    pass


@dataclass(frozen=True)
class And(Binary):
    pass


@dataclass(frozen=True)
class Or(Binary):
    pass


@dataclass(frozen=True)
class Xchg(Binary):
    pass


@dataclass(frozen=True)
class Shl(Binary):
    pass


@dataclass(frozen=True)
class Shr(Binary):
    pass


@dataclass(frozen=True)
class Sar(Binary):
    pass


@dataclass(frozen=True)
class Neg:
    dst: str


@dataclass(frozen=True)
class Inc:
    dst: str


@dataclass(frozen=True)
class Dec:
    dst: str


@dataclass(frozen=True)
class Not:
    dst: str


@dataclass(frozen=True)
class Address:
    """64-bit GPR address, or RIP-relative with a captured next instruction address."""

    base: str | None
    index: str | None = None
    scale: int = 1
    displacement: int = 0
    next_rip: int | None = None


@dataclass(frozen=True)
class Extend:
    """Register or memory source extended before the unified alias write."""

    dst: str
    src: str | Address
    source_width: int
    signed: bool


@dataclass(frozen=True)
class Lea:
    dst: str
    address: Address


@dataclass(frozen=True)
class Load:
    dst: str
    address: Address
    width: int = 64


@dataclass(frozen=True)
class Store:
    address: Address
    src: str | int
    width: int = 64


@dataclass(frozen=True)
class Push:
    src: str | int


@dataclass(frozen=True)
class Compare:
    left: str | Address
    right: str | int | Address
    width: int


@dataclass(frozen=True)
class Cmp(Compare):
    pass


@dataclass(frozen=True)
class Test(Compare):
    pass


@dataclass(frozen=True)
class Adc(Compare):
    """Same width-bearing operand pair; left is also the write destination."""


@dataclass(frozen=True)
class Sbb(Compare):
    pass


@dataclass(frozen=True)
class Multiply:
    """Implicit MUL/IMUL or explicit two/three-operand IMUL without a new IR framework."""

    src: str | Address
    width: int
    signed: bool = False
    dst: str | None = None
    immediate: int | None = None


@dataclass(frozen=True)
class Clc:
    pass


@dataclass(frozen=True)
class Stc:
    pass


@dataclass(frozen=True)
class Cmc:
    pass


@dataclass(frozen=True)
class Setcc:
    condition: str
    dst: str | Address


@dataclass(frozen=True)
class Cmovcc:
    condition: str
    dst: str
    src: str


@dataclass(frozen=True)
class Jcc:
    """Direct relative branch, resolved from this instruction's end address."""

    condition: str
    target: int
    fallthrough: int


@dataclass(frozen=True)
class IntegerOp:
    """Remaining finite integer forms, with explicit destination width."""
    mnemonic: str
    dst: str | Address | None = None
    src: str | int | Address | None = None
    width: int = 64


Operation = (Pop | Ret | Mov | Add | Sub | Xor | Leave | AddRegister |
             SubRegister | And | Or | Xchg | Shl | Shr | Neg | Extend | Lea | Load | Store | Push | Inc | Dec | Not | Cmp | Test | Setcc | Cmovcc | Jcc | Multiply | Adc | Sbb | Clc | Stc | Cmc | Sar | IntegerOp)
